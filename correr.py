# -*- coding: utf-8 -*-
"""
correr.py - Consola CLI para la Mesa de Pedidos.

Permite procesar mensajes de la bandeja, interactuar con el flujo Human-in-the-Loop,
reanudar hilos pausados por Gerencia Comercial y visualizar la topología del grafo.

Ejemplos de Uso:
    py correr.py --mensaje MSG-009          # Pedido estándar: se valida y confirma automáticamente
    py correr.py --mensaje MSG-013          # Pedido con exceso de crédito: se frena en interrupt()
    py correr.py --pendientes               # Lista pedidos esperando autorización gerencial
    py correr.py --mensaje MSG-013          # Si estaba frenado, lo RETOMA donde quedó
    py correr.py --mensaje MSG-013 --decision aprobar --razon "Cliente histórico de confianza"
    py correr.py --mensaje MSG-013 --reiniciar   # Descarta el historial previo y arranca de cero
    py correr.py --texto "Cuánto sale el flete a Salta?" --cuit 30-71455982-2
    py correr.py --todos                    # Procesa los 20 mensajes de bandeja.jsonl en lote
    py correr.py --grafo                    # Muestra el diagrama Mermaid del sistema

Demostración de Durabilidad (Checkpointer SQLite):
1. Ejecutá: `py correr.py --mensaje MSG-013`.
2. Cuando el sistema muestre la alerta de Gerencia Comercial, cerrá la terminal con Ctrl+C.
3. Abrí otra terminal o ejecutá: `py correr.py --pendientes`. Verás el pedido esperando.
4. Ejecutá: `py correr.py --mensaje MSG-013`. El grafo reanuda exactamente desde la aprobación,
   sin reprocesar clientes, clasificaciones ni validaciones previas.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from typing import Any, Optional

import config
from mesa_pedidos import datos, ejecucion
from mesa_pedidos.esquemas import MensajeEntrante
from mesa_pedidos.utilidades import pesos, porcentaje, tabla

ANCHO = 80


def _al_pasar(nodo: str, cambios: dict[str, Any]) -> None:
    """Callback de actualización en vivo: imprime cada nodo al finalizar su ejecución."""
    log_lineas = cambios.get("log") or [""]
    ultima_linea = log_lineas[-1] if log_lineas else ""
    detalle = ultima_linea.split("|")[-1].strip() if "|" in ultima_linea else ultima_linea
    print(f"  -> {nodo:<22} {detalle[: ANCHO - 28]}")


def _imprimir_resultado(mensaje_id: str, estado: dict[str, Any]) -> None:
    """Muestra el resumen final de la ejecución, validaciones, respuesta y bitácora."""
    ruta_final = estado.get("ruta", "(sin terminar)")
    print("\n" + "=" * ANCHO)
    print(f"  MENSAJE: {mensaje_id}   |   RUTA FINAL: {ruta_final}")
    print("=" * ANCHO)

    validacion = estado.get("validacion") or {}
    renglones = validacion.get("renglones") or []
    if renglones:
        filas = [
            [
                r.get("sku", ""),
                str(r.get("cantidad", 0)),
                pesos(r.get("precio_unitario", 0.0)),
                porcentaje(r.get("descuento_pct", 0.0)),
                pesos(r.get("subtotal_neto", 0.0)),
                "SI" if r.get("hay_stock") else f"NO (stock {r.get('stock_disponible', 0)})",
            ]
            for r in renglones
        ]
        print("\n  Items valorizados:")
        print("  " + tabla(filas, ["SKU", "Cant.", "Precio Lista", "Desc.", "Subtotal Neto", "Stock"]).replace("\n", "\n  "))

        neto = validacion.get("neto", 0.0)
        iva = validacion.get("iva", 0.0)
        flete = validacion.get("flete", 0.0)
        total = validacion.get("total", 0.0)
        print(f"\n  Neto: {pesos(neto)}  |  IVA: {pesos(iva)}  |  Flete: {pesos(flete)}  |  TOTAL: {pesos(total)}")

        alertas = validacion.get("alertas") or []
        if alertas:
            print("\n  Alertas comerciales:")
            for a in alertas:
                print(f"     ! {a}")

        observaciones = validacion.get("observaciones") or []
        if observaciones:
            print("\n  Observaciones:")
            for o in observaciones:
                print(f"     · {o}")

    respuesta = estado.get("respuesta")
    if respuesta:
        print("\n  Respuesta al cliente:")
        print("     " + str(respuesta).replace("\n", "\n     "))

    print("\n  Log de auditoria (reducer acumulativo por nodo):")
    for linea in estado.get("log", []):
        print(f"     {linea}")

    errores = estado.get("errores", []) or []
    if errores:
        print("\n  Errores registrados:")
        for err in errores:
            print(f"     ERROR: {err}")
    print()


def _mostrar_pedido_de_aprobacion(payload: dict[str, Any]) -> None:
    """Muestra la ficha detallada de un pedido retenido en interrupt()."""
    print("\n" + "!" * ANCHO)
    print(f"  APROBACION REQUERIDA - GERENCIA COMERCIAL  ({payload.get('mensaje_id', '')})")
    print("!" * ANCHO)
    print(f"  Cliente: {payload.get('razon_social', '')} (CUIT {payload.get('cuit', '')} · Cat. {payload.get('categoria', '')} · Mora {payload.get('mora_dias', 0)} dias)")
    print(
        f"  Credito: Limite {pesos(payload.get('limite_credito', 0.0))} | "
        f"Saldo {pesos(payload.get('saldo_actual', 0.0))} | "
        f"Disponible {pesos(payload.get('credito_disponible', 0.0))}"
    )
    print(
        f"  Pedido:  Neto {pesos(payload.get('neto', 0.0))} | "
        f"Desc. {porcentaje(payload.get('descuento_aplicado_pct', 0.0))} | "
        f"Total c/IVA {pesos(payload.get('total', 0.0))}"
    )
    
    alertas = payload.get("alertas", [])
    if alertas:
        print("  Alertas:")
        for a in alertas:
            print(f"    - {a}")

    renglones = payload.get("renglones", [])
    if renglones:
        filas = [
            [
                r.get("sku", ""),
                str(r.get("nombre", ""))[:32],
                str(r.get("cantidad", 0)),
                str(r.get("stock", 0)),
                pesos(r.get("subtotal_neto", 0.0)),
            ]
            for r in renglones
        ]
        print("\n  " + tabla(filas, ["SKU", "Producto", "Cant.", "Stock", "Subtotal Neto"]).replace("\n", "\n  "))


def _pedir_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Solicita la decisión interactiva por consola al operador humano."""
    _mostrar_pedido_de_aprobacion(payload)
    print("\n  (Podes cerrar la terminal con Ctrl+C: el pedido queda guardado en checkpoints.sqlite y se retoma despues.)\n")

    mapa_decisiones = {
        "a": "aprobar",
        "aprobar": "aprobar",
        "e": "editar",
        "editar": "editar",
        "r": "rechazar",
        "rechazar": "rechazar",
    }

    tipo = None
    while not tipo:
        try:
            entrada = input("  Decision ([A]probar / [E]ditar / [R]echazar): ").strip().lower()
            tipo = mapa_decisiones.get(entrada)
        except EOFError:
            tipo = "rechazar"

    decision: dict[str, Any] = {"tipo": tipo}

    if tipo == "editar":
        try:
            crudo = input("  Cantidades nuevas (SKU=cant,SKU=cant; 0 elimina renglon; vacio = igual): ").strip()
            if crudo:
                items_editados: dict[str, int] = {}
                for parte in crudo.split(","):
                    if "=" in parte:
                        sku, cant = parte.split("=", 1)
                        sku_limpio = sku.strip().upper()
                        if sku_limpio:
                            try:
                                items_editados[sku_limpio] = int(cant.strip())
                            except ValueError:
                                pass
                decision["items_editados"] = items_editados
                decision["ediciones"] = items_editados

            desc = input("  Descuento autorizado en % (vacio = mantener escala): ").strip()
            if desc:
                decision["descuento_autorizado"] = float(desc.replace(",", "."))
        except (EOFError, ValueError):
            pass

    try:
        decision["razon"] = input("  Razon o justificacion (queda en auditoria): ").strip()
    except EOFError:
        decision["razon"] = "Decision ingresada via consola"

    return decision


def procesar(
    grafo: Any,
    mensaje: MensajeEntrante,
    retriever: Optional[str] = None,
    decision: Optional[dict[str, Any]] = None,
    reiniciar: bool = False,
    interactivo: bool = True,
) -> dict[str, Any]:
    """Procesa un mensaje completo gestionando de forma transparente las pausas por aprobación."""
    thread_id = ejecucion.thread_de(mensaje.id)
    cfg = ejecucion.config_de(thread_id, retriever)
    previo = ejecucion.estado_de(grafo, cfg)

    if previo["pendiente"] and not reiniciar:
        print(f"\n  {mensaje.id} estaba pausado esperando aprobacion: se retoma directamente sin reprocesar.")
        resultado: dict[str, Any] = {"estado": previo["estado"], "pendiente": previo["pendiente"]}
    else:
        if previo["estado"] and reiniciar:
            ejecucion.borrar_thread(grafo, thread_id)
        print(f"\n  Procesando {mensaje.id} [{mensaje.canal.upper()} · {mensaje.remitente}]:")
        print(f"  \"{mensaje.texto[:140]}...\"\n")
        resultado = ejecucion.iniciar(grafo, mensaje, cfg, _al_pasar, origen="cli")

    while resultado.get("pendiente"):
        payload_pendiente = resultado["pendiente"]
        if decision is None and not interactivo:
            _mostrar_pedido_de_aprobacion(payload_pendiente)
            print(f"\n  El pedido queda en pausa. Para retomarlo ejecuta:\n     py correr.py --mensaje {mensaje.id}\n")
            return resultado["estado"]

        d = decision or _pedir_decision(payload_pendiente)
        decision = None
        print(f"\n  Reanudando ejecucion con decision: {d.get('tipo', '').upper()}...")
        resultado = ejecucion.retomar(grafo, d, cfg, _al_pasar, origen="cli")

    return resultado.get("estado", {})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mesa de Pedidos - Ferreteria Industrial Andina (CLI Oficial)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mensaje", help="ID del mensaje en la bandeja (ej: MSG-009, MSG-013)")
    parser.add_argument("--texto", help="Mensaje libre escrito a mano (usar con --cuit)")
    parser.add_argument("--cuit", default="30-71455982-2", help="CUIT del cliente para mensajes libres (def: 30-71455982-2)")
    parser.add_argument("--todos", action="store_true", help="Procesa los 20 mensajes de la bandeja en lote")
    parser.add_argument("--pendientes", action="store_true", help="Lista los pedidos esperando autorizacion comercial")
    parser.add_argument("--decision", choices=["aprobar", "editar", "rechazar"], help="Aplica la decision sin preguntar")
    parser.add_argument("--razon", default="", help="Motivo de la decision (usar con --decision)")
    parser.add_argument("--reiniciar", action="store_true", help="Descarta el estado previo y empieza de cero")
    parser.add_argument("--retriever", choices=["denso", "bm25", "hibrido"], help="Modo del RAG (denso, bm25, hibrido)")
    parser.add_argument("--grafo", action="store_true", help="Imprime la representacion del grafo en sintaxis Mermaid")
    args = parser.parse_args()

    from mesa_pedidos.grafo import construir_grafo, dibujar

    if args.grafo:
        print(dibujar())
        return 0

    grafo = construir_grafo()

    if args.pendientes:
        lista_pendientes = ejecucion.pendientes(grafo)
        if not lista_pendientes:
            print("\n  No hay pedidos esperando aprobacion gerencial.")
            return 0
        print("\n" + "=" * ANCHO)
        print("  PEDIDOS PENDIENTES DE APROBACION GERENCIAL")
        print("=" * ANCHO)
        for thread_id, p in lista_pendientes:
            alertas_count = len(p.get("alertas", []))
            print(
                f"  {thread_id:<22} {p.get('razon_social', ''):<34} "
                f"Neto: {pesos(p.get('neto', 0.0)):>14}  ({alertas_count} alertas)"
            )
        print("\n  Para reanudar un pedido: py correr.py --mensaje <ID>")
        return 0

    config.verificar_entorno()
    decision = {"tipo": args.decision, "razon": args.razon} if args.decision else None

    if args.todos:
        bandeja = datos.cargar_bandeja()
        print(f"\n  Procesando {len(bandeja)} mensajes de la bandeja...")
        for m in bandeja:
            try:
                estado = procesar(grafo, m, args.retriever, reiniciar=True, interactivo=False)
                _imprimir_resultado(m.id, estado)
            except Exception as exc:
                print(f"  {m.id}: ERROR -> {type(exc).__name__}: {exc}")
        return 0

    if args.texto:
        mensaje = MensajeEntrante(
            id=f"LIBRE-{datetime.now():%m%d-%H%M%S}",
            canal="web",
            fecha=str(date.today()),
            cuit=args.cuit,
            remitente="Consola CLI",
            texto=args.texto,
        )
    elif args.mensaje:
        mensaje = datos.buscar_mensaje(args.mensaje)
        if mensaje is None:
            print(f"\n  Error: No se encontro el mensaje '{args.mensaje}' en la bandeja.")
            return 1
    else:
        parser.print_help()
        return 1

    estado = procesar(
        grafo=grafo,
        mensaje=mensaje,
        retriever=args.retriever,
        decision=decision,
        reiniciar=args.reiniciar or bool(args.texto),
        interactivo=True,
    )
    _imprimir_resultado(mensaje.id, estado)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n  Operacion interrumpida por el usuario (Ctrl+C).")
        print("  Si un pedido quedo en pausa esperando aprobacion, su estado quedo persistido en SQLite.")
        print("  Consulta los pedidos pendientes con: py correr.py --pendientes\n")
        sys.exit(130)
