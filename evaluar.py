# -*- coding: utf-8 -*-
"""
evaluar.py - Evaluación formal del sistema contra el Golden Set (Fase 6).

Este módulo ejecuta el benchmark cuantitativo completo del sistema:
- Procesa los 20 mensajes de la bandeja (MSG-001 a MSG-020).
- Compara los resultados contra las respuestas esperadas de `datos/golden_set.json`.
- Gestiona pausas de throttling configurables para respetar los límites de tasa (RPM) de Google Gemini.
- Soporta modos para evaluar casos individuales, subconjuntos (consultas/pedidos) o comparar retrievers.
- Persiste el reporte detallado en `artefactos/salidas/evaluacion.json`.

Métricas evaluadas:
  1. Ruteo Correcto: De los 20 casos, cuántos alcanzaron la hoja terminal esperada.
  2. Presencia de Dato Clave: De las 8 consultas, en cuántas la respuesta contiene el dato relevante.
  3. Intención Correcta: Acierto en la clasificación semántica (consulta, pedido, derivar).
  4. Documento Citado: De las consultas, en cuántas se citó el PDF de políticas correcto.
  5. Extracción de Ítems: Precisión y Recall en los 9 pedidos (SKUs y cantidades extraídas).

Nota Arquitectónica (Trampa del Interrupt):
  En los casos que requieren aprobación humana (ej: MSG-013, MSG-016), el sistema debe
  pausar su ejecución. Si reanudamos con aprobación y evaluamos el estado final, la ruta
  será 'confirmar_pedido'. Por ello, la evaluación anota 'hubo_interrupt' ANTES de
  reanudar para medir si el sistema supo autónomamente cuándo debía consultar a Gerencia.

Uso:
  python evaluar.py                     # Ejecuta los 20 casos con el retriever configurado
  python evaluar.py --caso MSG-001      # Ejecuta un caso individual para pruebas rápidas
  python evaluar.py --consultas         # Evalúa únicamente las 8 consultas RAG
  python evaluar.py --pedidos           # Evalúa únicamente los 9 pedidos comerciales
  python evaluar.py --comparar          # Compara denso vs BM25 vs híbrido en las consultas
  python evaluar.py --pausa 3.0         # Ajusta el intervalo de throttling en segundos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from typing import Any, Optional

# Asegura compatibilidad de caracteres en consolas Windows (cp1252 / UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
from mesa_pedidos import datos, ejecucion
from mesa_pedidos.datos import normalizar_texto
from mesa_pedidos.utilidades import tabla


# ===========================================================================
# 1. Comparadores y Métricas Cuantitativas
# ===========================================================================

def comparar_items(esperados: list[dict[str, Any]], obtenidos: list[dict[str, Any]]) -> tuple[float, float]:
    """Calcula precisión y recall de la extracción estructurada de pedidos.
    
    Un ítem es considerado un acierto únicamente si coinciden exactamente el SKU Y la cantidad.
    Extraer el SKU correcto con cantidad equivocada es un pedido mal facturado.
    
    Retorna: (precision, recall)
    """
    def clave(i: dict[str, Any]) -> tuple[str, int]:
        return (str(i.get("sku") or "").upper().strip(), int(i.get("cantidad") or 0))

    esp = {clave(i) for i in esperados}
    obt = {clave(i) for i in obtenidos if i.get("sku")}
    
    if not esp:
        return (1.0, 1.0) if not obt else (0.0, 1.0)
        
    aciertos = len(esp & obt)
    precision = (aciertos / len(obt)) if obt else 0.0
    recall = aciertos / len(esp)
    return precision, recall


def contiene_todo(respuesta: str, requeridos: list[str]) -> bool:
    """Verifica si la respuesta contiene cada uno de los términos requeridos.
    
    Aplica coincidencia por palabra entera sobre texto normalizado (sin tildes),
    evitando que fragmentos cortos (ej. 'no') coincidan erróneamente dentro de
    palabras más largas como 'bueno' o 'norma'.
    """
    r = normalizar_texto(respuesta)
    return all(
        re.search(rf"(?<!\w){re.escape(normalizar_texto(x))}(?!\w)", r)
        for x in requeridos
    )


# ===========================================================================
# 2. Evaluación de un Caso Individual
# ===========================================================================

def evaluar_caso(
    caso: dict[str, Any],
    grafo: Any,
    retriever: str,
    aprobar: bool = True,
) -> dict[str, Any]:
    """Ejecuta un caso del Golden Set a través del grafo y compara los resultados."""
    mensaje = datos.buscar_mensaje(caso["id"])
    if not mensaje:
        raise ValueError(f"No se encontró el mensaje con ID '{caso['id']}' en la bandeja.")

    cfg = ejecucion.config_de(f"eval-{retriever}-{caso['id']}", retriever)
    
    # 1. Ejecutar hasta el final o hasta el interrupt
    resultado = ejecucion.iniciar(grafo, mensaje, cfg, origen="eval")
    
    # 2. Registrar si hubo interrupción humana ANTES de reanudar
    hubo_interrupt = resultado["pendiente"] is not None
    
    # 3. Reanudar con aprobación si corresponde para completar el ciclo
    if hubo_interrupt and aprobar:
        resultado = ejecucion.retomar(
            grafo,
            {"tipo": "aprobar", "razon": "Aprobación automática en benchmark de evaluación"},
            cfg,
            origen="eval",
        )
        
    estado = resultado["estado"]
    obtenida = "aprobacion_humana" if hubo_interrupt else estado.get("ruta", "(sin ruta)")

    fila: dict[str, Any] = {
        "id": caso["id"],
        "esperada": caso["ruta"],
        "obtenida": obtenida,
        "ruta_ok": obtenida == caso["ruta"],
        "intencion_esperada": caso["intencion"],
        "intencion_obtenida": (estado.get("clasificacion") or {}).get("intencion"),
        "intencion_ok": (estado.get("clasificacion") or {}).get("intencion") == caso["intencion"],
        "respuesta": estado.get("respuesta", ""),
        "log": estado.get("log", []),
    }

    # Evaluación de consultas RAG
    if caso.get("debe_contener"):
        fila["dato_ok"] = contiene_todo(fila["respuesta"], caso["debe_contener"])
        documentos_citados = [c["documento"] for c in estado.get("citas", [])]
        fila["doc_ok"] = caso.get("documento") in documentos_citados
        fila["citas"] = documentos_citados

    # Evaluación de extracción de pedidos
    if caso.get("items"):
        items_extraidos = (estado.get("pedido") or {}).get("items", [])
        p, r = comparar_items(caso["items"], items_extraidos)
        fila["precision"] = p
        fila["recall"] = r
        fila["items_esperados"] = caso["items"]
        fila["items_obtenidos"] = items_extraidos

    return fila


# ===========================================================================
# 3. Resumen y Formateo de Resultados
# ===========================================================================

def _si_no(valor: Optional[bool]) -> str:
    """Devuelve un formato visual limpio para valores booleanos en la tabla."""
    return "-" if valor is None else ("OK" if valor else "FALLA")


def resumir(filas: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula las métricas consolidadas del lote de evaluación."""
    total = len(filas)
    consultas = [f for f in filas if "dato_ok" in f]
    pedidos = [f for f in filas if "precision" in f]

    ruteo_aciertos = sum(1 for f in filas if f["ruta_ok"])
    dato_aciertos = sum(1 for f in consultas if f["dato_ok"])
    intencion_aciertos = sum(1 for f in filas if f["intencion_ok"])
    doc_aciertos = sum(1 for f in consultas if f.get("doc_ok"))

    items_p_prom = (sum(f["precision"] for f in pedidos) / len(pedidos)) if pedidos else None
    items_r_prom = (sum(f["recall"] for f in pedidos) / len(pedidos)) if pedidos else None

    return {
        "total_casos": total,
        "ruteo": f"{ruteo_aciertos}/{total} ({ruteo_aciertos / total * 100:.1f}%)" if total else "-",
        "ruteo_pct": round(ruteo_aciertos / total * 100, 1) if total else 0.0,
        "dato_clave": f"{dato_aciertos}/{len(consultas)} ({dato_aciertos / len(consultas) * 100:.1f}%)" if consultas else "-",
        "dato_clave_pct": round(dato_aciertos / len(consultas) * 100, 1) if consultas else 0.0,
        "intencion": f"{intencion_aciertos}/{total} ({intencion_aciertos / total * 100:.1f}%)" if total else "-",
        "documento_citado": f"{doc_aciertos}/{len(consultas)} ({doc_aciertos / len(consultas) * 100:.1f}%)" if consultas else "-",
        "items_precision": round(items_p_prom, 3) if items_p_prom is not None else None,
        "items_recall": round(items_r_prom, 3) if items_r_prom is not None else None,
    }


def imprimir_reporte(filas: list[dict[str, Any]], resumen: dict[str, Any]) -> None:
    """Imprime en consola una tabla comparativa con el detalle de cada caso."""
    encabezados = ["Caso", "Ruta Esperada", "Ruta Obtenida", "Ruta", "Dato Clave", "Doc Citado", "Items P/R"]
    tabla_datos = []
    
    for f in filas:
        pr_str = f"{f['precision']:.2f}/{f['recall']:.2f}" if "precision" in f else "-"
        tabla_datos.append([
            f["id"],
            f["esperada"],
            f["obtenida"],
            _si_no(f["ruta_ok"]),
            _si_no(f.get("dato_ok")),
            _si_no(f.get("doc_ok")),
            pr_str,
        ])

    print("\n" + "=" * 90)
    print("RESULTADOS DETALLADOS DE EVALUACION (GOLDEN SET)")
    print("=" * 90)
    print(tabla(tabla_datos, encabezados))
    
    print("\n" + "-" * 90)
    print("METRICAS GLOBALES DEL BENCHMARK:")
    print(f"  1. Ruteo Correcto:           {resumen['ruteo']}")
    print(f"  2. Consultas con Dato Clave: {resumen['dato_clave']}")
    print(f"  3. Intencion Correcta:       {resumen['intencion']}")
    print(f"  4. Documento Correcto:       {resumen['documento_citado']}")
    if resumen["items_precision"] is not None:
        print(f"  5. Extraccion de Items:      Precision {resumen['items_precision']:.3f} | Recall {resumen['items_recall']:.3f}")
    print("-" * 90)

    # Listar discrepancias si existieran
    fallas = [f for f in filas if not f["ruta_ok"] or f.get("dato_ok") is False or f.get("doc_ok") is False]
    if fallas:
        print("\n[!] CASOS CON DISCREPANCIAS O FALLAS:")
        for f in fallas:
            motivos = []
            if not f["ruta_ok"]:
                motivos.append(f"Ruta esperada '{f['esperada']}' != obtenida '{f['obtenida']}'")
            if f.get("dato_ok") is False:
                motivos.append("Dato clave no encontrado en respuesta")
            if f.get("doc_ok") is False:
                motivos.append("Documento de politica citado incorrecto")
            print(f"  * [{f['id']}]: {'; '.join(motivos)}")
            print(f"    Respuesta completa: {f['respuesta']!r}")
    else:
        print("\n[OK] 100% de efectividad en las metricas evaluadas.")


def guardar_reporte(filas: list[dict[str, Any]], resumen: dict[str, Any], retriever: str) -> Path:
    """Guarda el informe de evaluación estructurado exclusivamente en artefactos/salidas/."""
    reporte = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo_llm": config.MODELO_LLM,
        "modelo_embeddings": config.MODELO_EMBEDDINGS,
        "retriever": retriever,
        "pesos_hibrido": config.PESOS_HIBRIDO if retriever == "hibrido" else None,
        "chunking": config.ESTRATEGIA_CHUNKING,
        "chunk_size": config.CHUNK_SIZE,
        "k_recuperacion": config.K_RECUPERACION,
        "resumen": resumen,
        "casos": filas,
    }

    es_completo = len(filas) >= 20

    if es_completo:
        ruta_salida = datos.guardar_salida("evaluacion.json", reporte)
        ruta_versionada = datos.guardar_salida(f"evaluacion_{retriever}.json", reporte)
        print(f"\n[OK] Reporte oficial completo (20 casos) persistido en:")
        print(f"   -> {ruta_salida.resolve()}")
        print(f"   -> {ruta_versionada.resolve()}")
        return ruta_salida
    else:
        ruta_parcial = datos.guardar_salida(f"evaluacion_parcial_{retriever}.json", reporte)
        print(f"\nReporte parcial ({len(filas)} casos) persistido en:")
        print(f"   -> {ruta_parcial.resolve()}")
        return ruta_parcial


# ===========================================================================
# 4. Motor de Ejecución en Lote con Throttling
# ===========================================================================

def correr_evaluacion(
    casos: list[dict[str, Any]],
    retriever: str,
    aprobar: bool = True,
    pausa_segundos: float = 2.0,
    mostrar_progreso: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ejecuta secuencialmente una lista de casos con pausas de control de tasa."""
    from mesa_pedidos.grafo import construir_grafo

    # Para la evaluación utilizamos checkpointer en memoria para aislar cada caso
    grafo = construir_grafo(persistente=False)
    filas: list[dict[str, Any]] = []
    total = len(casos)

    for i, caso in enumerate(casos, start=1):
        if mostrar_progreso:
            print(f"  [{i:>2}/{total}] Evaluando {caso['id']} ({caso['intencion']}) ... ", end="", flush=True)

        try:
            fila = evaluar_caso(caso, grafo, retriever, aprobar=aprobar)
            estado_str = "OK" if fila["ruta_ok"] else f"FALLA (fue a {fila['obtenida']})"
            if mostrar_progreso:
                print(estado_str)
        except Exception as exc:
            fila = {
                "id": caso["id"],
                "esperada": caso["ruta"],
                "obtenida": f"ERROR: {type(exc).__name__}",
                "ruta_ok": False,
                "intencion_esperada": caso["intencion"],
                "intencion_obtenida": None,
                "intencion_ok": False,
                "respuesta": f"Excepción durante la ejecución: {exc}",
                "log": [],
            }
            if mostrar_progreso:
                print(f"ERROR: {exc}")

        filas.append(fila)

        # Throttling de cortesía para respetar límites de tasa (RPM) de Google Gemini
        if i < total and pausa_segundos > 0:
            time.sleep(pausa_segundos)

    return filas, resumir(filas)


# ===========================================================================
# 5. Entrada Principal (CLI)
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Suite de Evaluación Formal contra el Golden Set (Fase 6)"
    )
    parser.add_argument(
        "--retriever",
        choices=["denso", "bm25", "hibrido"],
        default=config.RETRIEVER,
        help=f"Estrategia de recuperación para RAG (por defecto: {config.RETRIEVER})",
    )
    parser.add_argument(
        "--caso",
        type=str,
        default=None,
        help="Evalúa un caso específico (ej: MSG-001, MSG-013)",
    )
    parser.add_argument(
        "--consultas",
        action="store_true",
        help="Evalúa únicamente las 8 consultas de políticas (MSG-001 a MSG-008)",
    )
    parser.add_argument(
        "--pedidos",
        action="store_true",
        help="Evalúa únicamente los 9 pedidos comerciales (MSG-009 a MSG-017)",
    )
    parser.add_argument(
        "--comparar",
        action="store_true",
        help="Compara las 8 consultas con denso, BM25 e híbrido lado a lado",
    )
    parser.add_argument(
        "--sin-aprobar",
        action="store_true",
        help="No reanuda los interrupts de Gerencia (más rápido y económico en tokens)",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=2.0,
        help="Pausa en segundos entre llamadas consecutivas para rate limit (por defecto: 2.0)",
    )

    args = parser.parse_args()
    config.verificar_entorno()

    # Cargar casos del Golden Set
    datos_golden = datos.cargar_golden_set()
    casos = datos_golden.get("casos", [])

    # Filtrar según argumentos
    if args.caso:
        caso_buscado = args.caso.strip().upper()
        casos = [c for c in casos if c["id"].upper() == caso_buscado]
        if not casos:
            print(f"❌ Error: El caso '{args.caso}' no existe en {config.GOLDEN_SET.name}")
            return 1
    elif args.consultas:
        casos = [c for c in casos if c["intencion"] == "consulta"]
    elif args.pedidos:
        casos = [c for c in casos if c["intencion"] == "pedido"]

    # Modo comparativo para el RAG
    if args.comparar:
        consultas = [c for c in casos if c["intencion"] == "consulta"]
        print("\n" + "=" * 80)
        print("COMPARATIVA MULTI-RETRIEVER: DENSO vs BM25 vs HIBRIDO (8 CONSULTAS)")
        print("=" * 80)
        
        resultados_modos: dict[str, dict[str, Any]] = {}
        for modo in ("denso", "bm25", "hibrido"):
            print(f"\n--- Evaluando Retriever: {modo.upper()} ---")
            filas_modo, resumen_modo = correr_evaluacion(
                consultas,
                retriever=modo,
                aprobar=not args.sin_aprobar,
                pausa_segundos=args.pausa,
            )
            guardar_reporte(filas_modo, resumen_modo, retriever=modo)
            resultados_modos[modo] = {f["id"]: f for f in filas_modo}

        # Tabla comparativa consolidada
        encabezados_comp = ["Consulta", "Denso (Dato/Doc)", "BM25 (Dato/Doc)", "Hibrido (Dato/Doc)"]
        filas_comp = []
        for c in consultas:
            cid = c["id"]
            filas_comp.append([
                cid,
                f"{_si_no(resultados_modos['denso'][cid].get('dato_ok'))} / {_si_no(resultados_modos['denso'][cid].get('doc_ok'))}",
                f"{_si_no(resultados_modos['bm25'][cid].get('dato_ok'))} / {_si_no(resultados_modos['bm25'][cid].get('doc_ok'))}",
                f"{_si_no(resultados_modos['hibrido'][cid].get('dato_ok'))} / {_si_no(resultados_modos['hibrido'][cid].get('doc_ok'))}",
            ])
        print("\n" + tabla(filas_comp, encabezados_comp))
        return 0

    # Modo estándar (corrida simple o batch completo)
    print(f"\nIniciando Evaluacion Benchmark contra Golden Set")
    print(f"   * Total de casos: {len(casos)}")
    print(f"   * Modelo LLM:     {config.MODELO_LLM}")
    print(f"   * Retriever:      {args.retriever}")
    print(f"   * Pausa Throttling: {args.pausa} seg\n")

    filas, resumen = correr_evaluacion(
        casos,
        retriever=args.retriever,
        aprobar=not args.sin_aprobar,
        pausa_segundos=args.pausa,
    )

    imprimir_reporte(filas, resumen)
    guardar_reporte(filas, resumen, args.retriever)

    # Si hay ruteos erróneos, salir con código 1 para CI/CD
    return 0 if resumen.get("ruteo_pct", 0) == 100.0 else 1


if __name__ == "__main__":
    sys.exit(main())
