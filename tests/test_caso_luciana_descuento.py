# -*- coding: utf-8 -*-
"""
tests/test_caso_luciana_descuento.py - Laboratorio Experimental de Reglas y Ruteo

Este script NO tiene asserts rígidos pre-cocinados. Es un laboratorio vivo
para pasarle diferentes combinaciones de cantidades, stock y descuentos,
y observar cómo reacciona el motor de decisión determinístico en tiempo real.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garantiza la importación de mesa_pedidos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesa_pedidos import reglas
from mesa_pedidos.utilidades import pesos, porcentaje


def ejecutar_experimento(
    titulo: str,
    cantidad_dewalt: int,
    descuento_solicitado: float,
    descuento_autorizado_gerencia: float | None = None,
) -> None:
    print("=" * 85)
    print(f"🧪 {titulo}")
    print("=" * 85)

    cliente = {
        "cuit": "30-71340098-6",
        "razon_social": "Distribuidora Cuyo Insumos S.A.",
        "categoria": "A",
        "descuento_habitual": 12.0,
        "limite_credito": 20_000_000.0,
        "saldo_actual": 0.0,
        "estado": "activo",
        "mora_dias": 0,
    }

    # Stock fijo de catálogo: Tyrolit hay 720 u., Dewalt hay 52 u.
    stock_dewalt_real = 52
    items = [
        {
            "sku": "AB-1035",
            "nombre": "Disco de corte Tyrolit 180mm",
            "rubro": "Abrasivos",
            "marca": "Tyrolit",
            "cantidad": 500,
            "precio_lista": 5900.0,
            "alicuota_iva": 21.0,
            "stock": 720,
        },
        {
            "sku": "HE-4110",
            "nombre": "Amoladora Dewalt 115mm",
            "rubro": "Herramienta electrica",
            "marca": "Dewalt",
            "cantidad": cantidad_dewalt,
            "precio_lista": 289000.0,
            "alicuota_iva": 21.0,
            "stock": stock_dewalt_real,
        },
    ]

    print(f"• Solicitud del Cliente: 500 Discos + {cantidad_dewalt} Amoladoras Dewalt (Hay {stock_dewalt_real} en stock)")
    print(f"• Descuento Solicitado por Cliente: {porcentaje(descuento_solicitado)}")
    if descuento_autorizado_gerencia is not None:
        print(f"• Intervención de Gerencia Comercial (HITL): Autoriza {porcentaje(descuento_autorizado_gerencia)}")

    # 1. Ejecutar Valorización
    val = reglas.valorizar(
        items=items,
        cliente=cliente,
        descuento_solicitado=descuento_solicitado,
        descuento_autorizado=descuento_autorizado_gerencia,
    )

    # 2. Simular evaluación de crédito y flete limpios para aislar stock y descuento
    credito = {"bloqueado": False, "requiere_autorizacion": False, "motivos": []}
    flete = {"alertas": []}

    # 3. Ejecutar Motor de Ruteo
    ruta, motivo, alertas = reglas.decidir_ruta(
        cliente=cliente,
        valorizacion=val,
        credito=credito,
        flete=flete,
        no_resueltos=[],
        inferidos=[],
        descuento_solicitado=descuento_solicitado,
    )

    # 4. Mostrar Resultados Vivos
    print("\n📋 ¿QUÉ DECIDIÓ EL SISTEMA?")
    print(f"  -> Ruta de salida:  [{ruta.upper()}]")
    print(f"  -> Motivo central:  {motivo}")

    if alertas:
        print("\n⚠️ Alertas comerciales disparadas en vivo:")
        for a in alertas:
            print(f"     ! {a}")
    else:
        print("\n✅ Sin alertas comerciales: el pedido entra directo en política.")

    print("\n💰 Desglose renglón por renglón calculado por Python:")
    for r in val["renglones"]:
        linea = f"     - {r.cantidad} u. de {r.nombre} | P.Lista: {pesos(r.precio_unitario)} | Desc: {porcentaje(r.descuento_pct)} | Subtotal: {pesos(r.subtotal_neto)}"
        if r.nota:
            linea += f"  [{r.nota}]"
        if not r.hay_stock:
            linea += f"  [ALERTA: Faltan {r.cantidad - r.stock_disponible} u.]"
        print(linea)

    print(f"\n  • Total Neto Mercadería: {pesos(val['neto'])}")
    print(f"  • IVA Estimado (21%):    {pesos(val['iva'])}")
    print(f"  • Total Pedido:          {pesos(val['neto'] + val['iva'])}\n")


def test_experimentos_dinamicos():
    """Test vacío para que pytest no chille si se ejecuta por test runner."""
    pass


def main() -> None:
    # CASO 1: Stock suficiente (pide 30, hay 52) y Descuento normal (10%)
    ejecutar_experimento(
        titulo="CASO 1: Cantidad MENOR al stock (30 u.) y descuento normal (10%)",
        cantidad_dewalt=30,
        descuento_solicitado=10.0,
        descuento_autorizado_gerencia=None,
    )

    # CASO 2: Quiebre de stock (pide 80, hay 52) y Descuento excesivo (25%)
    ejecutar_experimento(
        titulo="CASO 2: Cantidad MAYOR al stock (80 u.) y descuento MAYOR al tope (25%)",
        cantidad_dewalt=80,
        descuento_solicitado=25.0,
        descuento_autorizado_gerencia=None,
    )

    # CASO 3: Reanudación HITL donde Gerencia autoriza un descuento agresivo (28%)
    # pero ajusta la cantidad de Dewalt a las 52 unidades que hay en stock
    ejecutar_experimento(
        titulo="CASO 3: Gerencia autoriza 28% y ajusta cantidad de Dewalt a stock real (52 u.)",
        cantidad_dewalt=52,
        descuento_solicitado=25.0,
        descuento_autorizado_gerencia=28.0,
    )


if __name__ == "__main__":
    main()
