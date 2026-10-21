# -*- coding: utf-8 -*-
"""
probar_tools.py - Script interactivo de prueba para las Tools Comerciales (Fase 2).

Permite probar manualmente y de forma determinística (sin gastar tokens de LLM):
1. consultar_producto (exacto, tipográfico, ambiguo, inexistente).
2. verificar_credito (con crédito, excedido, en mora 30-60 días, suspendido).

Uso:
  py probar_tools.py                           (corre la suite de casos de demostración)
  py probar_tools.py --sku AB-1020 --cant 50
  py probar_tools.py --sku AB-1021             (prueba de tolerancia tipográfica)
  py probar_tools.py --desc "amoladora 115"    (prueba de ambigüedad)
  py probar_tools.py --cuit 30-71099873-2 --monto 500000 (prueba de mora 38 días)
"""

from __future__ import annotations

import argparse
import json
from mesa_pedidos.tools import consultar_producto, verificar_credito
from mesa_pedidos.utilidades import pesos


def imprimir_json(titulo: str, obj: dict) -> None:
    print(f"\n{'='*70}")
    print(f"👉 {titulo}")
    print(f"{'='*70}")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def correr_casos_demostracion() -> None:
    print("\n🔍 --- CASOS DE PRUEBA: CONSULTAR PRODUCTO ---")

    # Caso 1: SKU Exacto con stock suficiente
    res1 = consultar_producto.invoke({"sku": "AB-1020", "cantidad": 50})
    imprimir_json("Caso 1: SKU exacto 'AB-1020' (pedido 50 u)", res1)

    # Caso 2: Error tipográfico leve (AB-1021 no existe, pero es similar a AB-1020)
    res2 = consultar_producto.invoke({"sku": "AB-1021", "cantidad": 10})
    imprimir_json("Caso 2: Error tipográfico 'AB-1021' (debe sugerir AB-1020)", res2)

    # Caso 3: SKU inexistente total (<75% de similitud)
    res3 = consultar_producto.invoke({"sku": "ZZ-9999", "cantidad": 1})
    imprimir_json("Caso 3: SKU totalmente inventado 'ZZ-9999'", res3)

    # Caso 4: Descripción unívoca sin SKU
    res4 = consultar_producto.invoke({"descripcion": "zapatos de seguridad", "cantidad": 2})
    imprimir_json("Caso 4: Descripción unívoca 'zapatos de seguridad'", res4)

    # Caso 5: Descripción unívoca con especificación técnica (115 mm desempata entre las dos amoladoras)
    res5 = consultar_producto.invoke({"descripcion": "amoladora angular 115", "cantidad": 1})
    imprimir_json("Caso 5: Especificación técnica 'amoladora angular 115' (desempata a HE-4110)", res5)

    # Caso 5b: Ambigüedad real - Amoladora sin especificar medida (empatan HE-4110 de 115mm y HE-4180 de 230mm)
    res5b = consultar_producto.invoke({"descripcion": "amoladora angular", "cantidad": 1})
    imprimir_json("Caso 5b: Ambigüedad real 'amoladora angular' (empatan 115mm y 230mm)", res5b)

    # Caso 5c: Ambigüedad real - Discos de corte sin especificar medida (empatan AB-1020 y AB-1035)
    res5c = consultar_producto.invoke({"descripcion": "disco de corte", "cantidad": 50})
    imprimir_json("Caso 5c: Ambigüedad real 'disco de corte' (empatan 115mm y 180mm)", res5c)

    # Caso 5d: Ambigüedad de coincidencia perfecta múltiple ('guante ombu' coincide 100% con vaqueta y anticorte)
    res5d = consultar_producto.invoke({"descripcion": "guante ombu", "cantidad": 5})
    imprimir_json("Caso 5d: Ambigüedad 100% múltiple 'guante ombu' (empatan vaqueta SE-5040 y anticorte SE-5055)", res5d)

    print("\n💳 --- CASOS DE PRUEBA: VERIFICAR CRÉDITO ---")

    # Caso 6: Cliente normal con crédito disponible (Ferretería El Tornillo Feliz)
    res6 = verificar_credito.invoke({"cuit": "30-71455982-2", "monto_neto": 200_000.0})
    imprimir_json("Caso 6: Cliente activo sin mora dentro del límite", res6)

    # Caso 7: Cliente con mora de 38 días (Construcciones Del Valle S.R.L.) -> Requiere autorización
    res7 = verificar_credito.invoke({"cuit": "30-71099873-2", "monto_neto": 200_000.0})
    imprimir_json("Caso 7: Cliente con mora de 38 días (requiere autorización)", res7)

    # Caso 8: Cliente suspendido con 95 días de mora (Industrias Norte Grande S.R.L.)
    res8 = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": False})
    imprimir_json("Caso 8: Cliente suspendido (debe bloquearse)", res8)

    # Caso 9: El mismo cliente suspendido pero pagando por anticipado
    res9 = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": True})
    imprimir_json("Caso 9: Cliente suspendido con pago anticipado (se habilita)", res9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba manual de tools comerciales")
    parser.add_argument("--sku", type=str, default=None, help="SKU a consultar")
    parser.add_argument("--desc", type=str, default=None, help="Descripción a buscar")
    parser.add_argument("--cant", type=int, default=1, help="Cantidad pedida")
    parser.add_argument("--cuit", type=str, default=None, help="CUIT a verificar")
    parser.add_argument("--monto", type=float, default=100000.0, help="Monto neto")
    parser.add_argument("--anticipado", action="store_true", help="Pago anticipado")

    args = parser.parse_args()

    if args.sku or args.desc:
        res = consultar_producto.invoke({"sku": args.sku, "descripcion": args.desc, "cantidad": args.cant})
        imprimir_json(f"Consulta Producto: sku={args.sku}, desc={args.desc}", res)
    elif args.cuit:
        res = verificar_credito.invoke({"cuit": args.cuit, "monto_neto": args.monto, "pago_anticipado": args.anticipado})
        imprimir_json(f"Verificación Crédito: cuit={args.cuit}, monto={args.monto}", res)
    else:
        correr_casos_demostracion()


if __name__ == "__main__":
    main()
