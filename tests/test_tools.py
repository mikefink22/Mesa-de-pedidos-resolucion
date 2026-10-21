# -*- coding: utf-8 -*-
"""
tests/test_tools.py - Pruebas y verificación de las Tools Comerciales.

Permite:
1. Correr como tests automáticos de pytest:
       pytest tests/test_tools.py
2. Correr interactivamente con salidas visuales formateadas:
       py tests/test_tools.py
3. Probar casos individuales por consola:
       py tests/test_tools.py --sku AB-1020 --cant 50
       py tests/test_tools.py --sku AB-1021             (tolerancia tipográfica)
       py tests/test_tools.py --desc "amoladora 115"    (desempate técnico)
       py tests/test_tools.py --desc "amoladora"        (detección de ambigüedad)
       py tests/test_tools.py --cuit 30-71099873-2 --monto 200000 (mora 38 días)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Garantiza que Python encuentre el paquete mesa_pedidos sin importar desde dónde se ejecute
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesa_pedidos.tools import consultar_producto, verificar_credito


# ===========================================================================
# 1. Pruebas Automáticas para Pytest
# ===========================================================================

def test_consultar_producto_sku_exacto():
    res = consultar_producto.invoke({"sku": "AB-1020", "cantidad": 50})
    assert res["encontrado"] is True
    assert res["sku"] == "AB-1020"
    assert res["hay_stock"] is True
    assert res["precio_lista"] == 3_250.0


def test_consultar_producto_tolerancia_tipografica():
    # AB-1021 no existe, pero difflib detecta AB-1020 con similitud >= 75%
    res = consultar_producto.invoke({"sku": "AB-1021", "cantidad": 10})
    assert res["encontrado"] is False
    assert res["ambiguo"] is True
    assert any(s["sku"] == "AB-1020" for s in res["sugerencias"])


def test_consultar_producto_inexistente():
    res = consultar_producto.invoke({"sku": "ZZ-9999", "cantidad": 1})
    assert res["encontrado"] is False
    assert res["ambiguo"] is False
    assert len(res["sugerencias"]) == 0


def test_consultar_producto_desempate_tecnico():
    # 'amoladora angular 115' debe desambiguar a HE-4110 (115 mm) y no a HE-4180 (230 mm)
    res = consultar_producto.invoke({"descripcion": "amoladora angular 115", "cantidad": 1})
    assert res["encontrado"] is True
    assert res["sku"] == "HE-4110"


def test_consultar_producto_deteccion_ambiguedad():
    # 'amoladora angular' sin medida empata entre 115 mm y 230 mm -> ambiguo True
    res = consultar_producto.invoke({"descripcion": "amoladora angular", "cantidad": 1})
    assert res["encontrado"] is False
    assert res["ambiguo"] is True
    assert len(res["sugerencias"]) >= 2


def test_verificar_credito_cliente_normal():
    res = verificar_credito.invoke({"cuit": "30-71455982-2", "monto_neto": 200_000.0})
    assert res["encontrado"] is True
    assert res["entra_en_credito"] is True
    assert res["bloqueado"] is False
    assert res["requiere_autorizacion"] is False


def test_verificar_credito_mora_leve_autorizacion():
    # CUIT con mora de 38 días (requiere autorización según Art. 7)
    res = verificar_credito.invoke({"cuit": "30-71099873-2", "monto_neto": 200_000.0})
    assert res["encontrado"] is True
    assert res["requiere_autorizacion"] is True
    assert any("requiere autorización" in m.lower() for m in res["motivos"])


def test_verificar_credito_suspendido_bloqueado_sin_anticipado():
    # CUIT suspendido con 95 días de mora
    res = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": False})
    assert res["encontrado"] is True
    assert res["bloqueado"] is True
    assert res["entra_en_credito"] is False


def test_verificar_credito_suspendido_habilitado_con_anticipado():
    # El mismo CUIT suspendido pero con pago anticipado no consume crédito
    res = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": True})
    assert res["encontrado"] is True
    assert res["bloqueado"] is False
    assert res["entra_en_credito"] is True


# ===========================================================================
# 2. Ejecución Visual e Interactiva por Consola
# ===========================================================================

def imprimir_json(titulo: str, obj: dict) -> None:
    print(f"\n{'='*70}")
    print(f"👉 {titulo}")
    print(f"{'='*70}")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def correr_casos_demostracion() -> None:
    print("\n" + "=" * 75)
    print("🔍 DEMOSTRACIÓN VISUAL DE TOOLS COMERCIALES")
    print("=" * 75)

    print("\n📦 1. Búsqueda exacta por SKU ('AB-1020', 50 u):")
    res1 = consultar_producto.invoke({"sku": "AB-1020", "cantidad": 50})
    imprimir_json("Resultado SKU Exacto", res1)

    print("\n📦 2. Tolerancia tipográfica ('AB-1021' con typo -> sugiere 'AB-1020'):")
    res2 = consultar_producto.invoke({"sku": "AB-1021", "cantidad": 10})
    imprimir_json("Resultado Typo", res2)

    print("\n📦 3. Desempate técnico ('amoladora angular 115' -> HE-4110):")
    res3 = consultar_producto.invoke({"descripcion": "amoladora angular 115", "cantidad": 1})
    imprimir_json("Resultado Desempate Técnico", res3)

    print("\n📦 4. Detección de ambigüedad ('amoladora angular' sin medida -> eleva opciones):")
    res4 = consultar_producto.invoke({"descripcion": "amoladora angular", "cantidad": 1})
    imprimir_json("Resultado Ambigüedad", res4)

    print("\n💳 5. Crédito normal (El Tornillo Feliz):")
    res5 = verificar_credito.invoke({"cuit": "30-71455982-2", "monto_neto": 200_000.0})
    imprimir_json("Resultado Crédito Normal", res5)

    print("\n💳 6. Mora de 38 días (Construcciones Del Valle -> Requiere Autorización):")
    res6 = verificar_credito.invoke({"cuit": "30-71099873-2", "monto_neto": 200_000.0})
    imprimir_json("Resultado Mora Leve", res6)

    print("\n💳 7. Cliente suspendido (Industrias Norte Grande -> Bloqueado sin anticipado):")
    res7 = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": False})
    imprimir_json("Resultado Suspendido Bloqueado", res7)

    print("\n💳 8. Cliente suspendido pero con pago anticipado (Habilitado):")
    res8 = verificar_credito.invoke({"cuit": "30-71788450-3", "monto_neto": 100_000.0, "pago_anticipado": True})
    imprimir_json("Resultado Suspendido con Pago Anticipado", res8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de tools comerciales")
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
        imprimir_json(f"Verificar Crédito: cuit={args.cuit}, monto={args.monto}", res)
    else:
        correr_casos_demostracion()


if __name__ == "__main__":
    main()
