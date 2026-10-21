# -*- coding: utf-8 -*-
"""
tests/test_reglas.py - Pruebas determinísticas de las Reglas Comerciales.

Valida en Python puro (cero tokens de LLM y cero llamadas a API):
1. Descuentos por volumen (Art. 4).
2. Topes por marca y tipo de producto (Art. 5: 15% para herramientas eléctricas Dewalt/Makita/Bosch).
3. Descuento financiero por pago anticipado (Art. 6: 3% fuera de tope).
4. Reglas de crédito y mora (Art. 7: 30 a 60 días autorización, >60 días bloqueo salvo anticipado).
5. Cotización de flete, plazos, bonificaciones y recargo por carga peligrosa.
6. Motor decidir_ruta simulando los casos emblemáticos del negocio.

Ejecución:
    pytest tests/test_reglas.py
    py tests/test_reglas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garantiza la importación de mesa_pedidos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesa_pedidos import reglas
from mesa_pedidos.utilidades import pesos, porcentaje


# ===========================================================================
# 1. Pruebas Unitarias de Reglas Comerciales (Casos Sintéticos Aislados)
# ===========================================================================

def test_descuento_por_volumen_art_4():
    """Valida la escala escalonada del Art. 4."""
    assert reglas.descuento_por_volumen(500_000.0) == 0.0
    assert reglas.descuento_por_volumen(1_000_000.0) == 5.0
    assert reglas.descuento_por_volumen(2_500_000.0) == 5.0
    assert reglas.descuento_por_volumen(3_000_000.0) == 8.0
    assert reglas.descuento_por_volumen(5_500_000.0) == 8.0
    assert reglas.descuento_por_volumen(6_000_000.0) == 12.0
    assert reglas.descuento_por_volumen(10_000_000.0) == 12.0


def test_tope_del_producto_art_5():
    """Valida que el 15% solo aplique cuando es herramienta eléctrica Y de marca con tope."""
    # 1. Herramienta eléctrica Dewalt, Makita o Bosch -> 15%
    assert reglas.tope_del_producto("Herramienta electrica", "Dewalt") == 15.0
    assert reglas.tope_del_producto("Herramienta electrica", "Makita") == 15.0
    assert reglas.tope_del_producto("Herramienta electrica", "Bosch") == 15.0

    # 2. Herramienta eléctrica de OTRA marca -> 20%
    assert reglas.tope_del_producto("Herramienta electrica", "Black & Decker") == 20.0

    # 3. Producto Bosch que NO es herramienta eléctrica (ej: disco de corte Abrasivos Bosch) -> 20%
    assert reglas.tope_del_producto("Abrasivos", "Bosch") == 20.0
    assert reglas.tope_del_producto("Fijaciones", "Fischer") == 20.0


def test_valorizacion_renglon_por_renglon_con_topes():
    """Valida que si hay un 20% general, las herramientas Dewalt queden topeadas al 15%
    mientras los demás renglones conservan el 20%.
    """
    cliente = {"descuento_habitual": 10.0}
    items = [
        # Discos Tyrolit (Abrasivos): debe recibir 20%
        {
            "sku": "AB-1020",
            "nombre": "Disco de corte Tyrolit",
            "rubro": "Abrasivos",
            "marca": "Tyrolit",
            "cantidad": 100,
            "precio_lista": 3250.0,
            "alicuota_iva": 21.0,
            "stock": 500,
        },
        # Rotomartillo Dewalt: debe topear al 15%
        {
            "sku": "HE-4110",
            "nombre": "Amoladora angular Dewalt",
            "rubro": "Herramienta electrica",
            "marca": "Dewalt",
            "cantidad": 10,
            "precio_lista": 150000.0,
            "alicuota_iva": 21.0,
            "stock": 50,
        },
    ]

    # Bruto total: 100*3250 + 10*150000 = 325.000 + 1.500.000 = 1.825.000 (volumen 5%)
    # Comercial total: 10% habitual + 5% volumen = 15% general
    val1 = reglas.valorizar(items, cliente)
    assert val1["descuento_aplicado_pct"] == 15.0
    assert val1["renglones"][0].descuento_pct == 15.0
    assert val1["renglones"][1].descuento_pct == 15.0

    # Ahora si el cliente pide un 20% puntual dentro del tope (Art. 5):
    val2 = reglas.valorizar(items, cliente, descuento_solicitado=20.0)
    assert val2["descuento_aplicado_pct"] == 20.0
    # Renglón Abrasivos recibe 20%:
    assert val2["renglones"][0].descuento_pct == 20.0
    # Renglón Herramienta Dewalt queda TOPEADO al 15%:
    assert val2["renglones"][1].descuento_pct == 15.0
    assert "tope 15% por marca" in val2["renglones"][1].nota


def test_descuento_autorizado_gerencia_soberania_y_tope_marca():
    """Valida que si Gerencia Comercial autoriza una excepción mayor (ej. 30%):
    - Los productos generales (Abrasivos) reciban el 30% autorizado, superando el 20%.
    - Las herramientas eléctricas de marcas protegidas (Makita) mantengan el tope del 15% por contrato.
    - La sumatoria de renglones coincida exactamente al centavo con el neto del pedido.
    """
    cliente = {"razon_social": "Cliente Licitación", "descuento_habitual": 10.0, "categoria": "A"}
    items = [
        {
            "sku": "AB-1020",
            "nombre": "Disco de corte Tyrolit",
            "rubro": "Abrasivos",
            "marca": "Tyrolit",
            "cantidad": 100,
            "precio_lista": 3250.0,
            "alicuota_iva": 21.0,
            "stock": 500,
        },
        {
            "sku": "HE-4410",
            "nombre": "Rotomartillo SDS Plus Makita",
            "rubro": "Herramienta electrica",
            "marca": "Makita",
            "cantidad": 2,
            "precio_lista": 250000.0,
            "alicuota_iva": 21.0,
            "stock": 10,
        },
    ]

    val = reglas.valorizar(items, cliente, descuento_autorizado=30.0)

    # 1. El descuento aplicado general refleja la autorización de Gerencia
    assert val["descuento_aplicado_pct"] == 30.0

    # 2. El renglón de Abrasivos recibe el 30% pleno (perfora el tope del 20%)
    r_abrasivo = val["renglones"][0]
    assert r_abrasivo.descuento_pct == 30.0
    subtotal_esperado_abrasivo = round(100 * 3250.0 * 0.70, 2)  # 227.500,00
    assert r_abrasivo.subtotal_neto == subtotal_esperado_abrasivo
    assert r_abrasivo.nota == ""

    # 3. El renglón de Makita mantiene el tope del 15% contractual
    r_makita = val["renglones"][1]
    assert r_makita.descuento_pct == 15.0
    subtotal_esperado_makita = round(2 * 250000.0 * 0.85, 2)  # 425.000,00
    assert r_makita.subtotal_neto == subtotal_esperado_makita
    assert "tope 15% por marca" in r_makita.nota

    # 4. Coherencia matemática estricta: la suma de subtotales da el neto total
    neto_calculado = sum(r.subtotal_neto for r in val["renglones"])
    assert val["neto"] == neto_calculado
    assert val["neto"] == round(subtotal_esperado_abrasivo + subtotal_esperado_makita, 2)


def test_evaluar_credito_mora_y_bloqueo():
    """Valida el análisis crediticio del Art. 7."""
    cliente_ok = {
        "razon_social": "Cliente Al Día",
        "categoria": "A",
        "estado": "activo",
        "mora_dias": 0,
        "limite_credito": 1_000_000.0,
        "saldo_actual": 200_000.0,
    }
    c_ok = reglas.evaluar_credito(cliente_ok, monto_neto=300_000.0)
    assert c_ok["entra_en_credito"] is True
    assert c_ok["bloqueado"] is False
    assert c_ok["requiere_autorizacion"] is False

    # Mora de 45 días -> requiere autorización
    cliente_mora_leve = dict(cliente_ok, mora_dias=45)
    c_leve = reglas.evaluar_credito(cliente_mora_leve, monto_neto=100_000.0)
    assert c_leve["requiere_autorizacion"] is True
    assert c_leve["bloqueado"] is False

    # Mora de 80 días -> bloqueado si no es pago anticipado
    cliente_mora_grave = dict(cliente_ok, mora_dias=80)
    c_grave_cta_cte = reglas.evaluar_credito(cliente_mora_grave, monto_neto=100_000.0, pago_anticipado=False)
    assert c_grave_cta_cte["bloqueado"] is True
    assert c_grave_cta_cte["entra_en_credito"] is False

    # El mismo cliente grave pero con pago anticipado -> Habilitado
    c_grave_anticipado = reglas.evaluar_credito(cliente_mora_grave, monto_neto=100_000.0, pago_anticipado=True)
    assert c_grave_anticipado["bloqueado"] is False
    assert c_grave_anticipado["entra_en_credito"] is True


def test_cotizar_flete_zonas_y_bonificacion():
    """Valida tarifas, bonificaciones y recargos de flete."""
    # Gran Córdoba: Zona 1, base $35.000
    f_cba = reglas.cotizar_flete("Cordoba", "Cordoba Capital", monto_neto=500_000.0)
    assert f_cba["zona"] == 1
    assert f_cba["costo"] == 35_000.0

    # Neuquén: Zona 5, base $175.000
    f_nqn = reglas.cotizar_flete("Neuquen", "Neuquen Capital", monto_neto=500_000.0)
    assert f_nqn["zona"] == 5
    assert f_nqn["costo"] == 175_000.0

    # Bonificación en Zona 2 con neto > $1.500.000 -> 100% bonificado ($0)
    f_sf = reglas.cotizar_flete("Santa Fe", "Rosario", monto_neto=1_600_000.0)
    assert f_sf["bonificacion_pct"] == 100
    assert f_sf["costo"] == 0.0

    # Carga peligrosa (aerosoles): 15% recargo
    f_aerosol = reglas.cotizar_flete("Buenos Aires", "La Plata", monto_neto=200_000.0, unidades_aerosol=10)
    assert f_aerosol["costo"] == round(95_000.0 * 1.15, 2)


def test_decidir_ruta_casos_clave():
    """Valida la máquina de decisión determinística sin LLM."""
    cliente_a = {
        "categoria": "A",
        "estado": "activo",
        "mora_dias": 0,
        "limite_credito": 5_000_000.0,
        "saldo_actual": 0.0,
    }

    # 1. Pedido normal en regla -> confirmar_pedido
    item_ok = [
        reglas.RenglonValorizado(
            sku="AB-1020",
            nombre="Disco",
            marca="Tyrolit",
            rubro="Abrasivos",
            cantidad=100,
            precio_unitario=3250.0,
            subtotal_lista=325000.0,
            descuento_pct=10.0,
            subtotal_neto=292500.0,
            alicuota_iva=21.0,
            stock_disponible=500,
            hay_stock=True,
        )
    ]
    val_ok = {"renglones": item_ok, "neto": 292500.0}
    cred_ok = {"bloqueado": False, "requiere_autorizacion": False}
    flete_ok = {"alertas": []}

    ruta, _, alertas = reglas.decidir_ruta(cliente_a, val_ok, cred_ok, flete_ok, [], [])
    assert ruta == "confirmar_pedido"
    assert len(alertas) == 0

    # 2. Cliente pide 25% de descuento (>20% tope) -> aprobacion_humana
    ruta_desc, _, alertas_desc = reglas.decidir_ruta(
        cliente_a, val_ok, cred_ok, flete_ok, [], [], descuento_solicitado=25.0
    )
    assert ruta_desc == "aprobacion_humana"
    assert any("supera el tope" in a for a in alertas_desc)

    # 3. Cliente bloqueado por mora grave sin pago anticipado -> rechazar_pedido
    cred_bloq = {"bloqueado": True, "motivos": ["Cliente suspendido con 90 días de mora"]}
    ruta_rech, _, _ = reglas.decidir_ruta(cliente_a, val_ok, cred_bloq, flete_ok, [], [])
    assert ruta_rech == "rechazar_pedido"


# ===========================================================================
# 2. Ejecución Visual Didáctica por Consola
# ===========================================================================

def main() -> None:
    print("=" * 80)
    print("📊 DEMOSTRACIÓN VISUAL Y DIDÁCTICA: REGLAS COMERCIALES (SIN LLM)")
    print("=" * 80)

    # Simulación del caso de doble tope (Abrasivos vs Herramienta Eléctrica Dewalt)
    print("\n[ESCENARIO 1] Pedido mixto con Descuento General del 20%:")
    print("  • Cliente solicita 20% de descuento puntual (facultad delegada Art. 5).")
    print("  • Renglón 1: Discos Tyrolit (Abrasivos).")
    print("  • Renglón 2: Amoladora Dewalt (Herramienta eléctrica de marca protegida).")

    cliente = {"razon_social": "Ferretería El Tornillo Feliz", "descuento_habitual": 10.0, "categoria": "A"}
    items = [
        {
            "sku": "AB-1020",
            "nombre": "Disco de corte metal 115 mm Tyrolit",
            "rubro": "Abrasivos",
            "marca": "Tyrolit",
            "cantidad": 200,
            "precio_lista": 3250.0,
            "alicuota_iva": 21.0,
            "stock": 1840,
        },
        {
            "sku": "HE-4110",
            "nombre": "Amoladora angular 115 mm Dewalt",
            "rubro": "Herramienta electrica",
            "marca": "Dewalt",
            "cantidad": 5,
            "precio_lista": 189000.0,
            "alicuota_iva": 21.0,
            "stock": 52,
        },
    ]

    val = reglas.valorizar(items, cliente, descuento_solicitado=20.0)

    print("\n  Desglose Renglón por Renglón:")
    print(f"  {'SKU':<10} {'Rubro/Marca':<30} {'Cant':<6} {'P.Lista':<12} {'Desc.Aplicado':<15} {'Subtotal Neto'}")
    print("  " + "-" * 88)
    for r in val["renglones"]:
        rubro_marca = f"{r.rubro} ({r.marca})"
        print(f"  {r.sku:<10} {rubro_marca:<30} {r.cantidad:<6} {pesos(r.precio_unitario):<12} {porcentaje(r.descuento_pct):<15} {pesos(r.subtotal_neto)}")
        if r.nota:
            print(f"    ↳ NOTA DE AUDITORÍA: {r.nota}")

    print("\n  Totales del Pedido:")
    print(f"  • Total Bruto de Lista:     {pesos(val['bruto'])}")
    print(f"  • Descuento Volumen Art. 4: {porcentaje(val['descuento_volumen_pct'])}")
    print(f"  • Neto de Mercadería:       {pesos(val['neto'])}")
    print(f"  • IVA Estimado:             {pesos(val['iva'])}")
    print(f"  • Total con IVA:            {pesos(val['neto'] + val['iva'])}")

    print("\n" + "=" * 80)
    print("🎯 Conclusión: El producto Tyrolit recibió el 20%, pero la amoladora Dewalt")
    print("   quedó automáticamente restringida al 15% como exige el Artículo 5.")
    print("=" * 80)


if __name__ == "__main__":
    main()
