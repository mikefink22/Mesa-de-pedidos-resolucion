# -*- coding: utf-8 -*-
"""
reglas.py - Políticas comerciales y reglas de negocio en Python puro.

Este módulo implementa las reglas comerciales determinísticas de Ferretería Industrial Andina:
- Cero matemática y cero decisiones de crédito en el LLM.
- Funciones puras: reciben datos y devuelven datos, sin efectos secundarios, sin I/O y sin llamadas a API.
- Cada regla referencia explícitamente el artículo del PDF de políticas corporativas del que proviene.

Funciones principales:
  1. intencion_efectiva(...)  -> Corrige la intención clasificada por el LLM si no cumple condiciones.
  2. descuento_por_volumen(...) -> Escala por monto del pedido a precio de lista (Art. 4).
  3. tope_del_producto(...)   -> Tope general 20% y 15% para herramientas eléctricas Dewalt/Makita/Bosch (Art. 5).
  4. valorizar(...)           -> Calcula precios de lista, descuentos, neto, financiero e IVA.
  5. evaluar_credito(...)     -> Evaluación crediticia y de mora (Art. 7).
  6. cotizar_flete(...)       -> Tarifas por zona, plazos, bonificaciones y recargo por carga peligrosa.
  7. decidir_ruta(...)        -> Motor determinístico de ruteo: confirmar, aprobación humana, rechazar o derivar.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import config
from mesa_pedidos.datos import normalizar_texto
from mesa_pedidos.esquemas import RenglonValorizado
from mesa_pedidos.utilidades import pesos, porcentaje


# ===========================================================================
# 1. Ruteo de la Clasificación (Guardrails sobre el LLM)
# ===========================================================================

def intencion_efectiva(clasificacion: dict[str, Any], cliente: Optional[dict[str, Any]]) -> tuple[str, str]:
    """Valida y traduce la intención producida por el LLM a la rama real del grafo.
    
    Aplica dos guardrails determinísticos:
    1. Confianza baja (< 0.6): Si el LLM duda, se deriva a atención humana.
    2. Pedido de CUIT desconocido: Si un remitente intenta hacer un pedido pero
       su CUIT no figura en la base de clientes, se deriva a comercial/administración
       para el alta de cuenta corriente (no se puede validar crédito en el aire).
       (Nota: Una CONSULTA de un desconocido sí se responde, las políticas son públicas).
    
    Retorna: (intencion, nota_auditoria)
    """
    intencion = clasificacion.get("intencion", "derivar")
    confianza = float(clasificacion.get("confianza", 0.0))

    if confianza < config.CONFIANZA_MINIMA:
        return "derivar", f"Confianza del LLM {confianza:.2f} < umbral {config.CONFIANZA_MINIMA}: se deriva a operador humano"
    if intencion == "pedido" and not cliente:
        return "derivar", "Pedido de un CUIT inexistente en base de clientes: se deriva para alta de cuenta"
    if intencion not in ("consulta", "pedido", "derivar"):
        return "derivar", f"Intención desconocida '{intencion}': se deriva por precaución"
    return intencion, ""


# ===========================================================================
# 2. Descuentos y Valorización (Condiciones Comerciales, Arts. 4, 5 y 6)
# ===========================================================================

def descuento_por_volumen(monto_neto: float) -> float:
    """Escala por volumen según el monto a precio de lista (Art. 4).
    
    Escala:
      >= $6.000.000 -> 12.0%
      >= $3.000.000 ->  8.0%
      >= $1.000.000 ->  5.0%
       < $1.000.000 ->  0.0%
    """
    for desde, pct in config.ESCALA_VOLUMEN:
        if monto_neto >= desde:
            return pct
    return 0.0


def tope_del_producto(rubro: str, marca: str) -> float:
    """Tope máximo de descuento comercial según el Art. 5:
    - Regla general: 20%.
    - Excepción: Herramientas eléctricas de marcas Dewalt, Makita o Bosch: tope estricto del 15%.
    """
    es_herramienta_electrica = normalizar_texto(rubro) == normalizar_texto(config.RUBRO_CON_TOPE_MARCA)
    es_marca_con_tope = normalizar_texto(marca) in config.MARCAS_CON_TOPE
    if es_herramienta_electrica and es_marca_con_tope:
        return config.TOPE_DESCUENTO_MARCAS
    return config.TOPE_DESCUENTO_GENERAL


def valorizar(
    items: list[dict[str, Any]],
    cliente: dict[str, Any],
    descuento_solicitado: Optional[float] = None,
    pago_anticipado: bool = False,
    descuento_autorizado: Optional[float] = None,
) -> dict[str, Any]:
    """Calcula precios, descuentos por renglón, montos netos e IVA.
    
    Reglas aplicadas:
    1. Bruto: Suma de (cantidad * precio_lista).
    2. Descuento comercial general: habitual del cliente + volumen (Art. 4).
       - Si el cliente solicitó un descuento puntual que entra en el tope delegable (<= 20%),
         Mesa de Pedidos lo aplica de forma autónoma (Art. 5).
       - Si Gerencia autorizó uno (en el interrupt de edición), prevalece el autorizado.
       - Se topea al 20% general (Art. 5).
    3. Renglón por renglón: Se topea individualmente al 15% para herramientas eléctricas
       Dewalt / Makita / Bosch (Art. 5).
    4. Pago anticipado: 3% financiero sobre el neto de mercadería (Art. 6), por fuera del tope.
    5. IVA: Calculado según la alícuota de cada artículo en el catálogo.
    """
    bruto = sum(i["cantidad"] * i["precio_lista"] for i in items)
    volumen = descuento_por_volumen(bruto)
    comercial = float(cliente.get("descuento_habitual", 0.0)) + volumen

    notas: list[str] = [
        f"Descuento habitual {porcentaje(cliente.get('descuento_habitual', 0))} "
        f"+ volumen {porcentaje(volumen)} (Art. 4) = {porcentaje(comercial)}"
    ]

    if descuento_autorizado is not None:
        general = float(descuento_autorizado)
        notas.append(f"Descuento autorizado por Gerencia: {porcentaje(general)}")
    else:
        if descuento_solicitado and comercial < descuento_solicitado <= config.TOPE_DESCUENTO_GENERAL:
            comercial = float(descuento_solicitado)
            notas.append(f"Bonificación puntual solicitada por cliente: {porcentaje(comercial)} (Art. 5)")
        general = min(comercial, config.TOPE_DESCUENTO_GENERAL)
        if general < comercial:
            notas.append(f"Topeado al {porcentaje(general)} general (Art. 5)")

    renglones: list[RenglonValorizado] = []
    for i in items:
        tope = tope_del_producto(i["rubro"], i["marca"])
        pct = min(general, tope)
        subtotal_lista = i["cantidad"] * i["precio_lista"]
        subtotal_neto = round(subtotal_lista * (1 - pct / 100), 2)
        
        renglones.append(
            RenglonValorizado(
                sku=i["sku"],
                nombre=i["nombre"],
                marca=i["marca"],
                rubro=i["rubro"],
                cantidad=i["cantidad"],
                precio_unitario=i["precio_lista"],
                subtotal_lista=round(subtotal_lista, 2),
                descuento_pct=pct,
                subtotal_neto=subtotal_neto,
                alicuota_iva=i["alicuota_iva"],
                stock_disponible=i["stock"],
                hay_stock=i["stock"] >= i["cantidad"],
                deposito=i.get("deposito", ""),
                nota=f"tope {porcentaje(tope)} por marca (Art. 5)" if pct < general else "",
            )
        )

    neto_mercaderia = sum(r.subtotal_neto for r in renglones)
    factor_fin = (config.DESCUENTO_PAGO_ANTICIPADO / 100) if pago_anticipado else 0.0
    descuento_financiero = round(neto_mercaderia * factor_fin, 2)

    if pago_anticipado:
        notas.append(
            f"Pago anticipado: {porcentaje(config.DESCUENTO_PAGO_ANTICIPADO)} financiero adicional, "
            f"fuera del tope comercial (Art. 6)"
        )

    neto = round(neto_mercaderia - descuento_financiero, 2)
    iva = round(sum(r.subtotal_neto * (r.alicuota_iva / 100) for r in renglones) * (1 - factor_fin), 2)

    return {
        "renglones": renglones,
        "bruto": round(bruto, 2),
        "descuento_volumen_pct": volumen,
        "descuento_aplicado_pct": general,
        "descuento_financiero": descuento_financiero,
        "neto": neto,
        "iva": iva,
        "notas": notas,
    }


# ===========================================================================
# 3. Crédito y Mora (Condiciones Comerciales, Art. 7)
# ===========================================================================

def evaluar_credito(cliente: dict[str, Any], monto_neto: float, pago_anticipado: bool = False) -> dict[str, Any]:
    """Evalúa la cuenta corriente del cliente (Art. 7).
    
    Esta función es informativa: determina si entra en crédito, si está bloqueado o si
    requiere autorización, pero NO decide la ruta (eso lo hace `decidir_ruta`).
    """
    limite = float(cliente["limite_credito"])
    saldo = float(cliente["saldo_actual"])
    mora = int(cliente["mora_dias"])

    suspendido = (cliente["estado"] == "suspendido" or mora > config.MORA_SUSPENSION_DIAS)
    mora_con_autorizacion = (config.MORA_AUTORIZACION_DIAS <= mora <= config.MORA_SUSPENSION_DIAS)
    saldo_proyectado = saldo if pago_anticipado else (saldo + monto_neto)
    excede = saldo_proyectado > limite

    motivos: list[str] = []
    if suspendido:
        motivos.append(f"Cliente con estado '{cliente['estado']}' y {mora} días de mora (Art. 7)")
    if mora_con_autorizacion:
        motivos.append(
            f"Mora de {mora} días: entre {config.MORA_AUTORIZACION_DIAS} y "
            f"{config.MORA_SUSPENSION_DIAS} días requiere autorización de Gerencia (Art. 7)"
        )
    if excede:
        motivos.append(
            f"Saldo {pesos(saldo)} + pedido {pesos(monto_neto)} = "
            f"{pesos(saldo_proyectado)} supera el límite de {pesos(limite)} (Art. 7)"
        )

    return {
        "encontrado": True,
        "razon_social": cliente["razon_social"],
        "categoria": cliente["categoria"],
        "estado": cliente["estado"],
        "mora_dias": mora,
        "limite_credito": limite,
        "saldo_actual": saldo,
        "credito_disponible": max(0.0, limite - saldo),
        "monto_neto": monto_neto,
        "saldo_proyectado": saldo_proyectado,
        "pago_anticipado": pago_anticipado,
        "entra_en_credito": pago_anticipado or (not suspendido and not excede),
        "bloqueado": suspendido and not pago_anticipado,
        "requiere_autorizacion": (excede and not suspendido) or mora_con_autorizacion,
        "motivos": motivos,
    }


# ===========================================================================
# 4. Flete y Logística (Logística, Fletes y Plazos 2026)
# ===========================================================================

ZONAS_POR_PROVINCIA = {
    2: {"cordoba", "santa fe", "entre rios"},
    3: {"buenos aires", "la pampa", "san luis", "mendoza", "san juan"},
    4: {"salta", "jujuy", "tucuman", "santiago del estero", "catamarca", "la rioja",
        "chaco", "corrientes", "formosa", "misiones"},
    5: {"neuquen", "rio negro", "chubut", "santa cruz", "tierra del fuego"},
}

GRAN_CORDOBA = {
    "cordoba", "cordoba capital", "villa allende", "mendiolaza", "saldan",
    "la calera", "malvinas argentinas", "rio ceballos", "unquillo", "malagueno",
}

TARIFAS_FLETE = {  # zona: (costo_base, plazo, transportista)
    1: (35_000.0, "24 a 48 horas hábiles", "Cruz del Sur"),
    2: (68_000.0, "3 a 4 días hábiles", "Cruz del Sur"),
    3: (95_000.0, "4 a 6 días hábiles", "Cruz del Sur"),
    4: (130_000.0, "6 a 8 días hábiles", "Andesmar Cargas"),
    5: (175_000.0, "8 a 12 días hábiles", "Andesmar Cargas"),
}

UMBRAL_BONIFICACION_ZONAS_1_3 = 1_500_000.0
UMBRAL_BONIFICACION_50_ZONAS_4_5 = 1_500_000.0
UMBRAL_BONIFICACION_100_ZONAS_4_5 = 3_000_000.0
MAX_AEROSOLES_ZONAS_4_5 = 24
RECARGO_CARGA_PELIGROSA = 0.15


def zona_de(provincia: str, localidad: str = "") -> Optional[int]:
    """Determina el número de zona de envío (1 a 5)."""
    prov = normalizar_texto(provincia).strip()
    loc = normalizar_texto(localidad).strip()

    if prov == "cordoba" and loc in GRAN_CORDOBA:
        return 1
    for zona, provincias in ZONAS_POR_PROVINCIA.items():
        if prov in provincias:
            return zona
    return None


def cotizar_flete(
    provincia: str,
    localidad: str,
    monto_neto: float,
    pago_anticipado: bool = False,
    unidades_aerosol: int = 0,
) -> dict[str, Any]:
    """Calcula el costo del envío, plazo de entrega y transportista.
    
    Políticas de Logística:
    - Zonas 1 a 3: Bonificado al 100% si el monto neto supera $1.500.000 (Punto 3).
    - Zonas 4 y 5: Bonificado al 50% desde $1.500.000 y al 100% desde $3.000.000 (Punto 3).
    - Incompatibilidad: La bonificación de flete no se acumula con descuento por pago anticipado.
    - Carga peligrosa: Recargo del 15% por aerosoles (Punto 5). Máximo 24 u. a Zonas 4 y 5.
    """
    zona = zona_de(provincia, localidad)
    if zona is None:
        return {
            "zona": None,
            "costo": 0.0,
            "costo_base": 0.0,
            "plazo": "a convenir",
            "transportista": "",
            "bonificacion_pct": 0,
            "alertas": [],
            "detalle": f"Provincia '{provincia}' sin zona asignada",
        }

    base, plazo, transportista = TARIFAS_FLETE[zona]

    # Cálculo de bonificación por monto
    if zona <= 3:
        bonif = 100 if monto_neto > UMBRAL_BONIFICACION_ZONAS_1_3 else 0
    else:
        bonif = (
            100 if monto_neto >= UMBRAL_BONIFICACION_100_ZONAS_4_5
            else 50 if monto_neto >= UMBRAL_BONIFICACION_50_ZONAS_4_5
            else 0
        )

    detalle = [f"Zona {zona} ({provincia}): {pesos(base)}, {plazo}, {transportista}"]

    if bonif and pago_anticipado:
        detalle.append("La bonificación de flete no se acumula con el descuento por pago anticipado (Punto 3)")
        bonif = 0
    elif bonif:
        detalle.append(f"Flete bonificado al {bonif}% por monto (Punto 3)")

    costo = base * (1 - bonif / 100)
    alertas: list[str] = []

    if unidades_aerosol > 0:
        costo *= (1 + RECARGO_CARGA_PELIGROSA)
        detalle.append("Recargo 15% por carga peligrosa (aerosoles, Punto 5)")
        if zona >= 4 and unidades_aerosol > MAX_AEROSOLES_ZONAS_4_5:
            alertas.append(
                f"{unidades_aerosol} aerosoles a Zona {zona}: supera el máximo de "
                f"{MAX_AEROSOLES_ZONAS_4_5} unidades por envío (Logística, Punto 5)"
            )

    return {
        "zona": zona,
        "costo": round(costo, 2),
        "costo_base": base,
        "plazo": plazo,
        "transportista": transportista,
        "bonificacion_pct": bonif,
        "alertas": alertas,
        "detalle": "; ".join(detalle),
    }


# ===========================================================================
# 5. Cálculos de Apoyo Numérico para Consultas RAG
# ===========================================================================

_RE_MONTO = re.compile(r"\$?\s?(\d{1,3}(?:\.\d{3})+|\d{4,})(?:,\d+)?")


def montos_en_texto(texto: str) -> list[float]:
    """Extrae montos monetarios de un texto ignorando números chicos."""
    montos: list[float] = []
    for m in _RE_MONTO.finditer(texto or ""):
        valor = float(m.group(1).replace(".", ""))
        if valor >= 10_000:
            montos.append(valor)
    return montos


def calculos_para_consulta(texto: str, cliente: Optional[dict[str, Any]]) -> str:
    """Precalcula en Python los valores numéricos de volumen y flete si la consulta
    menciona montos concretos, para que el LLM no tenga que calcular matemática.
    """
    lineas: list[str] = []
    for monto in montos_en_texto(texto):
        vol = descuento_por_volumen(monto)
        lineas.append(f"Para un pedido de {pesos(monto)} neto: descuento por volumen {porcentaje(vol)} (Art. 4)")
        if cliente:
            flete = cotizar_flete(cliente["provincia"], cliente["localidad"], monto)
            if flete["zona"] is not None:
                lineas.append(
                    f"  Flete a {cliente['localidad']} ({cliente['provincia']}): {flete['detalle']}. "
                    f"Costo final estimado de flete: {pesos(flete['costo'])}."
                )
    return "\n".join(lineas) or "(la pregunta no menciona montos específicos)"


# ===========================================================================
# 6. Motor Determinístico de Ruteo de Pedidos (decidir_ruta)
# ===========================================================================

def decidir_ruta(
    cliente: dict[str, Any],
    valorizacion: dict[str, Any],
    credito: dict[str, Any],
    flete: dict[str, Any],
    no_resueltos: list[dict[str, Any]],
    inferidos: list[str],
    descuento_solicitado: Optional[float] = None,
) -> tuple[str, str, list[str]]:
    """Determina a qué hoja del grafo debe ir el pedido según las reglas de negocio.
    
    El orden de evaluación es estricto:
    1. Si no hay productos del catálogo identificados -> derivar_humano.
    2. Si el cliente está bloqueado en crédito (mora >60 días / suspendido sin anticipado) -> rechazar_pedido.
    3. Si existen excepciones comerciales -> aprobacion_humana (con lista de alertas para Gerencia).
    4. Si cumple todas las condiciones comerciales -> confirmar_pedido.
    
    Retorna: (ruta, motivo_principal, lista_de_alertas)
    """
    renglones: list[RenglonValorizado] = valorizacion.get("renglones", [])

    # 1. Sin productos reconocidos del catálogo: Derivación a humano
    if not renglones:
        return (
            "derivar_humano",
            "No se pudo identificar ningún producto del catálogo en el pedido",
            [f"Item no identificado: '{i.get('descripcion', '')}'" for i in no_resueltos],
        )

    # 2. Causales de rechazo directo (Art. 7)
    if credito.get("bloqueado"):
        return (
            "rechazar_pedido",
            f"Cliente en estado '{cliente.get('estado')}' con {cliente.get('mora_dias')} días de mora: "
            "no se otorgan pedidos en cuenta corriente, solo con pago anticipado (Art. 7)",
            list(credito.get("motivos", [])),
        )

    # 3. Causales de escalamiento a Aprobación Humana (Gerencia Comercial)
    alertas: list[str] = []

    # Alertas de crédito y mora
    if credito.get("requiere_autorizacion"):
        alertas.extend(credito.get("motivos", []))

    # Cliente categoría C sin pago anticipado requiere autorización de apertura de cuenta (Art. 6)
    if cliente.get("categoria") == "C" and not credito.get("pago_anticipado"):
        alertas.append("Cliente categoría C sin pago anticipado: la cuenta corriente requiere autorización de Gerencia (Art. 6)")

    # Descuento solicitado supera el tope general del 20% (Art. 5)
    if descuento_solicitado and descuento_solicitado > config.TOPE_DESCUENTO_GENERAL:
        alertas.append(
            f"Pide {porcentaje(descuento_solicitado)} de descuento: supera el tope máximo de "
            f"{porcentaje(config.TOPE_DESCUENTO_GENERAL)} (Art. 5)"
        )

    # Quiebre de stock en algún renglón (Art. 9)
    for r in renglones:
        if not r.hay_stock:
            alertas.append(
                f"Sin stock suficiente de {r.sku}: solicitan {r.cantidad} u., hay disponibles {r.stock_disponible} u. (Art. 9)"
            )

    # SKUs inferidos solo por descripción coloquial (confirmar que sea el producto correcto)
    for sku in inferidos:
        alertas.append(f"El producto {sku} fue inferido por descripción textual sin código SKU explícito: confirmar")

    # Ítems no identificados en el mensaje
    for i in no_resueltos:
        alertas.append(f"Ítem no reconocido en catálogo: '{i.get('descripcion', '')}'")

    # Pedido mínimo según la categoría del cliente (Art. 3)
    minimo = config.PEDIDO_MINIMO.get(cliente.get("categoria", ""), 0)
    if valorizacion.get("neto", 0.0) < minimo:
        alertas.append(
            f"Neto {pesos(valorizacion['neto'])} por debajo del pedido mínimo de "
            f"{pesos(minimo)} para categoría {cliente.get('categoria')} (Art. 3)"
        )

    # Alertas de flete y logística
    alertas.extend(flete.get("alertas", []))

    # Veredicto final
    if alertas:
        return "aprobacion_humana", alertas[0], alertas

    return (
        "confirmar_pedido",
        "Dentro del límite de crédito, con stock suficiente y descuentos dentro de los topes de política",
        [],
    )
