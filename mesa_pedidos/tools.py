# -*- coding: utf-8 -*-
"""
tools.py - Herramientas comerciales determinísticas para LangChain / LangGraph.

Implementa las dos tools solicitadas por la consigna (Sección 4.2):
1. consultar_producto: Consulta stock, precios, IVA y depósito en catalogo.csv.
   - Búsqueda exacta por SKU normalizado.
   - Búsqueda de SKUs similares por cercanía tipográfica (cutoff >= 75%).
   - Búsqueda laxa por descripción cuando no hay SKU.
   - Detección de ambigüedad cuando hay más de una coincidencia.
2. verificar_credito: Consulta cuenta corriente en clientes.csv.
   - Validación de límite crediticio con saldo proyectado.
   - Evaluación de mora leve (30 a 60 días -> requiere autorización).
   - Inhabilitación por mora grave (>60 días o suspendido -> solo anticipado).

Ambas tools utilizan el decorador @tool de LangChain y validación de tipos
con Pydantic (args_schema).
"""

from __future__ import annotations

from typing import Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import config
from mesa_pedidos import datos
from mesa_pedidos.utilidades import pesos


# ===========================================================================
# 1. Esquemas de Entrada (args_schema de Pydantic)
# ===========================================================================

class ConsultarProductoInput(BaseModel):
    """Parámetros de entrada para consultar un producto en el catálogo."""
    sku: Optional[str] = Field(
        default=None,
        description="Código exacto o aproximado del producto (ej: 'AB-1020', 'HE-4110')."
    )
    descripcion: Optional[str] = Field(
        default=None,
        description="Descripción o nombre coloquial del producto si no se conoce el código (ej: 'disco de corte', 'amoladora 115')."
    )
    cantidad: int = Field(
        default=1,
        ge=1,
        description="Cantidad de unidades requeridas (entero positivo mayor o igual a 1)."
    )


class VerificarCreditoInput(BaseModel):
    """Parámetros de entrada para verificar la cuenta corriente de un cliente."""
    cuit: str = Field(
        description="CUIT del cliente, con o sin guiones (ej: '30-71455982-2' o '30714559822')."
    )
    monto_neto: float = Field(
        ge=0.0,
        description="Monto neto total del pedido en pesos argentinos (sin IVA, tras aplicar descuentos)."
    )
    pago_anticipado: bool = Field(
        default=False,
        description="True si el cliente abona de forma anticipada (contado o transferencia previa)."
    )


# ===========================================================================
# 2. Tool 1: Consultar Producto en Catálogo
# ===========================================================================

@tool(args_schema=ConsultarProductoInput)
def consultar_producto(
    sku: Optional[str] = None,
    descripcion: Optional[str] = None,
    cantidad: int = 1,
) -> dict[str, Any]:
    """Consulta el catálogo de productos por código SKU o por descripción textual.
    Retorna precio de lista, alícuota de IVA, stock disponible, depósito y si el
    stock es suficiente para la cantidad requerida.
    Si el producto no existe o hay ambigüedad, devuelve sugerencias sin adivinar.
    """
    # 1. Búsqueda directa por SKU si fue provisto
    if sku:
        prod = datos.buscar_producto(sku)
        if prod:
            return {
                "encontrado": True,
                "ambiguo": False,
                "inferido_por_descripcion": False,
                "sku": prod.sku,
                "nombre": prod.nombre,
                "rubro": prod.rubro,
                "marca": prod.marca,
                "unidad": prod.unidad,
                "precio_lista": prod.precio_lista,
                "alicuota_iva": prod.alicuota_iva,
                "stock": prod.stock,
                "deposito": prod.deposito,
                "cantidad_pedida": cantidad,
                "hay_stock": prod.stock >= cantidad,
                "sugerencias": [],
            }

        # Si el SKU no existió de forma exacta, buscar errores tipográficos (similitud >= 75%)
        similares = datos.buscar_skus_similares(sku, limite=3, cutoff=0.75)
        if similares:
            return {
                "encontrado": False,
                "ambiguo": True,
                "sku_buscado": sku,
                "motivo": f"El código '{sku}' no existe en el catálogo. Se detectaron códigos con alta similitud tipográfica.",
                "sugerencias": [
                    {
                        "sku": s.sku,
                        "nombre": s.nombre,
                        "marca": s.marca,
                        "precio_lista": s.precio_lista,
                        "stock": s.stock,
                    }
                    for s in similares
                ],
            }

    # 2. Si no hay SKU o no se encontró, buscar por descripción
    if descripcion:
        candidatos = datos.buscar_productos_por_texto(descripcion)
        if len(candidatos) == 1:
            prod = candidatos[0]
            return {
                "encontrado": True,
                "ambiguo": False,
                "inferido_por_descripcion": True,
                "sku": prod.sku,
                "nombre": prod.nombre,
                "rubro": prod.rubro,
                "marca": prod.marca,
                "unidad": prod.unidad,
                "precio_lista": prod.precio_lista,
                "alicuota_iva": prod.alicuota_iva,
                "stock": prod.stock,
                "deposito": prod.deposito,
                "cantidad_pedida": cantidad,
                "hay_stock": prod.stock >= cantidad,
                "sugerencias": [],
            }
        elif len(candidatos) > 1:
            return {
                "encontrado": False,
                "ambiguo": True,
                "motivo": f"Ambigüedad: se encontraron {len(candidatos)} productos compatibles con '{descripcion}'.",
                "sugerencias": [
                    {
                        "sku": c.sku,
                        "nombre": c.nombre,
                        "marca": c.marca,
                        "precio_lista": c.precio_lista,
                        "stock": c.stock,
                    }
                    for c in candidatos
                ],
            }

    # 3. No se encontró ninguna coincidencia
    return {
        "encontrado": False,
        "ambiguo": False,
        "sku_buscado": sku,
        "descripcion_buscada": descripcion,
        "motivo": "Producto no identificado en el catálogo.",
        "sugerencias": [],
    }


# ===========================================================================
# 3. Tool 2: Verificar Crédito del Cliente
# ===========================================================================

@tool(args_schema=VerificarCreditoInput)
def verificar_credito(
    cuit: str,
    monto_neto: float,
    pago_anticipado: bool = False,
) -> dict[str, Any]:
    """Verifica la cuenta corriente del cliente según las políticas de crédito (Art. 7).
    Determina si el monto neto entra en el límite de crédito, si existen días de
    mora y si el pedido requiere autorización de Gerencia o debe ser rechazado.
    Informa los datos financieros; no toma la decisión de ruteo del pedido.
    """
    cliente = datos.buscar_cliente(cuit)
    if not cliente:
        return {
            "encontrado": False,
            "cuit": cuit,
            "entra_en_credito": False,
            "bloqueado": True,
            "requiere_autorizacion": True,
            "motivos": [f"El CUIT '{cuit}' no figura en la base de clientes."],
        }

    limite = float(cliente.limite_credito)
    saldo = float(cliente.saldo_actual)
    mora = int(cliente.mora_dias)

    # Estado suspendido o mora mayor a 60 días: bloqueado en cuenta corriente
    suspendido = (cliente.estado == "suspendido" or mora > config.MORA_SUSPENSION_DIAS)
    
    # Mora de 30 a 60 días: requiere autorización gerencial
    mora_con_autorizacion = (config.MORA_AUTORIZACION_DIAS <= mora <= config.MORA_SUSPENSION_DIAS)
    
    # Si paga de forma anticipada, no suma a la deuda en cuenta corriente
    saldo_proyectado = saldo if pago_anticipado else (saldo + monto_neto)
    excede_limite = saldo_proyectado > limite

    motivos: list[str] = []
    if suspendido:
        motivos.append(
            f"Cliente con estado '{cliente.estado}' y {mora} días de mora: "
            "no se otorgan pedidos en cuenta corriente, solo con pago anticipado (Art. 7)."
        )
    if mora_con_autorizacion:
        motivos.append(
            f"Mora de {mora} días (entre {config.MORA_AUTORIZACION_DIAS} y "
            f"{config.MORA_SUSPENSION_DIAS} días): requiere autorización de Gerencia (Art. 7)."
        )
    if excede_limite and not pago_anticipado:
        motivos.append(
            f"Saldo actual {pesos(saldo)} + pedido {pesos(monto_neto)} = "
            f"{pesos(saldo_proyectado)} supera el límite de crédito de {pesos(limite)} (Art. 7)."
        )

    entra_en_credito = pago_anticipado or (not suspendido and not excede_limite)
    bloqueado = suspendido and not pago_anticipado
    requiere_autorizacion = (excede_limite and not suspendido) or mora_con_autorizacion

    return {
        "encontrado": True,
        "cuit": cliente.cuit,
        "razon_social": cliente.razon_social,
        "categoria": cliente.categoria,
        "estado": cliente.estado,
        "mora_dias": mora,
        "limite_credito": limite,
        "saldo_actual": saldo,
        "credito_disponible": max(0.0, limite - saldo),
        "monto_neto": monto_neto,
        "saldo_proyectado": saldo_proyectado,
        "pago_anticipado": pago_anticipado,
        "descuento_habitual": cliente.descuento_habitual,
        "entra_en_credito": entra_en_credito,
        "bloqueado": bloqueado,
        "requiere_autorizacion": requiere_autorizacion,
        "motivos": motivos,
    }


# Lista de herramientas para exportar
TOOLS_COMERCIALES = [consultar_producto, verificar_credito]
