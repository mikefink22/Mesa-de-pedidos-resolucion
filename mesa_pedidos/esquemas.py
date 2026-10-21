# -*- coding: utf-8 -*-
"""
esquemas.py - Modelos de Pydantic para Structured Output y validación de datos.

Define los esquemas fuertemente tipados que:
1. El LLM debe producir de forma garantizada (clasificación, extracción de pedido, respuesta RAG).
2. Python utiliza para validar decisiones y estructurar el estado del sistema.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# ===========================================================================
# 1. Esquemas para RAG (Fase 1)
# ===========================================================================

class RespuestaConsulta(BaseModel):
    """Salida estructurada que el LLM genera al responder una consulta RAG."""
    respuesta: str = Field(
        description="Respuesta redactada en lenguaje natural, cordial y profesional para el cliente."
    )
    fuentes: list[int] = Field(
        default_factory=list,
        description="Lista de números enteros correspondientes a los fragmentos del contexto utilizados (ej. [1, 3])."
    )
    encontrado: bool = Field(
        default=True,
        description="True si la información solicitada se encontró en las políticas; False si no está disponible."
    )


# ===========================================================================
# 2. Esquemas para Clasificación y Pedidos (Fases 2 y 3)
# ===========================================================================

class ItemPedido(BaseModel):
    """Renglón de producto solicitado por el cliente."""
    producto: str = Field(description="Nombre, código o SKU del producto mencionado por el cliente.")
    cantidad: int = Field(gt=0, description="Cantidad solicitada (número entero positivo).")
    aclaracion: str = Field(default="", description="Detalles adicionales, marca o medidas.")


class PedidoExtraido(BaseModel):
    """Salida estructurada del LLM al extraer ítems de un mensaje de pedido."""
    items: list[ItemPedido] = Field(default_factory=list, description="Lista de renglones pedidos.")
    descuento_solicitado: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Porcentaje de descuento especial que solicita el cliente, si lo menciona explícitamente."
    )
    pago_anticipado: bool = Field(
        default=False,
        description="True si el cliente menciona que paga por adelantado, transferencia inmediata o contado."
    )
    es_pedido: bool = Field(
        default=True,
        description="Indica si efectivamente se detectaron ítems y cantidades para procesar como pedido."
    )


class ClasificacionMensaje(BaseModel):
    """Salida estructurada del LLM para clasificar la intención del cliente."""
    intencion: Literal["consulta", "pedido", "derivar"] = Field(
        description="Intención principal del mensaje: 'consulta' (preguntas de políticas), 'pedido' (compra) o 'derivar' (reclamos, cancelaciones, otros)."
    )
    confianza: float = Field(
        ge=0.0,
        le=1.0,
        description="Nivel de certidumbre de la clasificación (0.0 a 1.0)."
    )
    justificacion: str = Field(
        default="",
        description="Breve explicación de por qué se asignó esta intención."
    )


class DecisionHumana(BaseModel):
    """Decisión ingresada por Gerencia Comercial en el nodo Human-in-the-Loop."""
    tipo: Literal["aprobar", "editar", "rechazar"] = Field(
        description="Acción decidida por el operador: 'aprobar', 'editar' o 'rechazar'."
    )
    razon: str = Field(default="", description="Motivo de la aprobación, edición o rechazo.")
    ediciones: dict[str, int] = Field(
        default_factory=dict,
        description="En caso de editar: mapa {sku: nueva_cantidad} (cantidad 0 elimina el renglón)."
    )
    descuento_autorizado: float | None = Field(
        default=None,
        description="Descuento especial autorizado en caso de edición."
    )
