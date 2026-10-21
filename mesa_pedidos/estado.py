# -*- coding: utf-8 -*-
"""
estado.py - Definición del Estado Compartido en LangGraph (EstadoMesa).

El Estado es un diccionario tipado (`TypedDict`) que circula a través de los nodos del grafo.
Es el único canal mediante el cual los nodos intercambian información en LangGraph.

Conceptos Clave de Arquitectura:
1. Sin Reducer (Sobrescritura):
   Los campos normales (`ruta`, `respuesta`, `validacion`, etc.) se sobrescriben cuando
   un nodo devuelve un nuevo valor para esa clave. Siempre prevalece la última decisión.
2. Con Reducer (`Annotated[list[str], add]`):
   Para los campos de auditoría (`log` y `errores`), utilizamos el operador `add` de Python.
   Cada vez que un nodo retorna `{"log": ["..."]}`, LangGraph CONCATENA esos elementos
   a la lista acumulada existente en lugar de pisarla. Esto permite auditar paso a paso
   el recorrido del mensaje por cada uno de los nodos.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Optional, TypedDict

from mesa_pedidos.esquemas import MensajeEntrante


class EstadoMesa(TypedDict, total=False):
    """Estado global que fluye por la máquina de estados de LangGraph."""

    # --- 1. DATOS DE ENTRADA (cargados al inicio) ---
    mensaje_id: str                      # Identificador único (ej: "MSG-001")
    canal: str                           # Canal de origen: "whatsapp" | "email" | "web"
    fecha: str                           # Fecha de emisión del mensaje
    cuit: str                            # CUIT informado por el remitente
    remitente: str                       # Nombre o razón social del remitente
    texto: str                           # Texto en lenguaje natural del mensaje

    # --- 2. INFORMACIÓN MAESTRA DE NEGOCIO ---
    cliente: Optional[dict[str, Any]]    # Ficha del cliente de clientes.csv o None si no existe

    # --- 3. DECISIONES Y EVALUACIONES INTERMEDIAS ---
    clasificacion: dict[str, Any]        # Salida estructurada de ClasificacionMensaje
    intencion: str                       # Intención efectiva: "consulta" | "pedido" | "derivar"
    pedido: Optional[dict[str, Any]]     # Items extraídos por el LLM (PedidoExtraido)
    validacion: Optional[dict[str, Any]] # Resultado determinístico de validación comercial
    decision_humana: Optional[dict[str, Any]]  # Resolución ingresada en interrupt() por el operador

    # --- 4. SALIDAS Y ENTREGABLES FINALES ---
    ruta: str                            # Nodo terminal u hoja alcanzada
    respuesta: str                       # Mensaje final redactado para el cliente
    citas: list[dict[str, Any]]          # Documento y página de cada cita (RAG)
    fragmentos: list[dict[str, Any]]     # Fragmentos contextuales recuperados
    resumen: Optional[dict[str, Any]]    # ResumenPedido estructurado (confirmado o rechazado)
    ticket: Optional[dict[str, Any]]     # Ticket generado en caso de derivar a humano

    # --- 5. AUDITORÍA Y REGISTRO (CON REDUCER OBLIGATORIO) ---
    log: Annotated[list[str], add]       # Bitácora acumulativa de cada nodo por el que pasa
    errores: Annotated[list[str], add]   # Registro acumulativo de fallos o excepciones


def estado_inicial(mensaje: MensajeEntrante) -> EstadoMesa:
    """Construye el estado inicial a partir de un mensaje entrante de la bandeja."""
    return {
        "mensaje_id": mensaje.id,
        "canal": mensaje.canal,
        "fecha": mensaje.fecha,
        "cuit": mensaje.cuit,
        "remitente": mensaje.remitente,
        "texto": mensaje.texto,
        "log": [f"Mensaje {mensaje.id} ingresado por canal {mensaje.canal}"],
        "errores": [],
    }
