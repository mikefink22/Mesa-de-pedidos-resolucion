# -*- coding: utf-8 -*-
"""
llm.py - Fábrica centralizada de modelos de lenguaje (LLM).

Provee la función `chat()` para instanciar el cliente de chat:
- Sin argumentos: Devuelve el modelo base (para generación de texto libre).
- Con schema (Pydantic): Devuelve el modelo con Structured Output tipado.

Utiliza Google Gemini a través de `langchain-google-genai` respetando
la configuración centralizada de `config.py`.
"""

from __future__ import annotations

from typing import Any, Optional
import config


def chat(schema: Optional[type] = None) -> Any:
    """Retorna una instancia configurada de ChatGoogleGenerativeAI.
    
    Si se provee un modelo Pydantic en `schema`, se configura
    la salida estructurada mediante `with_structured_output`.
    """
    config.verificar_entorno()
    
    from langchain_google_genai import ChatGoogleGenerativeAI

    modelo = ChatGoogleGenerativeAI(
        model=config.MODELO_LLM,
        google_api_key=config.obtener_clave_api(),
        temperature=config.TEMPERATURA,
    )

    if schema is None:
        return modelo
    return modelo.with_structured_output(schema)
