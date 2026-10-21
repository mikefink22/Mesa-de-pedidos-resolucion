# -*- coding: utf-8 -*-
"""
ver_modelos.py - Script de diagnóstico para consultar el catálogo de modelos disponibles.

Consulta directamente a la API de Google Gemini (v1beta) usando la API Key configurada
y lista todos los modelos activos para esa cuenta, filtrando:
1. Modelos que soportan generación de embeddings (embedContent).
2. Modelos que soportan generación de texto / LLM (generateContent).

Uso:
    py -m mesa_pedidos.ver_modelos
"""

from __future__ import annotations

import sys
import config

# Asegurar codificación utf-8 en consolas Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def listar_modelos_disponibles() -> None:
    """Consulta la API de Google y muestra los modelos habilitados para la clave."""
    config.verificar_entorno()
    clave = config.obtener_clave_api()

    # Usamos el cliente oficial de Google Generative Language
    try:
        from google.ai.generativelanguage_v1beta import ModelServiceClient
        from google.api_core.client_options import ClientOptions
        
        # Inicializar cliente con la clave de API
        client = ModelServiceClient(
            client_options=ClientOptions(api_key=clave)
        )
        
        print(f"\n{'='*75}")
        print("CONSULTANDO MODELOS DISPONIBLES EN TU CUENTA DE GOOGLE GEMINI...")
        print(f"{'='*75}\n")
        
        modelos_embedding = []
        modelos_texto = []
        
        for m in client.list_models():
            metodos = m.supported_generation_methods
            nombre = m.name  # ej. "models/text-embedding-004"
            
            if "embedContent" in metodos or "batchEmbedContents" in metodos:
                modelos_embedding.append((nombre, metodos))
            elif "generateContent" in metodos:
                modelos_texto.append((nombre, m.display_name))
        
        # 1. Modelos de Embeddings
        print("[EMBEDDINGS] Modelos para RAG y busqueda vectorial:")
        if modelos_embedding:
            for nom, met in modelos_embedding:
                print(f"  - Nombre API: '{nom}' (o '{nom.replace('models/', '')}')")
                print(f"    Metodos soportados: {', '.join(met)}")
        else:
            print("  [AVISO] No se encontraron modelos con 'embedContent' en este proyecto.")
            
        print("\n" + "-"*75 + "\n")
        
        # 2. Modelos de Texto / LLM
        print("[LLM] Modelos para clasificacion y redaccion:")
        for nom, desc in modelos_texto[:10]:
            print(f"  - '{nom}' ({desc})")
        if len(modelos_texto) > 10:
            print(f"  ... y {len(modelos_texto) - 10} modelos mas.")
            
        print(f"\n{'='*75}\n")
        
    except Exception as e:
        print(f"\n[ERROR] al consultar la API de Google: {e}\n")


if __name__ == "__main__":
    listar_modelos_disponibles()
