# -*- coding: utf-8 -*-
"""
config.py - Configuración central del proyecto Mesa de Pedidos.

Centraliza rutas, modelos de IA y reglas de negocio.
Permite alternar de proveedor (Google Gemini u OpenAI) y modificar
los modelos directamente acá en el código.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Directorio raíz del proyecto
RAIZ = Path(__file__).resolve().parent
load_dotenv(RAIZ / ".env")

# ---------------------------------------------------------------------------
# 1. Configuración de Modelos y Proveedores
# ---------------------------------------------------------------------------
# Proveedor activo: "google" u "openai" (configurable acá o por variable de entorno)
PROVEEDOR = os.getenv("PROVEEDOR_IA", "google").lower()

# Catálogo de modelos y claves requeridas por proveedor
CONFIG_PROVEEDORES = {
    "google": {
        "llm": "gemini-1.5-flash",
        "embeddings": "models/text-embedding-004",
        "env_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    },
    "openai": {
        "llm": "gpt-4o-mini",
        "embeddings": "text-embedding-3-small",
        "env_keys": ["OPENAI_API_KEY"],
    },
}

# Modelos activos (visibles y editables directamente en el código)
_cfg_actual = CONFIG_PROVEEDORES.get(PROVEEDOR, CONFIG_PROVEEDORES["google"])
MODELO_LLM = os.getenv("MODELO_LLM", _cfg_actual["llm"])
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", _cfg_actual["embeddings"])
TEMPERATURA = 0.0

# ---------------------------------------------------------------------------
# 2. Rutas del Sistema
# ---------------------------------------------------------------------------
# Soporta tanto si los archivos están en la raíz como si se mueven a datos/
DATOS = RAIZ / "datos" if (RAIZ / "datos").exists() else RAIZ
POLITICAS = DATOS / "politicas"          # los 4 PDF: el corpus del RAG
CATALOGO_CSV = DATOS / "catalogo.csv"    # 40 productos
CLIENTES_CSV = DATOS / "clientes.csv"    # 12 clientes
BANDEJA = DATOS / "bandeja.jsonl"        # los 20 mensajes de entrada
GOLDEN_SET = DATOS / "golden_set.json"   # casos de referencia

# Lo que genera el sistema (FAISS, checkpoints, trazas). No se versionan.
ARTEFACTOS = RAIZ / "artefactos"
INDICE_FAISS = ARTEFACTOS / "indice_faiss"
CHECKPOINTS_DB = ARTEFACTOS / "checkpoints.sqlite"
SALIDAS = ARTEFACTOS / "salidas"

for _carpeta in (ARTEFACTOS, SALIDAS):
    _carpeta.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Parámetros de RAG
# ---------------------------------------------------------------------------
ESTRATEGIA_CHUNKING = "articulos"   # "articulos" o "recursivo"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
K_RECUPERACION = 4
RETRIEVER = "hibrido"               # "hibrido" (BM25 + denso) o "denso"
PESOS_HIBRIDO = (0.4, 0.6)          # (BM25, denso)

# ---------------------------------------------------------------------------
# 4. Reglas de Negocio (extraídas de las políticas de la empresa)
# ---------------------------------------------------------------------------
# Pedido mínimo por zona (neto sin IVA)
PEDIDO_MINIMO = {"A": 250_000, "B": 250_000, "C": 120_000}

# Escala de descuentos por volumen: [(monto_minimo, porcentaje)]
ESCALA_VOLUMEN = [
    (6_000_000, 12.0),
    (3_000_000, 8.0),
    (1_000_000, 5.0),
    (0, 0.0),
]

# Topes de descuento
TOPE_DESCUENTO_GENERAL = 20.0
TOPE_DESCUENTO_MARCAS = 15.0
RUBRO_CON_TOPE_MARCA = "Herramienta electrica"
MARCAS_CON_TOPE = {"dewalt", "makita", "bosch"}

# Descuento financiero por pago anticipado
DESCUENTO_PAGO_ANTICIPADO = 3.0

# Reglas de mora en días
MORA_AUTORIZACION_DIAS = 30   # 30 a 60 días: requiere autorización gerencial
MORA_SUSPENSION_DIAS = 60     # >60 días: suspendido (solo pago anticipado)

# Umbral de confianza mínima para clasificación
CONFIANZA_MINIMA = 0.6

# Datos fiscales de la empresa
EMPRESA = "Ferretería Industrial Andina S.R.L."
CUIT_EMPRESA = "30-71204598-8"


# ---------------------------------------------------------------------------
# 5. Funciones de Validación de Entorno
# ---------------------------------------------------------------------------
def obtener_clave_api(proveedor: str = PROVEEDOR) -> str | None:
    """Busca la clave correspondiente al proveedor configurado."""
    keys = CONFIG_PROVEEDORES.get(proveedor, {}).get("env_keys", [])
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return None


def hay_clave(proveedor: str = PROVEEDOR) -> bool:
    """Verifica si la clave de API está presente en las variables de entorno."""
    return bool(obtener_clave_api(proveedor))


def verificar_entorno() -> None:
    """Valida el entorno antes de iniciar la ejecución con un mensaje claro."""
    if not hay_clave():
        keys_esperadas = " o ".join(CONFIG_PROVEEDORES.get(PROVEEDOR, {}).get("env_keys", ["API_KEY"]))
        raise RuntimeError(
            f"Falta la clave de API para el proveedor '{PROVEEDOR}'. "
            f"Asegúrate de definir {keys_esperadas} en tu archivo .env."
        )
