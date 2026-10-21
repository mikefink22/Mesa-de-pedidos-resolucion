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
# 1. Configuración de Modelos (Google Gemini - Stack Oficial)
# ---------------------------------------------------------------------------
PROVEEDOR = "google"
MODELO_LLM = "gemini-3.8-flash"
MODELO_EMBEDDINGS = "models/gemini-embedding-001"
TEMPERATURA = 0.0

# Nombres de variables de entorno aceptadas para Google Gemini
CLAVES_ENV_GOOGLE = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]

# (Opcional: Respaldo para OpenAI, dejado como referencia comentada)
# PROVEEDOR_BACKUP = "openai"
# MODELO_LLM_BACKUP = "gpt-4o-mini"
# MODELO_EMBEDDINGS_BACKUP = "text-embedding-3-small"

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
def obtener_clave_api() -> str | None:
    """Busca la clave de Google Gemini en las variables de entorno."""
    for key in CLAVES_ENV_GOOGLE:
        val = os.getenv(key)
        if val:
            return val
    return None


def hay_clave() -> bool:
    """Verifica si la clave de API está presente en las variables de entorno."""
    return bool(obtener_clave_api())


def verificar_entorno() -> None:
    """Valida el entorno antes de iniciar la ejecución con un mensaje claro."""
    if not hay_clave():
        keys_esperadas = " o ".join(CLAVES_ENV_GOOGLE)
        raise RuntimeError(
            f"Falta la clave de API de Google Gemini. "
            f"Asegúrate de definir {keys_esperadas} en tu archivo .env."
        )
