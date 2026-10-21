# -*- coding: utf-8 -*-
"""
datos.py - Utilidades de carga de datos y normalización de texto.

Provee funciones para:
1. Normalizar texto (eliminar acentos, minúsculas, espacios) para búsquedas robustas.
2. Carga y caché de los CSVs de catálogo y clientes.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import config


def normalizar_texto(texto: str) -> str:
    """Normaliza un texto para comparaciones léxicas:
    - Convierte a minúsculas.
    - Elimina tildes y diacríticos (ej: 'Neuquén' -> 'neuquen').
    - Colapsa espacios múltiples.
    """
    if not texto:
        return ""
    # Descomponer caracteres con acentos
    texto_descompuesto = unicodedata.normalize("NFKD", texto)
    # Filtrar solo caracteres ASCII
    texto_sin_tildes = "".join(c for c in texto_descompuesto if not unicodedata.combining(c))
    # Limpiar espacios
    return re.sub(r"\s+", " ", texto_sin_tildes.lower()).strip()


@functools.lru_cache(maxsize=1)
def cargar_catalogo() -> pd.DataFrame:
    """Carga catalogo.csv en memoria. Retorna un DataFrame con columnas en tipos correctos."""
    if not config.CATALOGO_CSV.exists():
        raise FileNotFoundError(f"No se encontró el catálogo en {config.CATALOGO_CSV}")
    
    df = pd.read_csv(config.CATALOGO_CSV, encoding="utf-8")
    # Limpieza básica de columnas numéricas si hiciera falta
    return df


@functools.lru_cache(maxsize=1)
def cargar_clientes() -> pd.DataFrame:
    """Carga clientes.csv en memoria."""
    if not config.CLIENTES_CSV.exists():
        raise FileNotFoundError(f"No se encontró clientes.csv en {config.CLIENTES_CSV}")
    
    df = pd.read_csv(config.CLIENTES_CSV, encoding="utf-8")
    return df
