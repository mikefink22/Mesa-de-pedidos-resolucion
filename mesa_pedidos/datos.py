# -*- coding: utf-8 -*-
"""
datos.py - Capa de acceso a datos maestros y normalización de texto.

La única parte del sistema que abre y consulta archivos de datos (CSVs y JSONL).
Las tools y los nodos del grafo llaman a estas funciones para acceder a la
información del negocio sin depender de estructuras de base de datos externas.

Características:
- Carga en memoria en tiempo O(1) con diccionarios tipados mediante Pydantic.
- Normalización robusta de CUITs (sin guiones) y texto (minúsculas, sin tildes).
- Búsqueda laxa de productos por palabras clave cuando el cliente no especifica SKU.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import config
from mesa_pedidos.esquemas import Cliente, MensajeEntrante, Producto

_CACHE: dict[str, Any] = {}


# ===========================================================================
# 1. Utilidades de Normalización
# ===========================================================================

def normalizar_cuit(cuit: str) -> str:
    """Extrae solo los dígitos del CUIT para búsquedas consistentes sin importar formato."""
    return "".join(c for c in (cuit or "") if c.isdigit())


def normalizar_texto(texto: str) -> str:
    """Normaliza un texto para comparaciones léxicas:
    - Convierte a minúsculas.
    - Elimina tildes y diacríticos (ej: 'Neuquén' -> 'neuquen').
    - Colapsa espacios múltiples.
    """
    if not texto:
        return ""
    texto_descompuesto = unicodedata.normalize("NFKD", texto)
    texto_sin_tildes = "".join(c for c in texto_descompuesto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto_sin_tildes.lower()).strip()


def _leer_csv(ruta: Path) -> list[dict[str, str]]:
    """Lee un archivo CSV con encabezados usando utf-8-sig para tolerar BOM de Excel."""
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        return [dict(fila) for fila in csv.DictReader(f)]


# ===========================================================================
# 2. Catálogo de Productos
# ===========================================================================

def _productos() -> dict[str, Producto]:
    """Carga y cachea catalogo.csv indexado por SKU en mayúsculas."""
    if "productos" not in _CACHE:
        _CACHE["productos"] = {
            f["sku"].strip().upper(): Producto(
                sku=f["sku"].strip().upper(),
                nombre=f["nombre"],
                rubro=f["rubro"],
                marca=f["marca"],
                unidad=f["unidad"],
                precio_lista=float(f["precio_lista"]),
                alicuota_iva=float(f["alicuota_iva"]),
                stock=int(f["stock"]),
                deposito=f["deposito"],
            )
            for f in _leer_csv(config.CATALOGO_CSV)
        }
    return _CACHE["productos"]


def buscar_producto(sku: str) -> Optional[Producto]:
    """Búsqueda exacta por SKU (insensible a mayúsculas y espacios)."""
    return _productos().get((sku or "").strip().upper().replace(" ", ""))


def buscar_skus_similares(sku_buscado: str, limite: int = 3, cutoff: float = 0.75) -> list[Producto]:
    """Busca SKUs parecidos por similitud tipográfica (difflib) para tolerar errores de tipeo.
    Solo considera candidatos con al menos 75% de similitud; si tiene menos, no sugiere nada.
    """
    sku_limpio = (sku_buscado or "").strip().upper().replace(" ", "")
    todos = _productos()
    candidatos = difflib.get_close_matches(sku_limpio, list(todos.keys()), n=limite, cutoff=cutoff)
    return [todos[s] for s in candidatos]


def listar_productos() -> list[Producto]:
    """Lista completa de productos ordenados por SKU."""
    return sorted(_productos().values(), key=lambda p: p.sku)


_PALABRAS_VACIAS = {
    "de", "del", "la", "el", "los", "las", "para", "con", "sin", "por",
    "una", "uno", "unos", "unas", "x", "mm", "par", "pares", "unidad", "unidades"
}


def _raiz(palabra: str) -> str:
    """Stemming elemental en español: 'discos' -> 'disco', 'amoladoras' -> 'amoladora'."""
    for sufijo in ("es", "s"):
        if len(palabra) > 4 and palabra.endswith(sufijo):
            return palabra[:-len(sufijo)]
    return palabra


def _tokens(texto: str) -> set[str]:
    """Tokeniza y extrae raíces significativas para búsqueda léxica."""
    palabras = re.findall(r"[a-z0-9]+", normalizar_texto(texto))
    return {_raiz(p) for p in palabras if len(p) > 2 and p not in _PALABRAS_VACIAS}


def buscar_productos_por_texto(consulta: str, limite: int = 3) -> list[Producto]:
    """Búsqueda laxa por nombre, rubro y marca para cuando el cliente no indicó SKU.
    Calcula la proporción de palabras clave que coinciden con la ficha del producto.
    """
    buscadas = _tokens(consulta)
    if not buscadas:
        return []
    puntuados: list[tuple[float, Producto]] = []
    for prod in _productos().values():
        del_producto = _tokens(f"{prod.nombre} {prod.rubro} {prod.marca}")
        puntaje = len(buscadas & del_producto) / len(buscadas)
        if puntaje >= 0.5:
            puntuados.append((puntaje, prod))
    if not puntuados:
        return []
    mejor = max(p for p, _ in puntuados)
    return sorted((p for puntaje, p in puntuados if puntaje == mejor), key=lambda p: p.sku)[:limite]


# ===========================================================================
# 3. Base de Clientes
# ===========================================================================

def _clientes() -> dict[str, Cliente]:
    """Carga y cachea clientes.csv indexado por CUIT numérico."""
    if "clientes" not in _CACHE:
        _CACHE["clientes"] = {
            normalizar_cuit(f["cuit"]): Cliente(
                cuit=f["cuit"].strip(),
                razon_social=f["razon_social"],
                provincia=f["provincia"],
                localidad=f["localidad"],
                condicion_iva=f["condicion_iva"],
                categoria=f["categoria"].strip(),
                limite_credito=float(f["limite_credito"]),
                saldo_actual=float(f["saldo_actual"]),
                dias_plazo=int(f["dias_plazo"]),
                descuento_habitual=float(f["descuento_habitual"]),
                estado=f["estado"].strip(),
                mora_dias=int(f["mora_dias"]),
            )
            for f in _leer_csv(config.CLIENTES_CSV)
        }
    return _CACHE["clientes"]


def buscar_cliente(cuit: str) -> Optional[Cliente]:
    """Obtiene la ficha del cliente por CUIT (con o sin guiones), o None si no existe."""
    return _clientes().get(normalizar_cuit(cuit))


def listar_clientes() -> list[Cliente]:
    """Lista completa de clientes ordenada alfabéticamente por razón social."""
    return sorted(_clientes().values(), key=lambda c: c.razon_social)


def limpiar_cache() -> None:
    """Invalida la caché en memoria (útil si se modifican los CSVs durante la ejecución)."""
    _CACHE.clear()


# ===========================================================================
# 4. Bandeja de Entrada, Golden Set y Compatibilidad con Pandas
# ===========================================================================

def cargar_bandeja() -> list[MensajeEntrante]:
    """Carga los 20 mensajes de la bandeja de entrada (bandeja.jsonl)."""
    with config.BANDEJA.open(encoding="utf-8") as f:
        return [MensajeEntrante(**json.loads(linea)) for linea in f if linea.strip()]


def buscar_mensaje(mensaje_id: str) -> Optional[MensajeEntrante]:
    """Busca un mensaje específico en la bandeja por su ID (ej. 'MSG-009')."""
    for m in cargar_bandeja():
        if m.id.upper() == mensaje_id.upper():
            return m
    return None


def cargar_golden_set() -> dict[str, Any]:
    """Carga el dataset de evaluación golden_set.json."""
    return json.loads(config.GOLDEN_SET.read_text(encoding="utf-8"))


def caso_golden(caso_id: str) -> Optional[dict[str, Any]]:
    """Busca un caso específico en el golden_set por su ID (ej. 'MSG-009')."""
    for c in cargar_golden_set().get("casos", []):
        if c.get("id", "").upper() == caso_id.upper():
            return c
    return None


def guardar_salida(nombre: str, objeto: Any) -> Path:
    """Escribe un JSON en artefactos/salidas/ y devuelve la ruta absoluta del archivo."""
    ruta = config.SALIDAS / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(objeto, "model_dump"):
        objeto = objeto.model_dump()
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ruta


def cargar_catalogo() -> pd.DataFrame:
    """Compatibilidad: Retorna el catálogo como pandas DataFrame."""
    return pd.read_csv(config.CATALOGO_CSV, encoding="utf-8")


def cargar_clientes() -> pd.DataFrame:
    """Compatibilidad: Retorna la cartera de clientes como pandas DataFrame."""
    return pd.read_csv(config.CLIENTES_CSV, encoding="utf-8")
