# -*- coding: utf-8 -*-
"""
indexar.py - Ingesta, chunking semántico y generación del índice vectorial FAISS.

Convierte los 4 PDFs de políticas de la empresa en un índice vectorial persistido.
Permite buscar los fragmentos más relevantes para responder consultas de clientes
citando con precisión el documento y la página.

Flujo del pipeline:
    cargar_paginas()  ->  partir_en_chunks()  ->  construir_indice()
    PDF -> Document       Document -> chunks      chunks -> FAISS en disco

Estrategia de chunking:
- "articulos" (por defecto): cada chunk representa un artículo comercial o ficha técnica
  de producto (SKU). Esto garantiza que la regla de negocio y sus excepciones queden
  juntas en el mismo bloque semántico.
- "recursivo": partición por tamaño fijo (RecursiveCharacterTextSplitter), útil para comparar.

Uso desde terminal:
    py -m mesa_pedidos.indexar --ver       # Inspeccionar los chunks generados (sin llamar a la API)
    py -m mesa_pedidos.indexar             # Construir el índice vectorial FAISS
    py -m mesa_pedidos.indexar --forzar    # Reconstruir el índice forzosamente
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from langchain_core.documents import Document

import config

MANIFIESTO = config.INDICE_FAISS / "manifiesto.json"

# Expresiones regulares para detectar encabezados semánticos en los PDFs
_RE_ARTICULO = re.compile(r"^ART\. \d+ - ")                 # Condiciones comerciales (ej. "ART. 4 - ")
_RE_PUNTO = re.compile(r"^\d+\. [A-Z][A-Z ,]+$")            # Logística y devoluciones (ej. "1. ENTREGAS...")
_RE_SKU = re.compile(r"^[A-Z]{2}-\d{4} - ")                 # Fichas técnicas (ej. "HE-4330 - Atornillador...")
_RE_RUBRO = re.compile(r"^[A-Z][A-Z ,]+$")                  # Categoría de producto (ej. "HERRAMIENTA ELECTRICA")


# ===========================================================================
# Paso 1: Cargar páginas desde los PDFs
# ===========================================================================

def cargar_paginas() -> list[Document]:
    """Carga los 4 PDFs de `datos/politicas/`.
    
    Devuelve una lista de objetos Document (uno por página), asignando
    metadatos normalizados:
      - documento: nombre del archivo PDF (ej. 'logistica_fletes_2026.pdf')
      - pagina: número de página humano (iniciando en 1)
    """
    from langchain_community.document_loaders import PyPDFLoader

    paginas: list[Document] = []
    archivos_pdf = sorted(config.POLITICAS.glob("*.pdf"))
    if not archivos_pdf:
        raise FileNotFoundError(f"No se encontraron archivos PDF en {config.POLITICAS}")

    for pdf in archivos_pdf:
        cargador = PyPDFLoader(str(pdf))
        for doc in cargador.load():
            doc.metadata["documento"] = pdf.name
            # PyPDFLoader cuenta desde 0; normalizamos a 1 para citar correctamente
            doc.metadata["pagina"] = int(doc.metadata.get("page", 0)) + 1
            paginas.append(doc)
    return paginas


# ===========================================================================
# Paso 2: Partir en chunks semánticos (por Artículos / SKUs)
# ===========================================================================

def _lineas_utiles(paginas: list[Document]) -> tuple[str, list[tuple[int, str]]]:
    """Filtra encabezados repetitivos y pies de página comerciales.
    
    Retorna:
      (titulo_documento, [(numero_pagina, texto_linea), ...])
    """
    titulo = ""
    lineas: list[tuple[int, str]] = []
    saltear_siguiente = False

    for pag in paginas:
        num_pag = pag.metadata["pagina"]
        for i, cruda in enumerate(pag.page_content.splitlines()):
            linea = cruda.strip()
            if not linea:
                continue
            if saltear_siguiente:
                saltear_siguiente = False
                if linea == "Comercial.":
                    continue
            # En la primera página, la 2da línea suele ser el título formal
            if num_pag == 1 and i <= 1:
                if i == 1:
                    titulo = linea
                continue
            # Pie recurrente en los PDFs de la empresa
            if linea.startswith("Documento interno de uso comercial"):
                saltear_siguiente = True
                continue
            lineas.append((num_pag, linea))

    return titulo, lineas


def _chunks_por_articulo(paginas: list[Document]) -> list[Document]:
    """Genera chunks agrupando por artículo, punto o SKU técnico.
    
    A cada chunk le antepone: '<Título Documento> | <Encabezado>'
    para que la búsqueda vectorial y léxica entienda el contexto temático.
    """
    documento = paginas[0].metadata["documento"]
    titulo, lineas = _lineas_utiles(paginas)
    es_ficha = documento.startswith("fichas")

    chunks: list[Document] = []
    rubro = ""
    actual: dict[str, Any] | None = None

    def cerrar_chunk() -> None:
        if actual and actual["lineas"]:
            cuerpo = " ".join(actual["lineas"])
            encabezado = f"{titulo} | {rubro}" if es_ficha else titulo
            p_inicio, p_fin = actual["pagina"], actual["ultima"]
            chunks.append(Document(
                page_content=f"{encabezado}\n{cuerpo}",
                metadata={
                    "source": str(config.POLITICAS / documento),
                    "documento": documento,
                    "pagina": p_inicio,
                    "paginas": f"{p_inicio}-{p_fin}" if p_fin != p_inicio else str(p_inicio),
                    "seccion": actual["seccion"],
                },
            ))

    for pagina, linea in lineas:
        abre_seccion = (_RE_SKU.match(linea) if es_ficha
                        else _RE_ARTICULO.match(linea) or _RE_PUNTO.match(linea))

        if es_ficha and not abre_seccion and _RE_RUBRO.match(linea) and len(linea) < 60:
            cerrar_chunk()
            actual, rubro = None, linea
            continue

        if abre_seccion:
            cerrar_chunk()
            seccion = linea.split(" - ")[0] if es_ficha else linea
            actual = {
                "seccion": seccion,
                "pagina": pagina,
                "ultima": pagina,
                "lineas": [linea],
            }
        elif actual is not None:
            actual["lineas"].append(linea)
            actual["ultima"] = pagina

    cerrar_chunk()
    return chunks


def _chunks_recursivos(paginas: list[Document]) -> list[Document]:
    """Estrategia alternativa de partición por tamaño fijo (para comparar)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(paginas)
    for c in chunks:
        c.metadata["paginas"] = str(c.metadata["pagina"])
        c.metadata["seccion"] = c.page_content.strip().splitlines()[0][:60]
    return chunks


def partir_en_chunks(paginas: list[Document] | None = None,
                     estrategia: str | None = None) -> list[Document]:
    """Segmenta las páginas en fragmentos atómicos según la estrategia configurada."""
    paginas = paginas if paginas is not None else cargar_paginas()
    estrategia = estrategia or config.ESTRATEGIA_CHUNKING

    if estrategia == "recursivo":
        return _chunks_recursivos(paginas)

    # Agrupar páginas por documento para procesar cada uno por separado
    por_documento: dict[str, list[Document]] = {}
    for p in paginas:
        por_documento.setdefault(p.metadata["documento"], []).append(p)

    chunks: list[Document] = []
    for paginas_doc in por_documento.values():
        chunks.extend(_chunks_por_articulo(paginas_doc))

    # Red de seguridad: si un artículo fuese inusualmente largo, se subdivide
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=0)
    final: list[Document] = []
    for c in chunks:
        if len(c.page_content) <= config.CHUNK_SIZE + 300:
            final.append(c)
        else:
            final.extend(splitter.split_documents([c]))
    return final


# ===========================================================================
# Paso 3: Embeddings, Indexación con FAISS y Persistencia
# ===========================================================================

def obtener_embeddings() -> Any:
    """Instancia el modelo de embeddings de Google Gemini (models/text-embedding-004)
    usando la clave cargada en config.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    clave = config.obtener_clave_api()
    return GoogleGenerativeAIEmbeddings(
        model=config.MODELO_EMBEDDINGS,
        google_api_key=clave,
    )


def _huella() -> dict[str, Any]:
    """Calcula la 'huella digital' de la configuración y de los archivos PDF.
    Si cualquier PDF o parámetro de chunking cambia, se debe reindexar.
    """
    return {
        "modelo_embeddings": config.MODELO_EMBEDDINGS,
        "estrategia": config.ESTRATEGIA_CHUNKING,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "pdfs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(config.POLITICAS.glob("*.pdf"))
        },
    }


def indice_al_dia() -> bool:
    """Verifica si el índice FAISS persistido en disco corresponde exactamente
    al estado actual de los PDFs y a la configuración.
    """
    archivo_faiss = config.INDICE_FAISS / "index.faiss"
    if not archivo_faiss.exists() or not MANIFIESTO.exists():
        return False
    guardado = json.loads(MANIFIESTO.read_text(encoding="utf-8"))
    return {k: guardado.get(k) for k in _huella()} == _huella()


def construir_indice(forzar: bool = False) -> Any:
    """Carga el índice FAISS si ya existe y está al día; de lo contrario,
    genera los embeddings y persiste el nuevo índice en `artefactos/indice_faiss/`.
    """
    from langchain_community.vectorstores import FAISS

    if not forzar and indice_al_dia():
        # allow_dangerous_deserialization: FAISS guarda metadatos con pickle.
        # Es seguro porque es un archivo generado internamente por nosotros.
        return FAISS.load_local(
            str(config.INDICE_FAISS),
            obtener_embeddings(),
            allow_dangerous_deserialization=True,
        )

    chunks = partir_en_chunks()
    emb = obtener_embeddings()
    vs = FAISS.from_documents(chunks, emb)

    config.INDICE_FAISS.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(config.INDICE_FAISS))

    # Guardar manifiesto con metadatos del build
    manifiesto_data = {
        **_huella(),
        "n_chunks": len(chunks),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    MANIFIESTO.write_text(json.dumps(manifiesto_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return vs


# ===========================================================================
# Ejecución CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa las políticas en un índice FAISS")
    parser.add_argument("--forzar", action="store_true", help="Reconstruir el índice aunque esté al día")
    parser.add_argument("--ver", action="store_true", help="Mostrar los chunks generados sin llamar a la API")
    args = parser.parse_args()

    # Opción 1: Solo inspeccionar los chunks (Modo seguro, 0 tokens)
    if args.ver:
        chunks = partir_en_chunks()
        print(f"\n{'='*70}")
        print(f"CHUNKS GENERADOS: {len(chunks)} fragmentos (Estrategia: '{config.ESTRATEGIA_CHUNKING}')")
        print(f"{'='*70}\n")
        for i, c in enumerate(chunks, start=1):
            m = c.metadata
            print(f"[{i}] {m['documento']} (Pág. {m['paginas']}) | Sección: {m['seccion']} ({len(c.page_content)} caracteres)")
            print("-" * 50)
            preview = c.page_content[:250].replace('\n', ' ')
            print(f"{preview}...")
            print()
        return

    # Opción 2: Indexación real hacia FAISS
    config.verificar_entorno()
    estaba_al_dia = indice_al_dia()
    vs = construir_indice(forzar=args.forzar)
    estado = "reabierto (ya estaba al día)" if estaba_al_dia and not args.forzar else "construido y guardado"
    print(f"\n[OK] Índice FAISS {estado} en: {config.INDICE_FAISS}")
    print(f"[OK] Total de vectores indexados: {vs.index.ntotal}")


if __name__ == "__main__":
    main()
