# -*- coding: utf-8 -*-
"""
recuperacion.py - Recuperación vectorial, híbrida y generación de respuestas RAG con citas.

Implementa los tres modos de recuperación:
1. "denso": Búsqueda vectorial sobre FAISS usando MMR (Maximal Marginal Relevance)
   para maximizar la diversidad temática de los fragmentos recuperados.
2. "bm25": Búsqueda léxica por palabras exactas, ideal para códigos de SKU (ej. HE-4330).
3. "hibrido": Fusión de ambos rankings mediante EnsembleRetriever ponderado.

Mecanismo de Citación Anti-Alucinación:
- El LLM recibe los fragmentos numerados [1], [2], [3]...
- El LLM devuelve en Structured Output (RespuestaConsulta) la lista de números que utilizó: [1, 3].
- Python traduce esos números a metadatos reales y auditables (documento, página, sección).

Uso:
    py -m mesa_pedidos.recuperacion "¿Cuánto sale el flete a Neuquén?"
    py -m mesa_pedidos.recuperacion --solo-buscar "HE-4330"
    py -m mesa_pedidos.recuperacion --modo bm25 "devolución calzado"
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from langchain_core.documents import Document

import config
from mesa_pedidos.datos import normalizar_texto
from mesa_pedidos.esquemas import RespuestaConsulta

MODOS = ("denso", "bm25", "hibrido")
_CACHE: dict[str, Any] = {}


# ===========================================================================
# 1. Tokenizador para Búsqueda Léxica (BM25)
# ===========================================================================

def tokenizar(texto: str) -> list[str]:
    """Tokenizador especializado para BM25:
    - Normaliza caracteres (sin tildes, minúsculas).
    - Preserva los códigos de SKU completos (ej. 'he-4330') evitando que los signos
      de puntuación (ej: '¿HE-4330?') rompan la coincidencia léxica.
    """
    return re.findall(r"[a-z]{2}-\d{4}|[a-z0-9]+", normalizar_texto(texto))


def _chunks() -> list[Document]:
    """Carga en caché los chunks de políticas para construir el índice BM25 en memoria."""
    if "chunks" not in _CACHE:
        from mesa_pedidos.indexar import partir_en_chunks
        _CACHE["chunks"] = partir_en_chunks()
    return _CACHE["chunks"]


# ===========================================================================
# 2. Retrievers: Denso, BM25 e Híbrido
# ===========================================================================

def retriever_denso(k: int = config.K_RECUPERACION) -> Any:
    """Búsqueda semántica densa sobre FAISS con MMR (Maximal Marginal Relevance).
    Trae fetch_k candidatos preliminares y selecciona los k más diversos y relevantes.
    """
    if "faiss" not in _CACHE:
        from mesa_pedidos.indexar import construir_indice
        _CACHE["faiss"] = construir_indice()
    
    return _CACHE["faiss"].as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": config.FETCH_K,
            "lambda_mult": config.LAMBDA_MMR,
        },
    )


def retriever_bm25(k: int = config.K_RECUPERACION) -> Any:
    """Búsqueda léxica exacta por BM25 sobre los 59 chunks estructurados."""
    from langchain_community.retrievers import BM25Retriever

    clave = f"bm25-{k}"
    if clave not in _CACHE:
        _CACHE[clave] = BM25Retriever.from_documents(
            _chunks(),
            preprocess_func=tokenizar,
            k=k,
        )
    return _CACHE[clave]


def retriever_hibrido(k: int = config.K_RECUPERACION) -> Any:
    """Búsqueda híbrida: fusiona los rankings de BM25 y FAISS denso (EnsembleRetriever)."""
    from langchain_classic.retrievers import EnsembleRetriever

    return EnsembleRetriever(
        retrievers=[retriever_bm25(k), retriever_denso(k)],
        weights=list(config.PESOS_HIBRIDO),
    )


def recuperar(pregunta: str, modo: str | None = None, k: int = config.K_RECUPERACION) -> list[Document]:
    """Recupera los `k` fragmentos más relevantes para la consulta del usuario."""
    modo = modo or config.RETRIEVER
    if modo not in MODOS:
        raise ValueError(f"Modo '{modo}' no reconocido; opciones: {MODOS}")

    mapa = {
        "denso": retriever_denso,
        "bm25": retriever_bm25,
        "hibrido": retriever_hibrido,
    }
    retriever = mapa[modo](k)
    return retriever.invoke(pregunta)[:k]


# ===========================================================================
# 3. Formateo y Traducción de Citas por Python
# ===========================================================================

def formatear_contexto(documentos: list[Document]) -> str:
    """Formatea los fragmentos numerados con encabezado [1], [2] para el LLM."""
    if not documentos:
        return "(No se encontraron fragmentos relevantes)"
    
    bloques = []
    for i, doc in enumerate(documentos, start=1):
        m = doc.metadata
        encabezado = f"[{i}] ({m['documento']}, pág. {m['paginas']} - {m.get('seccion', '')})"
        bloques.append(f"{encabezado}\n{doc.page_content}")
    return "\n\n".join(bloques)


def citas_desde_fuentes(documentos: list[Document], fuentes: list[int]) -> list[dict[str, Any]]:
    """Traduce los números devueltos por el LLM a citas reales con metadatos fidedignos.
    Cualquier número fuera de rango es descartado silenciosamente para evitar alucinaciones.
    """
    citas: list[dict[str, Any]] = []
    vistos: set[int] = set()

    for n in fuentes:
        if 1 <= n <= len(documentos) and n not in vistos:
            vistos.add(n)
            m = documentos[n - 1].metadata
            citas.append({
                "documento": m["documento"],
                "pagina": m["pagina"],
                "paginas": m.get("paginas", str(m["pagina"])),
                "seccion": m.get("seccion", ""),
            })
    return citas


def describir(documentos: list[Document]) -> list[dict[str, Any]]:
    """Serializa los fragmentos recuperados para almacenarlos en el estado o logs."""
    return [
        {
            "indice": i,
            "documento": d.metadata["documento"],
            "pagina": d.metadata["pagina"],
            "paginas": d.metadata.get("paginas", str(d.metadata["pagina"])),
            "seccion": d.metadata.get("seccion", ""),
            "texto": d.page_content,
        }
        for i, d in enumerate(documentos, start=1)
    ]


# ===========================================================================
# 4. Pipeline Completo: RAG con Citación Verificable
# ===========================================================================

def responder_consulta(
    pregunta: str,
    modo: str | None = None,
    cliente_info: str | None = None,
    fecha_mensaje: str | None = None,
) -> dict[str, Any]:
    """Ejecuta el pipeline RAG completo:
    1. Recupera los fragmentos más relevantes.
    2. Construye el prompt con contexto numerado.
    3. Invoca a Google Gemini con Structured Output (RespuestaConsulta).
    4. Python traduce los números devueltos a citas verificables.
    5. Retorna la respuesta formateada con citas al pie.
    """
    config.verificar_entorno()
    modo = modo or config.RETRIEVER
    docs = recuperar(pregunta, modo=modo)
    contexto = formatear_contexto(docs)

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model=config.MODELO_LLM,
        google_api_key=config.obtener_clave_api(),
        temperature=config.TEMPERATURA,
    )
    llm_estructurado = llm.with_structured_output(RespuestaConsulta)

    info_adicional = ""
    if cliente_info:
        info_adicional += f"\n- Datos del cliente que consulta: {cliente_info}"
    if fecha_mensaje:
        info_adicional += f"\n- Fecha de emisión del mensaje: {fecha_mensaje}"

    prompt = f"""Sos el asistente oficial de atención de {config.EMPRESA}.
Tu tarea es responder la consulta del cliente basándote EXCLUSIVAMENTE en los fragmentos de políticas provistos a continuación.

Reglas obligatorias:
1. Sé conciso, claro, cordial y profesional.
2. Basate ÚNICAMENTE en la información de los fragmentos del contexto. No inventes reglas, plazos ni excepciones.
3. Si la respuesta NO está en los fragmentos, seteá 'encontrado = False' y respondé cordialmente indicando que dicha información no figura en las políticas disponibles y sugerí derivar la consulta con un asesor comercial.
4. En el campo 'fuentes', indicá los números enteros entre corchetes de los fragmentos que utilizaste (por ejemplo: [1] o [1, 2]).{info_adicional}

---
CONTEXTO DE POLÍTICAS CORPORATIVAS:
{contexto}
---

CONSULTA DEL CLIENTE:
"{pregunta}"
"""

    resultado: RespuestaConsulta = llm_estructurado.invoke(prompt)
    citas = citas_desde_fuentes(docs, resultado.fuentes)

    # Formatear respuesta con pie de fuentes fidedigno
    respuesta_final = resultado.respuesta.strip()
    if citas:
        pie_citas = []
        for c in citas:
            pie_citas.append(f"{c['documento']} (Pág. {c['paginas']})")
        respuesta_final += f"\n\n[Fuente: {', '.join(pie_citas)}]"

    return {
        "pregunta": pregunta,
        "respuesta": respuesta_final,
        "citas": citas,
        "encontrado": resultado.encontrado,
        "documentos": describir(docs),
    }


# ===========================================================================
# Ejecución CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de recuperación RAG sobre las políticas")
    parser.add_argument("pregunta", nargs="*", default=["¿Cuánto sale el flete a Neuquén?"], help="Pregunta a consultar")
    parser.add_argument("--modo", choices=MODOS, default=None, help="Modo de recuperación: denso, bm25 o hibrido")
    parser.add_argument("--solo-buscar", action="store_true", help="Solo muestra los fragmentos recuperados sin llamar al LLM")
    args = parser.parse_args()

    pregunta = " ".join(args.pregunta)
    modo = args.modo or config.RETRIEVER

    print(f"\n{'='*75}")
    print(f"CONSULTA: {pregunta}")
    print(f"MODO RETRIEVER: {modo.upper()} (Pesos: {config.PESOS_HIBRIDO})")
    print(f"{'='*75}\n")

    if args.solo_buscar:
        docs = recuperar(pregunta, modo=modo)
        print(formatear_contexto(docs))
        return

    print("Generando respuesta RAG con Google Gemini...")
    res = responder_consulta(pregunta, modo=modo)
    print("\nRESPUESTA GENERADA:")
    print("-" * 60)
    print(res["respuesta"])
    print("-" * 60)
    if res["citas"]:
        print("\nCITAS AUDITADAS (construidas por Python):")
        for c in res["citas"]:
            print(f"  • {c['documento']} | Pág. {c['paginas']} | Sección: {c['seccion']}")
    else:
        print("\n(No se identificaron citas aplicables)")
    print()


if __name__ == "__main__":
    main()
