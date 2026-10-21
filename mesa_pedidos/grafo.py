# -*- coding: utf-8 -*-
"""
grafo.py - Ensamblado de la Máquina de Estados con LangGraph.

Este módulo construye el flujo orquestado del sistema:
- Define los 3 Routers oficiales (funciones pasamanos que leen el estado, sin lógica 'if' de negocio).
- Ensambla los 9 nodos de `nodos.py` sobre StateGraph(EstadoMesa).
- Configura las aristas fijas y las 3 aristas condicionales.
- Integra el checkpointer transaccional `SqliteSaver` sobre `artefactos/checkpoints.sqlite`
  para dar soporte a la interrupción e interactividad Human-in-the-Loop.
- Aplica `RetryPolicy` en los nodos que consumen cuota de LLM ante fallas transitorias de red.
- Permite exportar y visualizar la topología del grafo en sintaxis Mermaid.

Topología del Grafo (9 Nodos · 3 Routers · 5 Hojas Terminales):

                        START
                          │
                   [cargar_cliente]
                          │
                     [clasificar]
                          │  (Router 1: ruta_por_intencion)
               ┌──────────┼──────────┐
         (consulta)   (pedido)   (derivar)
               │          │          │
    [responder_consulta]  │   [derivar_humano] <───────────────┐
               │    [extraer_pedido] │                         │
              END         │         END                        │
                   [validar_pedido]                            │
                          │  (Router 2: ruta_por_validacion)   │
         ┌────────┬───────┴────────┬──────────┐                │
   (confirmar) (humano)     (rechazar)    (derivar) ───────────┘
         │        │                │
         │ [aprobacion_humana]     │
         │        │  (Router 3: ruta_post_aprobacion)
         │   ┌────┴────┐           │
         │ confirmar  rechazar     │
         │   │         │           │
   [confirmar_pedido]  [rechazar_pedido]
         │                     │
        END                   END
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import config
from mesa_pedidos import nodos
from mesa_pedidos.estado import EstadoMesa


# ===========================================================================
# 1. Los 3 Routers Oficiales (Pasamanos puros: leen el estado, no deciden)
# ===========================================================================

def ruta_por_intencion(state: EstadoMesa) -> str:
    """Router 1: Posterior al nodo `clasificar`.
    
    Lee el campo `intencion` del estado (el cual ya fue validado y
    resuelto con los guardrails de `reglas.intencion_efectiva`).
    Retorna: 'consulta' | 'pedido' | 'derivar'
    """
    return state.get("intencion") or "derivar"


def ruta_por_validacion(state: EstadoMesa) -> str:
    """Router 2: Posterior al nodo `validar_pedido`.
    
    Lee la ruta comercial sugerida por el motor determinístico en `state['validacion']['ruta']`.
    Retorna: 'confirmar_pedido' | 'aprobacion_humana' | 'rechazar_pedido' | 'derivar_humano'
    """
    validacion = state.get("validacion") or {}
    return validacion.get("ruta") or "derivar_humano"


def ruta_post_aprobacion(state: EstadoMesa) -> str:
    """Router 3: Posterior a la reanudación del nodo `aprobacion_humana`.
    
    Lee la acción tomada por el operador humano en `state['decision_humana']['tipo']`.
    - Si fue 'aprobar' o 'editar' -> bifurca a 'confirmar'.
    - Si fue 'rechazar' (o cualquier valor imprevisto) -> bifurca a 'rechazar'.
    """
    decision = state.get("decision_humana") or {}
    tipo = decision.get("tipo")
    if tipo in ("aprobar", "editar"):
        return "confirmar"
    return "rechazar"


# ===========================================================================
# 2. Persistencia y Checkpointer Transaccional
# ===========================================================================

def checkpointer_sqlite(ruta_db=config.CHECKPOINTS_DB) -> Any:
    """Crea una instancia de SqliteSaver sobre archivo en disco.
    
    Permite que el estado de la conversación y las pausas de Human-in-the-Loop
    sobrevivan a reinicios de máquina o cierres de consola.
    `check_same_thread=False` evita bloqueos en ejecuciones asíncronas de LangGraph.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    conexion = sqlite3.connect(str(ruta_db), check_same_thread=False)
    return SqliteSaver(conexion)


# ===========================================================================
# 3. Construcción y Compilación del StateGraph
# ===========================================================================

def construir_grafo(
    checkpointer: Optional[Any] = None,
    persistente: bool = True,
    max_reintentos_llm: int = 3,
) -> Any:
    """Ensambla, cablea y compila el StateGraph del proyecto.
    
    Parámetros:
    - checkpointer: Instancia personalizada de checkpointer (o None para auto-configurar).
    - persistente: Si es True, utiliza SqliteSaver en archivo (producción y CLI).
                   Si es False, utiliza InMemorySaver (ideal para tests rápidos y medición).
    - max_reintentos_llm: Número de reintentos automáticos para llamadas al LLM ante fallas transitorias.
    """
    politica_reintentos = RetryPolicy(max_attempts=max_reintentos_llm)

    # 1. Instanciación del StateGraph con el TypedDict EstadoMesa
    grafo = StateGraph(EstadoMesa)

    # 2. Registro de los 9 Nodos
    grafo.add_node("cargar_cliente", nodos.cargar_cliente)
    grafo.add_node("clasificar", nodos.clasificar, retry_policy=politica_reintentos)
    grafo.add_node("responder_consulta", nodos.responder_consulta, retry_policy=politica_reintentos)
    grafo.add_node("extraer_pedido", nodos.extraer_pedido, retry_policy=politica_reintentos)
    grafo.add_node("validar_pedido", nodos.validar_pedido)
    grafo.add_node("aprobacion_humana", nodos.aprobacion_humana)
    grafo.add_node("confirmar_pedido", nodos.confirmar_pedido, retry_policy=politica_reintentos)
    grafo.add_node("rechazar_pedido", nodos.rechazar_pedido, retry_policy=politica_reintentos)
    grafo.add_node("derivar_humano", nodos.derivar_humano)

    # 3. Cableado de Aristas Fijas
    grafo.add_edge(START, "cargar_cliente")
    grafo.add_edge("cargar_cliente", "clasificar")
    grafo.add_edge("extraer_pedido", "validar_pedido")

    # 4. Cableado de las 3 Aristas Condicionales (Routers + Mapas de Destino)
    
    # Router 1: Por intención tras clasificar
    grafo.add_conditional_edges(
        "clasificar",
        ruta_por_intencion,
        {
            "consulta": "responder_consulta",
            "pedido": "extraer_pedido",
            "derivar": "derivar_humano",
        },
    )

    # Router 2: Por validación comercial tras validar_pedido
    grafo.add_conditional_edges(
        "validar_pedido",
        ruta_por_validacion,
        {
            "confirmar_pedido": "confirmar_pedido",
            "aprobacion_humana": "aprobacion_humana",
            "rechazar_pedido": "rechazar_pedido",
            "derivar_humano": "derivar_humano",
        },
    )

    # Router 3: Posterior a la aprobación/rechazo humano
    grafo.add_conditional_edges(
        "aprobacion_humana",
        ruta_post_aprobacion,
        {
            "confirmar": "confirmar_pedido",
            "rechazar": "rechazar_pedido",
        },
    )

    # 5. Aristas hacia END (Las cuatro hojas terminales de negocio)
    grafo.add_edge("responder_consulta", END)
    grafo.add_edge("confirmar_pedido", END)
    grafo.add_edge("rechazar_pedido", END)
    grafo.add_edge("derivar_humano", END)

    # 6. Selección y activación del checkpointer
    if checkpointer is None:
        if persistente:
            checkpointer = checkpointer_sqlite()
        else:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()

    # 7. Compilación final de la máquina de estados
    return grafo.compile(checkpointer=checkpointer)


# ===========================================================================
# 4. Exportación Visual a Mermaid y HTML
# ===========================================================================

def _limpiar_mermaid(raw: str) -> str:
    """Normaliza la salida de draw_mermaid() de LangGraph para máxima compatibilidad:
    1. Remueve el frontmatter YAML ('--- config ... ---') que rompe visores Markdown estándar.
    2. Quita el punto y coma del encabezado ('graph TD;' -> 'graph TD') requerido por VS Code.
    3. Elimina tags HTML como '<p>' y '</p>' en los nodos de inicio/fin.
    4. Convierte las etiquetas de aristas condicionales a la sintaxis estándar: -.->|etiqueta|.
    """
    import re
    # Eliminar frontmatter YAML si está presente (soportando \n o \r\n de Windows)
    texto = re.sub(r"^---\s*\r?\n.*?\r?\n---\s*\r?\n", "", raw, flags=re.DOTALL).strip()
    # Eliminar el punto y coma en el encabezado ('graph TD;' -> 'graph TD')
    texto = re.sub(r"^(graph|flowchart)\s+([A-Z]{2});", r"\1 \2", texto)
    # Limpiar tags HTML en identificadores
    texto = texto.replace("<p>", "").replace("</p>", "")
    # Convertir flechas punteadas con texto a la sintaxis estándar de Mermaid (-.->|etiqueta|)
    texto = re.sub(r"-\.\s*(?:&nbsp;|\s)*([a-zA-Z0-9_]+)(?:&nbsp;|\s)*\.->", r"-.->|\1|", texto)
    return texto.strip()


def dibujar(guardar_en: Optional[Path | str] = None) -> str:
    """Genera la representación gráfica del StateGraph en formato Mermaid.
    
    Si se especifica `guardar_en` (ej: 'docs/grafo.md'), escribe el archivo Markdown
    con el bloque ```mermaid limpio (compatible con GitHub y visores Markdown).
    """
    app = construir_grafo(persistente=False)
    diagrama_raw = app.get_graph().draw_mermaid()
    diagrama = _limpiar_mermaid(diagrama_raw)
    
    if guardar_en:
        destino = Path(guardar_en)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido_md = (
            "# 🗺️ Topología del Grafo de LangGraph\n\n"
            "Este diagrama representa la máquina de estados de la Mesa de Pedidos "
            "(9 nodos, 3 routers y 5 hojas terminales).\n\n"
            f"```mermaid\n{diagrama}\n```\n"
        )
        destino.write_text(contenido_md, encoding="utf-8")
        
    return diagrama


if __name__ == "__main__":
    ruta_salida = config.RAIZ / "docs" / "grafo.md"
    dibujar(guardar_en=ruta_salida)
    print(f"Diagrama Mermaid guardado en: {ruta_salida}")


if __name__ == "__main__":
    ruta_salida = config.ARTEFACTOS / "grafo.md"
    dibujar(guardar_en=ruta_salida)
    print(f"Diagrama Mermaid guardado en: {ruta_salida}")
