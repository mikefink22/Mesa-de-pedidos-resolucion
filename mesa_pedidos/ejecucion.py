# -*- coding: utf-8 -*-
"""
ejecucion.py - Motor de ejecución modular para LangGraph con Human-in-the-Loop.

Este módulo desacopla la ejecución del grafo de cualquier interfaz específica
(consola CLI correr.py, tests unitarios, o futuras interfaces web/Streamlit).
Toda la lógica de iniciar corridas, detectar frenados en interrupt(), persistir
estados y reanudar con decisiones humanas vive aquí.

Conceptos Clave de Arquitectura:
- thread_id: Identificador unívoco del hilo de conversación (f"mensaje-{mensaje_id}").
  Permite al checkpointer SqliteSaver guardar y aislar la historia de cada mensaje.
  Dos ejecuciones con el mismo thread_id comparten el mismo estado; por eso se puede
  cerrar la consola con Ctrl+C y reanudar horas o días después sin reprocesar.
- stream_mode="updates": En lugar de esperar al final, el grafo emite lo que devuelve
  cada nodo apenas termina su ejecución, permitiendo registrar la traza en vivo (callback al_pasar).
- get_state(cfg).next: Lista de nodos pendientes de ejecución. Si tras correr no está
  vacía, el grafo se pausó en un interrupt() esperando intervención humana.
- Command(resume=decision): Mecanismo oficial de LangGraph para retomar un grafo en pausa
  inyectando la decisión del operador directamente como valor de retorno de interrupt().
- Prevención de deadlocks en SqliteSaver: Al listar hilos pendientes, se recopilan primero
  los thread_ids en memoria y luego se consulta cada snapshot individualmente.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from langgraph.types import Command

from mesa_pedidos.esquemas import MensajeEntrante
from mesa_pedidos.estado import estado_inicial

AlPasar = Optional[Callable[[str, dict], None]]


def thread_de(mensaje_id: str) -> str:
    """Devuelve el thread_id estándar para un identificador de mensaje."""
    return f"mensaje-{mensaje_id}"


def config_de(thread_id: str, retriever: Optional[str] = None) -> dict:
    """Construye el diccionario de configuración que requiere LangGraph."""
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if retriever:
        configurable["retriever"] = retriever
    return {"configurable": configurable}


def _pendiente(snapshot: Any) -> Optional[dict]:
    """Extrae el payload del interrupt si el thread quedó pausado; de lo contrario, None."""
    if not snapshot or not getattr(snapshot, "next", None):
        return None
    for tarea in getattr(snapshot, "tasks", []):
        for intr in getattr(tarea, "interrupts", []):
            return intr.value
    return None


def _correr(
    grafo: Any,
    entrada: Any,
    cfg: dict,
    al_pasar: AlPasar = None,
    origen: str = "python",
    tipo: str = "inicio",
) -> dict[str, Any]:
    """Ejecuta el grafo hasta terminar o hasta encontrar un interrupt().
    
    Devuelve un diccionario con:
    - pasos: lista de tuplas (nodo, cambios) ejecutados en esta corrida.
    - estado: diccionario con los valores del estado actual (snapshot.values).
    - pendiente: payload de interrupción si se frenó esperando a Gerencia; None si concluyó.
    """
    pasos: list[tuple[str, dict]] = []
    
    for evento in grafo.stream(entrada, cfg, stream_mode="updates"):
        for nodo, cambios in evento.items():
            if nodo == "__interrupt__":
                continue
            pasos.append((nodo, cambios or {}))
            if al_pasar:
                al_pasar(nodo, cambios or {})

    snapshot = grafo.get_state(cfg)
    pendiente = _pendiente(snapshot)
    return {
        "pasos": pasos,
        "estado": dict(snapshot.values) if snapshot else {},
        "pendiente": pendiente,
    }


def iniciar(
    grafo: Any,
    mensaje: MensajeEntrante | dict,
    cfg: dict,
    al_pasar: AlPasar = None,
    origen: str = "python",
) -> dict[str, Any]:
    """Arranca el procesamiento de un mensaje desde cero en el thread indicado en cfg."""
    if isinstance(mensaje, dict):
        mensaje = MensajeEntrante(**mensaje)
    return _correr(grafo, estado_inicial(mensaje), cfg, al_pasar, origen, "inicio")


def retomar(
    grafo: Any,
    decision: dict | str,
    cfg: dict,
    al_pasar: AlPasar = None,
    origen: str = "python",
) -> dict[str, Any]:
    """Reanuda un thread pausado inyectando la decisión del operador mediante Command(resume=...)."""
    return _correr(grafo, Command(resume=decision), cfg, al_pasar, origen, "retomar")


def estado_de(grafo: Any, cfg: dict) -> dict[str, Any]:
    """Consulta el estado actual de un thread y determina si está esperando aprobación."""
    snapshot = grafo.get_state(cfg)
    return {
        "estado": dict(snapshot.values) if snapshot else {},
        "pendiente": _pendiente(snapshot),
    }


def borrar_thread(grafo: Any, thread_id: str) -> None:
    """Elimina el historial y checkpoints de un thread para reiniciar su procesamiento."""
    cp = getattr(grafo, "checkpointer", None)
    if cp is None:
        return
    if hasattr(cp, "delete_thread"):
        cp.delete_thread(thread_id)
    elif hasattr(cp, "conn"):
        try:
            with cp.conn as conn:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        except Exception:
            pass


def pendientes(grafo: Any) -> list[tuple[str, dict]]:
    """Obtiene la lista de todos los threads frenados esperando aprobación humana.
    
    Técnica anti-deadlock:
    SqliteSaver mantiene tomada la conexión durante el generador checkpointer.list().
    Si llamáramos a get_state() dentro del bucle se produciría un bloqueo mutuo (deadlock).
    Por eso primero recolectamos todos los thread_ids y luego consultamos sus snapshots.
    """
    cp = getattr(grafo, "checkpointer", None)
    if cp is None:
        return []

    threads: set[str] = set()
    try:
        for check in cp.list(None):
            cfg_item = getattr(check, "config", {}) or {}
            tid = cfg_item.get("configurable", {}).get("thread_id")
            if tid:
                threads.add(tid)
    except Exception:
        pass

    salida: list[tuple[str, dict]] = []
    for thread_id in threads:
        try:
            snapshot = grafo.get_state(config_de(thread_id))
            payload = _pendiente(snapshot)
            if payload:
                salida.append((thread_id, payload))
        except Exception:
            continue

    return sorted(salida, key=lambda x: x[0])


# Alias semántico
listar_pendientes = pendientes
