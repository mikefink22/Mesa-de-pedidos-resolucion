# -*- coding: utf-8 -*-
"""
tests/test_ejecucion.py - Pruebas unitarias para el motor de ejecución y Human-in-the-Loop.

Valida:
1. Generación de configuraciones y nombres de thread (`thread_de`, `config_de`).
2. Interrupción y reanudación con `Command(resume=...)` usando un StateGraph con `interrupt()`.
3. Durabilidad transaccional en `SqliteSaver`: simula el cierre de sesión ("cerrar la terminal"),
   reapertura con un nuevo proceso/conexión SQLite y reanudación sin pérdida de estado.
4. Prevención de deadlocks en `pendientes()` al listar hilos retenidos.
5. Limpieza de hilos mediante `borrar_thread()`.

Cumple con la regla de aislamiento del Golden Set: utiliza casos y grafos sintéticos
para evitar sesgos y no consumir cuotas innecesarias de LLM en tests de plomería.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any
from operator import add

# Asegurar importación de mesa_pedidos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from mesa_pedidos import ejecucion
from mesa_pedidos.esquemas import MensajeEntrante


# ===========================================================================
# 1. Esquema y Grafo Sintético para pruebas de Plomería e Interrupt
# ===========================================================================

class EstadoSintetico(TypedDict, total=False):
    mensaje_id: str
    canal: str
    texto: str
    cuit: str
    remitente: str
    fecha: str
    etapa: str
    decision: dict[str, Any]
    log: Annotated[list[str], add]
    errores: Annotated[list[str], add]


def nodo_preparar(state: EstadoSintetico) -> dict[str, Any]:
    return {
        "etapa": "esperando_aprobacion",
        "log": ["[nodo_preparar] Pedido validado con alertas, requiere autorización"],
    }


def nodo_pausa(state: EstadoSintetico) -> dict[str, Any]:
    # Simula el interrupt() de LangGraph con un payload representativo
    payload = {
        "mensaje_id": state.get("mensaje_id", ""),
        "razon_social": "Cliente Sintético S.A.",
        "neto": 1_500_000.0,
        "alertas": ["Exceso de crédito comercial"],
    }
    decision = interrupt(payload)
    if isinstance(decision, str):
        decision = {"tipo": decision}
    return {
        "etapa": "aprobado" if decision.get("tipo") == "aprobar" else "rechazado",
        "decision": decision,
        "log": [f"[nodo_pausa] Decisión humana aplicada: {decision.get('tipo', '').upper()}"],
    }


def nodo_finalizar(state: EstadoSintetico) -> dict[str, Any]:
    return {
        "log": [f"[nodo_finalizar] Proceso concluido en etapa {state.get('etapa')}"],
    }


def crear_grafo_sintetico(ruta_db: Path):
    conexion = sqlite3.connect(str(ruta_db), check_same_thread=False)
    checkpointer = SqliteSaver(conexion)
    builder = StateGraph(EstadoSintetico)
    builder.add_node("preparar", nodo_preparar)
    builder.add_node("pausa", nodo_pausa)
    builder.add_node("finalizar", nodo_finalizar)

    builder.add_edge(START, "preparar")
    builder.add_edge("preparar", "pausa")
    builder.add_edge("pausa", "finalizar")
    builder.add_edge("finalizar", END)

    return builder.compile(checkpointer=checkpointer), conexion


# ===========================================================================
# 2. Tests de Plomería y Configuración
# ===========================================================================

def test_thread_de_y_config_de():
    tid = ejecucion.thread_de("MSG-013")
    assert tid == "mensaje-MSG-013"

    cfg = ejecucion.config_de(tid, retriever="denso")
    assert cfg["configurable"]["thread_id"] == "mensaje-MSG-013"
    assert cfg["configurable"]["retriever"] == "denso"


# ===========================================================================
# 3. Tests de Human-in-the-Loop y Durabilidad SQLite
# ===========================================================================

def test_interrupt_frena_y_emite_payload(tmp_path: Path):
    db_file = tmp_path / "test_checkpoints.sqlite"
    grafo, conexion = crear_grafo_sintetico(db_file)

    mensaje = MensajeEntrante(
        id="MSG-SINT-01",
        canal="whatsapp",
        fecha="2026-10-21",
        cuit="30-71455982-2",
        remitente="Test User",
        texto="Pedido de prueba con interrupción",
    )
    cfg = ejecucion.config_de("thread-sint-01")

    # Iniciar la corrida
    resultado = ejecucion.iniciar(grafo, mensaje, cfg)

    # Debe frenar en nodo_pausa
    assert resultado["pendiente"] is not None
    payload = resultado["pendiente"]
    assert payload["mensaje_id"] == "MSG-SINT-01"
    assert payload["neto"] == 1_500_000.0
    assert "Exceso de crédito comercial" in payload["alertas"]

    # El estado contiene el log acumulativo hasta la pausa
    assert any("esperando_aprobacion" in l or "nodo_preparar" in l for l in resultado["estado"]["log"])

    conexion.close()


def test_cerrar_terminal_reabrir_y_retomar(tmp_path: Path):
    """Simula:
    1. Proceso A arranca y se frena en interrupt().
    2. Proceso A cierra la conexión SQLite ('cierra la terminal').
    3. Proceso B abre la misma base de datos, lista pendientes y retoma con 'aprobar'.
    4. El grafo termina limpiamente sin repetir pasos previos.
    """
    db_file = tmp_path / "durabilidad.sqlite"

    # --- PROCESO A ---
    grafo_a, conexion_a = crear_grafo_sintetico(db_file)
    mensaje = MensajeEntrante(
        id="MSG-SINT-02",
        canal="email",
        fecha="2026-10-21",
        cuit="30-70988341-7",
        remitente="Gerente Compras",
        texto="Pedido urgente",
    )
    cfg = ejecucion.config_de("thread-sint-02")
    res_a = ejecucion.iniciar(grafo_a, mensaje, cfg)
    assert res_a["pendiente"] is not None

    # Simular cierre de terminal
    conexion_a.close()

    # --- PROCESO B (Nuevo proceso, misma base SQLite) ---
    grafo_b, conexion_b = crear_grafo_sintetico(db_file)

    # 1. Comprobar que pendientes() lo detecta
    lista_pendientes = ejecucion.pendientes(grafo_b)
    threads_pendientes = [t for t, _ in lista_pendientes]
    assert "thread-sint-02" in threads_pendientes

    # 2. Retomar con decisión de aprobación
    decision = {"tipo": "aprobar", "razon": "Aprobado por el directorio"}
    res_b = ejecucion.retomar(grafo_b, decision, cfg)

    # 3. Debe concluir sin pendientes
    assert res_b["pendiente"] is None
    assert res_b["estado"]["etapa"] == "aprobado"
    assert res_b["estado"]["decision"]["tipo"] == "aprobar"

    # Verificar que el reducer de log acumuló la historia completa
    log_final = res_b["estado"]["log"]
    assert any("nodo_preparar" in l for l in log_final)
    assert any("nodo_pausa" in l for l in log_final)
    assert any("nodo_finalizar" in l for l in log_final)

    # 4. Ya no debe figurar en pendientes()
    assert [t for t, _ in ejecucion.pendientes(grafo_b)] == []

    conexion_b.close()


def test_retomar_con_rechazo(tmp_path: Path):
    db_file = tmp_path / "rechazo.sqlite"
    grafo, conexion = crear_grafo_sintetico(db_file)

    mensaje = MensajeEntrante(
        id="MSG-SINT-03",
        canal="web",
        fecha="2026-10-21",
        cuit="30-70988341-7",
        remitente="Compras",
        texto="Pedido a rechazar",
    )
    cfg = ejecucion.config_de("thread-sint-03")
    ejecucion.iniciar(grafo, mensaje, cfg)

    res_retomar = ejecucion.retomar(grafo, {"tipo": "rechazar", "razon": "Mora excesiva"}, cfg)
    assert res_retomar["pendiente"] is None
    assert res_retomar["estado"]["etapa"] == "rechazado"

    conexion.close()


def test_borrar_thread(tmp_path: Path):
    db_file = tmp_path / "borrar.sqlite"
    grafo, conexion = crear_grafo_sintetico(db_file)

    mensaje = MensajeEntrante(
        id="MSG-SINT-04",
        canal="whatsapp",
        fecha="2026-10-21",
        cuit="30-70988341-7",
        remitente="Cliente",
        texto="Pedido para borrar",
    )
    cfg = ejecucion.config_de("thread-sint-04")
    ejecucion.iniciar(grafo, mensaje, cfg)

    # Verificar que está pendiente
    assert len(ejecucion.pendientes(grafo)) == 1

    # Borrar thread
    ejecucion.borrar_thread(grafo, "thread-sint-04")

    # Ahora debe estar vacío
    assert len(ejecucion.pendientes(grafo)) == 0

    conexion.close()
