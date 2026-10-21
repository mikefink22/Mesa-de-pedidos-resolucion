# -*- coding: utf-8 -*-
"""
tests/test_estado.py - Verificación con LangGraph REAL del Estado y Reducers.

Prueba el EstadoMesa compilándolo en un StateGraph real de LangGraph para validar:
1. Que LangGraph acepte el TypedDict EstadoMesa y compile sin errores.
2. Que el reducer Annotated[list[str], add] acumule las líneas del log en cada nodo.
3. Que los campos sin reducer sobrescriban su valor con la última actualización.
4. Que el estado retornado sea serializable para SQLite Checkpoints.

Se puede ejecutar con pytest:
    pytest tests/test_estado.py
O directamente como script con salida visual didáctica:
    py tests/test_estado.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Garantiza que Python encuentre el paquete mesa_pedidos sin importar desde dónde se ejecute
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END

from mesa_pedidos.esquemas import MensajeEntrante
from mesa_pedidos.estado import EstadoMesa, estado_inicial


# --- Nodos de Prueba para el Grafo ---

def nodo_cargar_cliente(estado: EstadoMesa) -> dict:
    """Lee el CUIT del estado y simula la carga del cliente, sumando al log."""
    cuit = estado.get("cuit", "")
    return {
        "cliente": {
            "cuit": cuit,
            "razon_social": "Construcciones del Valle S.A.",
            "categoria": "A",
            "limite_credito": 5_000_000.0,
            "saldo_actual": 1_200_000.0,
        },
        "log": [f"[cargar_cliente] Cliente identificado con CUIT {cuit}"],
    }


def nodo_clasificar(estado: EstadoMesa) -> dict:
    """Clasifica la intención y suma su propia línea al log."""
    return {
        "intencion": "pedido",
        "clasificacion": {
            "intencion": "pedido",
            "confianza": 0.98,
            "justificacion": "Pide mercadería específica con cantidades",
        },
        "log": ["[clasificar] Intención clasificada como 'pedido' (confianza 0.98)"],
    }


def nodo_validar(estado: EstadoMesa) -> dict:
    """Define la ruta y suma la tercera línea al log."""
    return {
        "ruta": "aprobacion_humana",
        "log": ["[validar_pedido] Requiere aprobación humana por límite de crédito"],
    }


def construir_grafo_prueba():
    """Construye y compila el flujo mínimo de prueba en LangGraph."""
    builder = StateGraph(EstadoMesa)
    builder.add_node("cargar_cliente", nodo_cargar_cliente)
    builder.add_node("clasificar", nodo_clasificar)
    builder.add_node("validar", nodo_validar)

    builder.add_edge(START, "cargar_cliente")
    builder.add_edge("cargar_cliente", "clasificar")
    builder.add_edge("clasificar", "validar")
    builder.add_edge("validar", END)

    return builder.compile()


# --- Test para Pytest ---

def test_estado_con_langgraph():
    """Valida formalmente el funcionamiento del estado y reducer con aserciones."""
    msg = MensajeEntrante(
        id="MSG-013",
        canal="whatsapp",
        fecha="2026-03-02 09:15",
        cuit="30-71455982-2",
        remitente="Construcciones del Valle S.A.",
        texto="Hola, necesito 15 discos de corte AB-1020.",
    )
    estado_inicio = estado_inicial(msg)
    grafo = construir_grafo_prueba()
    estado_final = grafo.invoke(estado_inicio)

    # 1. Verificación del reducer: El log debe tener 4 entradas acumuladas
    # (1 inicial + 1 de cada uno de los 3 nodos)
    assert len(estado_final["log"]) == 4, f"Se esperaban 4 entradas en log, hay {len(estado_final['log'])}"
    assert "[cargar_cliente]" in estado_final["log"][1]
    assert "[clasificar]" in estado_final["log"][2]
    assert "[validar_pedido]" in estado_final["log"][3]

    # 2. Verificación de sobrescritura normal
    assert estado_final["intencion"] == "pedido"
    assert estado_final["ruta"] == "aprobacion_humana"

    # 3. Verificación de serialización
    json_str = json.dumps(estado_final, ensure_ascii=False)
    assert len(json_str) > 0


# --- Ejecución interactiva visual ---

def main() -> None:
    print("=" * 75)
    print("🧪 PRUEBA VISUAL DE ESTADO Y REDUCERS CON LANGGRAPH REAL")
    print("=" * 75)

    msg = MensajeEntrante(
        id="MSG-013",
        canal="whatsapp",
        fecha="2026-03-02 09:15",
        cuit="30-71455982-2",
        remitente="Construcciones del Valle S.A.",
        texto="Hola, necesito 15 discos de corte AB-1020 y 2 amoladoras angulares 115mm.",
    )

    estado_inicio = estado_inicial(msg)
    print(f"\n1. Estado de arranque preparado para: {estado_inicio['mensaje_id']}")
    print(f"   Log inicial: {estado_inicio['log']}")

    print("\n2. Compilando StateGraph(EstadoMesa) en LangGraph...")
    grafo = construir_grafo_prueba()
    print("   ✅ Compilación limpia: LangGraph validó el TypedDict y los reducers.")

    print("\n3. Ejecutando grafo.invoke(estado_inicio)...")
    estado_final = grafo.invoke(estado_inicio)

    print("\n" + "=" * 75)
    print("📊 RESULTADO RETORNADO POR EL MOTOR DE LANGGRAPH:")
    print("=" * 75)
    print(f"• Cliente:             {estado_final.get('cliente', {}).get('razon_social')}")
    print(f"• Intención detectada: {estado_final.get('intencion')}")
    print(f"• Ruta final:          {estado_final.get('ruta')}")
    print(f"\n• Auditoría de Log (Total entradas acumuladas: {len(estado_final.get('log', []))}):")
    for i, entrada in enumerate(estado_final.get("log", []), start=1):
        print(f"   [{i}] {entrada}")

    print("\n4. Verificando serialización JSON para SQLite Checkpoints...")
    serializado = json.dumps(estado_final, indent=2, ensure_ascii=False)
    print(f"   ✅ Serialización OK ({len(serializado)} bytes).")
    print("=" * 75)


if __name__ == "__main__":
    main()
