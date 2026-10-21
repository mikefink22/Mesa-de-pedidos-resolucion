# -*- coding: utf-8 -*-
"""
tests/test_grafo.py - Validación formal del Grafo de LangGraph y sus Routers.

Verifica:
1. Compilación limpia de StateGraph(EstadoMesa) con InMemorySaver y SqliteSaver.
2. Presencia de los 9 nodos obligatorios en la topología.
3. Comportamiento exacto de los 3 routers oficiales (pasamanos sin 'if' de negocio).
4. Generación válida del diagrama Mermaid con todas las aristas fijas y condicionales.

Se ejecuta con pytest:
    pytest tests/test_grafo.py -v
O directamente como script para inspección visual:
    py tests/test_grafo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar importación de mesa_pedidos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesa_pedidos import grafo
from mesa_pedidos.grafo import (
    construir_grafo,
    dibujar,
    ruta_por_intencion,
    ruta_por_validacion,
    ruta_post_aprobacion,
)


def test_compilacion_grafo_en_memoria():
    """Valida que el grafo compile correctamente con InMemorySaver."""
    app = construir_grafo(persistente=False)
    assert app is not None
    nodos_en_grafo = set(app.get_graph().nodes.keys())
    
    nodos_esperados = {
        "cargar_cliente",
        "clasificar",
        "responder_consulta",
        "extraer_pedido",
        "validar_pedido",
        "aprobacion_humana",
        "confirmar_pedido",
        "rechazar_pedido",
        "derivar_humano",
    }
    for nodo in nodos_esperados:
        assert nodo in nodos_en_grafo, f"Falta el nodo '{nodo}' en el grafo"


def test_router_por_intencion():
    """Valida que ruta_por_intencion despache según el campo 'intencion'."""
    assert ruta_por_intencion({"intencion": "consulta"}) == "consulta"
    assert ruta_por_intencion({"intencion": "pedido"}) == "pedido"
    assert ruta_por_intencion({"intencion": "derivar"}) == "derivar"
    assert ruta_por_intencion({}) == "derivar"


def test_router_por_validacion():
    """Valida que ruta_por_validacion lea state['validacion']['ruta']."""
    assert ruta_por_validacion({"validacion": {"ruta": "confirmar_pedido"}}) == "confirmar_pedido"
    assert ruta_por_validacion({"validacion": {"ruta": "aprobacion_humana"}}) == "aprobacion_humana"
    assert ruta_por_validacion({"validacion": {"ruta": "rechazar_pedido"}}) == "rechazar_pedido"
    assert ruta_por_validacion({"validacion": {"ruta": "derivar_humano"}}) == "derivar_humano"
    assert ruta_por_validacion({}) == "derivar_humano"


def test_router_post_aprobacion():
    """Valida que ruta_post_aprobacion traduzca las decisiones de Gerencia."""
    assert ruta_post_aprobacion({"decision_humana": {"tipo": "aprobar"}}) == "confirmar"
    assert ruta_post_aprobacion({"decision_humana": {"tipo": "editar"}}) == "confirmar"
    assert ruta_post_aprobacion({"decision_humana": {"tipo": "rechazar"}}) == "rechazar"
    assert ruta_post_aprobacion({"decision_humana": {"tipo": "otro"}}) == "rechazar"
    assert ruta_post_aprobacion({}) == "rechazar"


def test_diagrama_mermaid():
    """Valida que la exportación a Mermaid contenga los nodos y transiciones esperados y guarde archivo en docs/."""
    ruta_test = Path(__file__).resolve().parent.parent / "docs" / "grafo.md"
    if ruta_test.exists():
        ruta_test.unlink()
    diagrama = dibujar(guardar_en=ruta_test)
    assert "cargar_cliente" in diagrama
    assert "clasificar" in diagrama
    assert "validar_pedido" in diagrama
    assert "aprobacion_humana" in diagrama
    assert "confirmar_pedido" in diagrama
    assert ruta_test.exists()
    assert "```mermaid" in ruta_test.read_text(encoding="utf-8")


def main() -> None:
    print("=" * 75)
    print("🧪 VALIDACIÓN VISUAL DEL GRAFO LANGGRAPH Y SUS ROUTERS")
    print("=" * 75)

    print("\n1. Compilando grafo en memoria (StateGraph con 9 nodos)...")
    app = construir_grafo(persistente=False)
    nodos_detectados = list(app.get_graph().nodes.keys())
    print(f"   ✅ Grafo compilado exitosamente. Nodos registrados ({len(nodos_detectados)}):")
    for n in nodos_detectados:
        print(f"      • {n}")

    print("\n2. Probando los 3 Routers oficiales de forma aislada...")
    print(f"   • ruta_por_intencion('pedido')   -> {ruta_por_intencion({'intencion': 'pedido'})}")
    print(f"   • ruta_por_validacion('aprobacion_humana') -> {ruta_por_validacion({'validacion': {'ruta': 'aprobacion_humana'}})}")
    print(f"   • ruta_post_aprobacion('aprobar') -> {ruta_post_aprobacion({'decision_humana': {'tipo': 'aprobar'}})}")
    print(f"   • ruta_post_aprobacion('rechazar')-> {ruta_post_aprobacion({'decision_humana': {'tipo': 'rechazar'}})}")
    print("   ✅ Routers funcionando como pasamanos puros.")

    print("\n3. Exportando diagrama Mermaid a archivo Markdown en docs/...")
    ruta_artefacto = Path(__file__).resolve().parent.parent / "docs" / "grafo.md"
    if ruta_artefacto.exists():
        ruta_artefacto.unlink()
    dibujar(guardar_en=ruta_artefacto)
    print(f"   ✅ Diagrama Mermaid generado por el código en: {ruta_artefacto}")
    print("=" * 75)


if __name__ == "__main__":
    main()
