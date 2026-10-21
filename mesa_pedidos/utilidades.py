# -*- coding: utf-8 -*-
"""
utilidades.py - Funciones auxiliares comunes para la Mesa de Pedidos.

Provee:
- pesos(x): Formato monetario argentino ($1.234.567,89).
- porcentaje(x): Formato porcentual limpio (12,5% o 15%).
- cuit_valido(c): Validación de CUIT mediante algoritmo módulo 11.
- numero_pedido(), numero_ticket(): Generadores de identificadores.
- registrar(nodo, txt): Generador de trazas de auditoría para el reducer 'add' de LangGraph.
- tabla(filas, encabezados): Formateador de tablas de texto plano.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Iterable, Sequence


def pesos(monto: float) -> str:
    """Formato monetario argentino: punto para miles, coma para decimales."""
    s = f"{monto:,.2f}"
    return "$" + s.replace(",", "_").replace(".", ",").replace("_", ".")


def porcentaje(valor: float) -> str:
    """Formato de porcentaje limpio: sin decimales innecesarios (15% en vez de 15,0%)."""
    texto = f"{valor:.1f}".replace(".", ",")
    return (texto[:-2] if texto.endswith(",0") else texto) + "%"


def cuit_valido(cuit: str) -> bool:
    """Valida el dígito verificador del CUIT mediante algoritmo módulo 11.
    No requiere conexión externa; valida la coherencia matemática del documento.
    """
    limpio = "".join(c for c in (cuit or "") if c.isdigit())
    if len(limpio) != 11:
        return False
    pesos_dv = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    resto = sum(int(d) * p for d, p in zip(limpio[:10], pesos_dv)) % 11
    dv = 0 if resto == 0 else (9 if resto == 1 else 11 - resto)
    return dv == int(limpio[10])


def numero_pedido() -> str:
    """Genera un código único de pedido comercial."""
    return f"PED-{datetime.now().year}-{random.randint(1000, 9999)}"


def numero_ticket() -> str:
    """Genera un código único de ticket de derivación/reclamo."""
    return f"TCK-{datetime.now().year}-{random.randint(1000, 9999)}"


def registrar(nodo: str, detalle: str) -> list[str]:
    """Devuelve UNA línea de log, envuelta en lista para el reducer `add` de LangGraph.

    Uso dentro de un nodo:
        return {"log": registrar("clasificar", "intencion=pedido")}

    Va en lista porque el reducer concatena listas: [viejas] + [nueva].
    """
    marca = datetime.now().strftime("%H:%M:%S")
    return [f"{marca} | {nodo:<18} | {detalle}"]


def tabla(filas: Iterable[Sequence[Any]], encabezados: Sequence[str]) -> str:
    """Arma una tabla de texto simple alineada en columnas para imprimir en consola."""
    filas_str = [[str(c) for c in f] for f in filas]
    anchos = [len(str(h)) for h in encabezados]
    for f in filas_str:
        for i, c in enumerate(f):
            if i < len(anchos):
                anchos[i] = max(anchos[i], len(c))
            else:
                anchos.append(len(c))

    def renglon(valores: Sequence[str]) -> str:
        return "  ".join(v.ljust(anchos[i]) for i, v in enumerate(valores))

    lineas = [
        renglon(encabezados),
        "  ".join("-" * anchos[i] for i in range(len(encabezados))),
    ]
    lineas.extend(renglon(f) for f in filas_str)
    return "\n".join(lineas)
