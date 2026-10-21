# -*- coding: utf-8 -*-
"""
nodos.py - Implementación de los 9 Nodos del Grafo de LangGraph.

Cada nodo es una función atómica pura en el grafo que:
1. Recibe el estado actual (EstadoMesa).
2. Ejecuta su tarea específica (consultar datos, invocar LLM con schema, o aplicar reglas).
3. Devuelve ÚNICAMENTE un diccionario con los campos calculados o modificados.
4. Genera una línea de traza auditable para el log acumulativo (`registrar(nodo, detalle)`).

Principio de Arquitectura:
- NINGÚN NODO CONTIENE 'IF' DE RUTEO DE NEGOCIO.
  El nodo ejecuta su lógica y escribe el resultado en el estado.
  Las decisiones sobre cuál es el siguiente paso las toman los routers externos en `grafo.py`.

Mapa de los 9 Nodos:
| # | Nodo                | Usa LLM | Propósito                                               |
|---|---------------------|---------|---------------------------------------------------------|
| 1 | cargar_cliente      | No      | Resuelve el CUIT contra clientes.csv                    |
| 2 | clasificar          | Sí      | Clasifica intención (consulta/pedido/derivar) + guardrail |
| 3 | responder_consulta  | Sí      | RAG sobre los 4 PDFs con citas verificables              |
| 4 | extraer_pedido      | Sí      | Extrae renglones, cantidades y pago anticipado con schema |
| 5 | validar_pedido      | NO      | Valida stock, precios, crédito y flete en Python puro   |
| 6 | aprobacion_humana   | No      | Human-in-the-Loop: interrupción y reanudación con interrupt|
| 7 | confirmar_pedido    | Sí*     | Emite ResumenPedido y redacta confirmación con datos duros|
| 8 | rechazar_pedido     | Sí*     | Emite ResumenPedido rechazado y redacta alternativa     |
| 9 | derivar_humano      | No      | Emite Ticket formal para atención humana por área       |

(*) El LLM solo redacta el texto cordial a partir de los números ya calculados en Python.
"""

from __future__ import annotations

from typing import Any, Optional
from langgraph.types import interrupt

import config
from mesa_pedidos import datos, llm, prompts, recuperacion, reglas, tools
from mesa_pedidos.datos import normalizar_texto
from mesa_pedidos.esquemas import (
    ClasificacionMensaje,
    DecisionHumana,
    PedidoExtraido,
    RespuestaConsulta,
    ResultadoValidacion,
    ResumenPedido,
    Ticket,
)
from mesa_pedidos.utilidades import (
    numero_pedido,
    numero_ticket,
    pesos,
    porcentaje,
    registrar,
)


# ===========================================================================
# 1. cargar_cliente (Sin LLM)
# ===========================================================================

def cargar_cliente(state: dict[str, Any]) -> dict[str, Any]:
    """Resuelve el CUIT informado en el mensaje contra la base de clientes.csv.
    
    Es el nodo inicial del grafo porque la condición del cliente (categoría,
    crédito, mora, localidad) condiciona todas las evaluaciones posteriores.
    Si el CUIT no existe, no falla: anota el error y deja cliente=None.
    """
    cuit = state.get("cuit", "")
    cliente = datos.buscar_cliente(cuit)
    
    if cliente is None:
        return {
            "cliente": None,
            "errores": [f"CUIT {cuit} no registrado en clientes.csv"],
            "log": registrar("cargar_cliente", f"CUIT {cuit} desconocido (no registrado)"),
        }
    
    detalle_auditoria = (
        f"{cliente.razon_social} | Cat. {cliente.categoria} | Estado: {cliente.estado} | "
        f"Mora: {cliente.mora_dias} d | Crédito disp.: {pesos(cliente.credito_disponible)}"
    )
    return {
        "cliente": cliente.model_dump(),
        "log": registrar("cargar_cliente", detalle_auditoria),
    }


# ===========================================================================
# 2. clasificar (LLM + Structured Output + Guardrail en Python)
# ===========================================================================

def clasificar(state: dict[str, Any]) -> dict[str, Any]:
    """Clasifica la intención del mensaje del cliente en: consulta, pedido o derivar.
    
    El LLM interpreta el lenguaje natural del remitente y devuelve una
    instancia fuertemente tipada de ClasificacionMensaje.
    Inmediatamente, Python aplica los guardrails determinísticos de `reglas.py`
    (confianza mínima de 0.6 o pedidos de CUITs no registrados).
    """
    texto = state.get("texto", "")
    
    # 1. Invocación al LLM con Structured Output
    invocador = prompts.CLASIFICAR | llm.chat(ClasificacionMensaje)
    resultado: ClasificacionMensaje = invocador.invoke({"texto": texto})
    clasificacion_dict = resultado.model_dump()
    
    # 2. Guardrails determinísticos en Python puro
    intencion_final, nota_guardrail = reglas.intencion_efectiva(
        clasificacion_dict, state.get("cliente")
    )
    
    detalle_log = f"{resultado.intencion} (confianza {resultado.confianza:.2f}) - {resultado.justificacion}"
    if nota_guardrail:
        detalle_log += f" -> [Guardrail]: {nota_guardrail}"
        
    return {
        "clasificacion": clasificacion_dict,
        "intencion": intencion_final,
        "log": registrar("clasificar", detalle_log),
    }


# ===========================================================================
# 3. responder_consulta (RAG con Citación Verificable)
# ===========================================================================

def _formatear_ficha_cliente(cliente: Optional[dict[str, Any]]) -> str:
    """Prepara un resumen plano de los datos del cliente para el contexto del RAG."""
    if not cliente:
        return "Cliente remitente no identificado en base de cuentas corrientes."
    return (
        f"{cliente.get('razon_social', '')}, {cliente.get('localidad', '')} ({cliente.get('provincia', '')}), "
        f"Condición IVA: {cliente.get('condicion_iva', '')}, Categoría comercial: {cliente.get('categoria', '')}"
    )


def responder_consulta(state: dict[str, Any]) -> dict[str, Any]:
    """Responde preguntas sobre políticas de la empresa citando documento y página.
    
    Flujo del nodo RAG:
    1. Recupera los fragmentos más relevantes de los 4 PDFs (híbrido BM25 + FAISS).
    2. Precalcula apoyos numéricos en Python si la consulta menciona montos.
    3. Invoca a Google Gemini con Structured Output (RespuestaConsulta).
    4. Python traduce los números de 'fuentes' a citas verificables auditadas.
    """
    texto = state.get("texto", "")
    cliente = state.get("cliente")
    documentos = recuperacion.recuperar(texto)
    
    # Pre-cálculos determinísticos para evitar que el LLM calcule números
    calculos_apoyo = reglas.calculos_para_consulta(texto, cliente)
    contexto_formateado = recuperacion.formatear_contexto(documentos)
    
    invocador = prompts.CONSULTA | llm.chat(RespuestaConsulta)
    resultado: RespuestaConsulta = invocador.invoke({
        "fecha": state.get("fecha", ""),
        "cliente": _formatear_ficha_cliente(cliente),
        "calculos": calculos_apoyo,
        "contexto": contexto_formateado,
        "texto": texto,
    })
    
    # Traducción de fuentes por Python (garantía anti-alucinaciones)
    citas = recuperacion.citas_desde_fuentes(documentos, resultado.fuentes)
    
    respuesta_final = resultado.respuesta.strip()
    if citas:
        pie_citas = "; ".join(f"{c['documento']}, pág. {c['pagina']}" for c in citas)
        respuesta_final += f"\n\n[Fuente: {pie_citas}]"
        
    estado_rag = "respondida con citas" if resultado.encontrado else "NO encontrada en políticas"
    detalle_log = f"RAG {estado_rag} | {len(citas)} citas verificadas | {len(documentos)} fragmentos evaluados"
    
    return {
        "respuesta": respuesta_final,
        "citas": citas,
        "fragmentos": recuperacion.describir(documentos),
        "ruta": "responder_consulta",
        "log": registrar("responder_consulta", detalle_log),
    }


# ===========================================================================
# 4. extraer_pedido (LLM + Structured Output)
# ===========================================================================

def extraer_pedido(state: dict[str, Any]) -> dict[str, Any]:
    """Extrae del texto del cliente los ítems solicitados, cantidades y condiciones.
    
    No realiza validaciones contra el catálogo ni cálculos comerciales; su única
    responsabilidad es estructurar el lenguaje natural en un modelo PedidoExtraido.
    """
    texto = state.get("texto", "")
    invocador = prompts.EXTRAER | llm.chat(PedidoExtraido)
    resultado: PedidoExtraido = invocador.invoke({"texto": texto})
    
    resumen_items = ", ".join(f"{i.sku or '?'} x{i.cantidad}" for i in resultado.items) or "ninguno"
    detalles_extra: list[str] = []
    if resultado.descuento_solicitado:
        detalles_extra.append(f"solicita {porcentaje(resultado.descuento_solicitado)} desc.")
    if resultado.pago_anticipado:
        detalles_extra.append("pago anticipado")
    if resultado.destino:
        detalles_extra.append(f"destino: {resultado.destino}")
        
    extra_str = f" | {', '.join(detalles_extra)}" if detalles_extra else ""
    return {
        "pedido": resultado.model_dump(),
        "log": registrar("extraer_pedido", f"{len(resultado.items)} ítems extraídos ({resumen_items}){extra_str}"),
    }


# ===========================================================================
# 5. validar_pedido (CERO LLM: Tools + Reglas Comerciales en Python Puro)
# ===========================================================================

def _resolver_items_catalogo(pedido: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Cruza cada renglón del pedido contra catalogo.csv usando la tool `consultar_producto`.
    
    Retorna: (resueltos, no_resueltos, inferidos_por_descripcion)
    """
    resueltos: list[dict[str, Any]] = []
    no_resueltos: list[dict[str, Any]] = []
    inferidos: list[str] = []
    
    for item in pedido.get("items", []):
        info = tools.consultar_producto.invoke({
            "sku": item.get("sku"),
            "descripcion": item.get("descripcion"),
            "cantidad": item.get("cantidad", 1),
        })
        
        if info.get("encontrado"):
            info["cantidad"] = item.get("cantidad", 1)
            resueltos.append(info)
            if info.get("inferido_por_descripcion"):
                inferidos.append(info["sku"])
        elif len(info.get("sugerencias", [])) == 1:
            # Coincidencia única por sugerencia de alta similitud
            sku_sugerido = info["sugerencias"][0]["sku"]
            prod_info = tools.consultar_producto.invoke({
                "sku": sku_sugerido,
                "cantidad": item.get("cantidad", 1),
            })
            prod_info["cantidad"] = item.get("cantidad", 1)
            resueltos.append(prod_info)
            inferidos.append(sku_sugerido)
        else:
            no_resueltos.append({
                "descripcion": item.get("descripcion", ""),
                "sku": item.get("sku"),
                "sugerencias": info.get("sugerencias", []),
            })
            
    return resueltos, no_resueltos, inferidos


def _calcular_totales_pedido(
    cliente: dict[str, Any],
    items: list[dict[str, Any]],
    pago_anticipado: bool,
    descuento_solicitado: Optional[float] = None,
    descuento_autorizado: Optional[float] = None,
) -> dict[str, Any]:
    """Ejecuta los cálculos determinísticos de valorización, crédito y flete."""
    valor = reglas.valorizar(
        items=items,
        cliente=cliente,
        descuento_solicitado=descuento_solicitado,
        pago_anticipado=pago_anticipado,
        descuento_autorizado=descuento_autorizado,
    )
    
    credito = tools.verificar_credito.invoke({
        "cuit": cliente.get("cuit", ""),
        "monto_neto": valor["neto"],
        "pago_anticipado": pago_anticipado,
    })
    
    # Contar unidades de aerosoles para recargo de carga peligrosa (Logística, Punto 5)
    unidades_aerosol = sum(
        i.get("cantidad", i.get("cantidad_pedida", 1)) for i in items if "aerosol" in normalizar_texto(i.get("nombre", ""))
    )
    
    flete = reglas.cotizar_flete(
        provincia=cliente.get("provincia", ""),
        localidad=cliente.get("localidad", ""),
        monto_neto=valor["neto"],
        pago_anticipado=pago_anticipado,
        unidades_aerosol=unidades_aerosol,
    )
    
    observaciones = list(valor.get("notas", [])) + [flete.get("detalle", "")]
    
    # Verificar despachos multi-depósito (Córdoba y Rosario)
    depositos = {i.get("deposito") for i in items if i.get("deposito")}
    if len(depositos) > 1:
        observaciones.append(
            "Artículos repartidos en depósitos Córdoba y Rosario: despacho centralizado en Córdoba "
            "(+2 días hábiles de transferencia interna, Logística Punto 1)"
        )
        
    total_con_iva = round(valor["neto"] + valor["iva"] + flete["costo"] * 1.21, 2)
    
    return {
        "valor": valor,
        "credito": credito,
        "flete": flete,
        "observaciones": observaciones,
        "total": total_con_iva,
    }


def validar_pedido(state: dict[str, Any]) -> dict[str, Any]:
    """El motor comercial central: valida stock, precios, crédito y flete sin usar LLM.
    
    Aplica las reglas comerciales y determina a qué hoja del grafo corresponde
    enviar el pedido (confirmar_pedido, aprobacion_humana, rechazar_pedido, derivar_humano).
    """
    cliente = state.get("cliente") or {}
    pedido = state.get("pedido") or {}
    pago_anticipado = bool(pedido.get("pago_anticipado", False))
    
    # 1. Resolver ítems contra el catálogo CSV
    items_resueltos, no_resueltos, inferidos = _resolver_items_catalogo(pedido)
    
    # 2. Calcular importes, crédito y flete
    calculos = _calcular_totales_pedido(
        cliente=cliente,
        items=items_resueltos,
        pago_anticipado=pago_anticipado,
        descuento_solicitado=pedido.get("descuento_solicitado"),
    )
    valor = calculos["valor"]
    
    # Advertencia si el destino mencionado difiere de la localidad de la ficha
    destino_mencionado = pedido.get("destino")
    if destino_mencionado and normalizar_texto(cliente.get("localidad", "")) not in normalizar_texto(destino_mencionado):
        calculos["observaciones"].append(
            f"El cliente mencionó destino '{destino_mencionado}'; el flete fue cotizado al "
            f"domicilio fiscal registrado ({cliente.get('localidad')}, {cliente.get('provincia')})"
        )
        
    # 3. Aplicar motor determinístico de ruteo
    ruta_determinada, motivo_principal, alertas = reglas.decidir_ruta(
        cliente=cliente,
        valorizacion=valor,
        credito=calculos["credito"],
        flete=calculos["flete"],
        no_resueltos=no_resueltos,
        inferidos=inferidos,
        descuento_solicitado=pedido.get("descuento_solicitado"),
    )
    
    resultado = ResultadoValidacion(
        ruta=ruta_determinada,
        motivo=motivo_principal,
        alertas=alertas,
        observaciones=calculos["observaciones"],
        renglones=valor.get("renglones", []),
        no_resueltos=no_resueltos,
        bruto=valor.get("bruto", 0.0),
        descuento_volumen_pct=valor.get("descuento_volumen_pct", 0.0),
        descuento_aplicado_pct=valor.get("descuento_aplicado_pct", 0.0),
        descuento_financiero=valor.get("descuento_financiero", 0.0),
        neto=valor.get("neto", 0.0),
        iva=valor.get("iva", 0.0),
        flete=calculos["flete"].get("costo", 0.0),
        total=calculos["total"],
        credito=calculos["credito"],
        cotizacion_flete=calculos["flete"],
        pago_anticipado=pago_anticipado,
    )
    
    detalle_log = (
        f"Neto: {pesos(resultado.neto)} | Desc: {porcentaje(resultado.descuento_aplicado_pct)} | "
        f"{len(alertas)} alertas comerciales -> Sugiere: {ruta_determinada}"
    )
    return {
        "validacion": resultado.model_dump(),
        "log": registrar("validar_pedido", detalle_log),
    }


# ===========================================================================
# 6. aprobacion_humana (Human-in-the-Loop con interrupt)
# ===========================================================================

def _payload_aprobacion(state: dict[str, Any]) -> dict[str, Any]:
    """Genera el payload completo que se pasa a interrupt() para el operador humano."""
    v = state.get("validacion") or {}
    c = state.get("cliente") or {}
    return {
        "mensaje_id": state.get("mensaje_id", ""),
        "razon_social": c.get("razon_social", ""),
        "cuit": c.get("cuit", ""),
        "categoria": c.get("categoria", ""),
        "mora_dias": c.get("mora_dias", 0),
        "limite_credito": c.get("limite_credito", 0.0),
        "saldo_actual": c.get("saldo_actual", 0.0),
        "credito_disponible": max(0.0, float(c.get("limite_credito", 0)) - float(c.get("saldo_actual", 0))),
        "neto": v.get("neto", 0.0),
        "descuento_aplicado_pct": v.get("descuento_aplicado_pct", 0.0),
        "total": v.get("total", 0.0),
        "motivo": v.get("motivo", ""),
        "alertas": v.get("alertas", []),
        "renglones": [
            {
                "sku": r["sku"],
                "nombre": r["nombre"],
                "cantidad": r["cantidad"],
                "stock": r["stock_disponible"],
                "descuento_pct": r["descuento_pct"],
                "subtotal_neto": r["subtotal_neto"],
            }
            for r in v.get("renglones", [])
        ],
        "texto_original": state.get("texto", ""),
        "opciones": ["aprobar", "editar", "rechazar"],
    }


def _aplicar_edicion_humana(state: dict[str, Any], decision: DecisionHumana) -> dict[str, Any]:
    """Recalcula el pedido si la Gerencia Comercial modificó cantidades o autorizó un descuento."""
    v = state.get("validacion") or {}
    cliente = state.get("cliente") or {}
    items_recalculados: list[dict[str, Any]] = []
    
    for r in v.get("renglones", []):
        nueva_cantidad = decision.ediciones.get(r["sku"], r["cantidad"])
        if nueva_cantidad > 0:
            items_recalculados.append({
                "sku": r["sku"],
                "nombre": r["nombre"],
                "marca": r["marca"],
                "rubro": r["rubro"],
                "cantidad": nueva_cantidad,
                "precio_lista": r["precio_unitario"],
                "alicuota_iva": r["alicuota_iva"],
                "stock": r["stock_disponible"],
                "deposito": r.get("deposito", ""),
            })
            
    calc = _calcular_totales_pedido(
        cliente=cliente,
        items=items_recalculados,
        pago_anticipado=v.get("pago_anticipado", False),
        descuento_autorizado=decision.descuento_autorizado,
    )
    valor = calc["valor"]
    
    editado = ResultadoValidacion(
        ruta="confirmar_pedido",
        motivo=f"Editado y autorizado por {decision.autorizado_por}: {decision.razon}".strip(": "),
        alertas=v.get("alertas", []),
        observaciones=calc["observaciones"] + ["Pedido ajustado por Gerencia Comercial previo a la confirmación"],
        renglones=valor.get("renglones", []),
        bruto=valor.get("bruto", 0.0),
        descuento_volumen_pct=valor.get("descuento_volumen_pct", 0.0),
        descuento_aplicado_pct=valor.get("descuento_aplicado_pct", 0.0),
        descuento_financiero=valor.get("descuento_financiero", 0.0),
        neto=valor.get("neto", 0.0),
        iva=valor.get("iva", 0.0),
        flete=calc["flete"].get("costo", 0.0),
        total=calc["total"],
        credito=calc["credito"],
        cotizacion_flete=calc["flete"],
        pago_anticipado=v.get("pago_anticipado", False),
    )
    return editado.model_dump()


def aprobacion_humana(state: dict[str, Any]) -> dict[str, Any]:
    """Punto Human-in-the-Loop: pausa la ejecución con interrupt() y espera la decisión humana.
    
    Al invocar interrupt(), LangGraph guarda el checkpoint en SQLite y frena.
    Al reanudar con Command(resume=decision), la ejecución continúa desde este punto.
    """
    # 1. Pausa y emisión del payload para el supervisor humano
    decision_cruda = interrupt(_payload_aprobacion(state))
    
    # --- De aquí hacia abajo solo se ejecuta al retomar (resume) ---
    if isinstance(decision_cruda, str):
        decision_cruda = {"tipo": decision_cruda}
        
    try:
        decision = DecisionHumana(**decision_cruda)
    except Exception:
        # Por seguridad comercial, cualquier decisión no estructurada o ilegible rechaza el pedido
        decision = DecisionHumana(tipo="rechazar", razon=f"Decisión inválida o no reconocida: {decision_cruda!r}")
        
    salida: dict[str, Any] = {"decision_humana": decision.model_dump()}
    detalle = f"{decision.tipo.upper()} por {decision.autorizado_por}"
    if decision.razon:
        detalle += f" - Motivo: {decision.razon}"
        
    if decision.tipo == "editar":
        salida["validacion"] = _aplicar_edicion_humana(state, decision)
        detalle += f" | Nuevo neto recalculado: {pesos(salida['validacion']['neto'])}"
        
    salida["log"] = registrar("aprobacion_humana", detalle)
    return salida


# ===========================================================================
# 7. confirmar_pedido (Hoja Terminal: Emisión de Resumen y Redacción Cordial)
# ===========================================================================

def confirmar_pedido(state: dict[str, Any]) -> dict[str, Any]:
    """Genera el pedido confirmado formal, lo persiste en disco y redacta el mensaje de confirmación."""
    v = state.get("validacion") or {}
    c = state.get("cliente") or {}
    decision = state.get("decision_humana") or {}
    
    resumen = ResumenPedido(
        numero=numero_pedido(),
        mensaje_id=state.get("mensaje_id", ""),
        cuit=c.get("cuit", ""),
        razon_social=c.get("razon_social", ""),
        estado="confirmado",
        renglones=v.get("renglones", []),
        neto=v.get("neto", 0.0),
        descuento_aplicado_pct=v.get("descuento_aplicado_pct", 0.0),
        flete=v.get("flete", 0.0),
        total=v.get("total", 0.0),
        autorizado_por=decision.get("autorizado_por"),
        motivo=v.get("motivo", "Pedido aprobado bajo condiciones comerciales vigentes"),
    )
    
    # Preparar datos ya calculados para el prompt (cero matemática en el LLM)
    items_lineas = []
    for r in v.get("renglones", []):
        linea = f"- {r['cantidad']} x {r['nombre']} ({r['sku']}) [Desc: {porcentaje(r['descuento_pct'])}]"
        if r.get("nota"):
            linea += f" ({r['nota']})"
        items_lineas.append(linea)
    items_texto = "\n".join(items_lineas)
    datos_redaccion = (
        f"Cliente: {c.get('razon_social', '')}\n"
        f"Número de pedido asignado: {resumen.numero}\n"
        f"Renglones confirmados:\n{items_texto}\n"
        f"Neto de mercadería: {pesos(v.get('neto', 0.0))}\n"
        f"Descuento comercial aplicado: {porcentaje(v.get('descuento_aplicado_pct', 0.0))}\n"
        f"Flete: {'Bonificado al 100%' if v.get('flete', 0) == 0 else pesos(v.get('flete', 0)) + ' + IVA'}\n"
        f"Total con IVA: {pesos(v.get('total', 0.0))}\n"
        f"Plazo estimado de entrega: {v.get('cotizacion_flete', {}).get('plazo', '24 a 48 hs hábiles')}"
    )
    
    invocador = prompts.CONFIRMACION | llm.chat()
    respuesta_llm = invocador.invoke({
        "datos": datos_redaccion,
        "texto": state.get("texto", ""),
    }).content
    
    # Persistir pedido en disco (artefactos/salidas/pedidos/MSG-XXX.json)
    ruta_guardada = datos.guardar_salida(f"pedidos/{state.get('mensaje_id', 'pedido')}.json", resumen)
    
    autorizador_str = f" | Autorizado por {resumen.autorizado_por}" if resumen.autorizado_por else ""
    return {
        "resumen": resumen.model_dump(),
        "respuesta": respuesta_llm.strip(),
        "ruta": "confirmar_pedido",
        "log": registrar(
            "confirmar_pedido",
            f"{resumen.numero} emitido | Total: {pesos(resumen.total)}{autorizador_str} | Guardado en {ruta_guardada.name}",
        ),
    }


# ===========================================================================
# 8. rechazar_pedido (Hoja Terminal: Emisión de Rechazo y Alternativa Comercial)
# ===========================================================================

def rechazar_pedido(state: dict[str, Any]) -> dict[str, Any]:
    """Registra formalmente el rechazo del pedido y redacta una respuesta cordial con alternativa comercial."""
    v = state.get("validacion") or {}
    c = state.get("cliente") or {}
    decision = state.get("decision_humana") or {}
    
    if decision.get("tipo") == "rechazar":
        motivo = f"Gerencia Comercial no autorizó la operación en cuenta corriente. {decision.get('razon', '')}".strip()
        alternativa = "Podemos revisar cantidades del pedido o canalizar la compra mediante pago anticipado."
    else:
        motivo = v.get("motivo", "Inhabilitado para operar en cuenta corriente.")
        alternativa = "Podemos procesar el pedido con pago anticipado por transferencia bancaria (Art. 7)."
        
    resumen = ResumenPedido(
        numero=numero_pedido(),
        mensaje_id=state.get("mensaje_id", ""),
        cuit=c.get("cuit", ""),
        razon_social=c.get("razon_social", ""),
        estado="rechazado",
        renglones=v.get("renglones", []),
        neto=v.get("neto", 0.0),
        descuento_aplicado_pct=v.get("descuento_aplicado_pct", 0.0),
        flete=v.get("flete", 0.0),
        total=v.get("total", 0.0),
        autorizado_por=decision.get("autorizado_por"),
        motivo=motivo,
    )
    
    datos_redaccion = (
        f"Cliente: {c.get('razon_social', '')}\n"
        f"Remitente: {state.get('remitente', '')}\n"
        f"Motivo del rechazo: {motivo}\n"
        f"Alternativa comercial sugerida: {alternativa}"
    )
    
    invocador = prompts.RECHAZO | llm.chat()
    respuesta_llm = invocador.invoke({
        "datos": datos_redaccion,
        "texto": state.get("texto", ""),
    }).content
    
    ruta_guardada = datos.guardar_salida(f"pedidos/{state.get('mensaje_id', 'pedido')}.json", resumen)
    
    return {
        "resumen": resumen.model_dump(),
        "respuesta": respuesta_llm.strip(),
        "ruta": "rechazar_pedido",
        "log": registrar("rechazar_pedido", f"Rechazado: {motivo[:100]} | Guardado en {ruta_guardada.name}"),
    }


# ===========================================================================
# 9. derivar_humano (Hoja Terminal: Emisión de Ticket de Soporte)
# ===========================================================================

_AREAS_NOMBRE: dict[str, str] = {
    "mesa_de_pedidos": "Mesa de Pedidos",
    "reclamos_y_logistica": "Reclamos y Logística",
    "compras_proveedores": "Compras y Proveedores",
    "asesoramiento_tecnico": "Asesoramiento Técnico",
    "administracion": "Administración y Finanzas",
}


def derivar_humano(state: dict[str, Any]) -> dict[str, Any]:
    """Genera un Ticket formal para atención humana por parte de un operador o sector especialista.
    
    Distingue dos orígenes:
    - 'clasificacion': El mensaje no era consulta ni pedido estándar (reclamo, proveedor, baja confianza).
    - 'validacion': Era un pedido, pero ningún producto fue identificado en el catálogo.
    """
    clasif = state.get("clasificacion") or {}
    validacion = state.get("validacion") or {}
    
    if validacion.get("ruta") == "derivar_humano":
        origen = "validacion"
        area = "mesa_de_pedidos"
        motivo = validacion.get("motivo", "No se identificó ningún producto en catálogo")
    else:
        origen = "clasificacion"
        area = clasif.get("area_sugerida") or "mesa_de_pedidos"
        motivo = clasif.get("justificacion") or clasif.get("motivo") or "Mensaje fuera del alcance del sistema automatizado"
        if state.get("intencion") == "derivar" and clasif.get("intencion") != "derivar":
            motivo = f"{motivo} (derivado por regla de negocio: confianza baja o CUIT no registrado)"
            
    cliente = state.get("cliente") or {}
    area_formateada = _AREAS_NOMBRE.get(area, area)
    
    ticket = Ticket(
        numero=numero_ticket(),
        mensaje_id=state.get("mensaje_id", ""),
        area=area_formateada,
        motivo=motivo,
        origen=origen,
        resumen=state.get("texto", "")[:160],
        cliente=cliente.get("razon_social"),
    )
    
    ruta_guardada = datos.guardar_salida(f"tickets/{state.get('mensaje_id', 'ticket')}.json", ticket)
    
    # Saludo cordial y derivación sin gastar tokens de LLM
    nombre_remitente = (state.get("remitente") or "").split(" ")[0]
    saludo = f"Hola {nombre_remitente}" if nombre_remitente and "@" not in nombre_remitente else "Hola"
    respuesta = (
        f"{saludo}, gracias por comunicarte con {config.EMPRESA}. "
        f"Tu mensaje fue derivado al sector de {ticket.area} para una atención personalizada. "
        f"Un asesor se contactará a la brevedad. Tu número de ticket de seguimiento es {ticket.numero}."
    )
    
    return {
        "ticket": ticket.model_dump(),
        "respuesta": respuesta,
        "ruta": "derivar_humano",
        "log": registrar("derivar_humano", f"Ticket {ticket.numero} -> {ticket.area} (origen: {origen})"),
    }
