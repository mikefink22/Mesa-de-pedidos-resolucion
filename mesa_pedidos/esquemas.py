# -*- coding: utf-8 -*-
"""
esquemas.py - Modelos de Pydantic para Structured Output y validación de datos.

Define los esquemas fuertemente tipados que:
1. El LLM debe producir de forma garantizada (clasificación, extracción de pedido, respuesta RAG).
2. Las fuentes de datos representan (Cliente, Producto).
3. Python utiliza para validar decisiones y estructurar el estado del sistema.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# 1. Esquemas de Entidad (Datos Maestros de CSVs y Bandeja)
# ===========================================================================

class MensajeEntrante(BaseModel):
    """Un mensaje de entrada tal como proviene de bandeja.jsonl (WhatsApp o email)."""
    id: str
    canal: Literal["whatsapp", "email", "web"]
    fecha: str
    cuit: str
    remitente: str
    texto: str


class Cliente(BaseModel):
    """Representa un cliente con cuenta corriente en clientes.csv."""
    cuit: str
    razon_social: str
    provincia: str
    localidad: str
    condicion_iva: str
    categoria: Literal["A", "B", "C"]
    limite_credito: float
    saldo_actual: float
    dias_plazo: int
    descuento_habitual: float
    estado: Literal["activo", "suspendido"]
    mora_dias: int

    @property
    def credito_disponible(self) -> float:
        """Crédito que le queda disponible al cliente (sin saldo negativo)."""
        return max(0.0, self.limite_credito - self.saldo_actual)


class Producto(BaseModel):
    """Representa un artículo del catálogo en catalogo.csv."""
    sku: str
    nombre: str
    rubro: str
    marca: str
    unidad: str
    precio_lista: float
    alicuota_iva: float
    stock: int
    deposito: str


# ===========================================================================
# 2. Esquemas para RAG (Fase 1)
# ===========================================================================

class Cita(BaseModel):
    """Metadatos auditables de una fuente citada en una respuesta RAG."""
    documento: str
    pagina: int
    seccion: str = ""
    fragmento: str = ""


class RespuestaConsulta(BaseModel):
    """Salida estructurada que el LLM genera al responder una consulta RAG."""
    respuesta: str = Field(
        description="Respuesta redactada en lenguaje natural, cordial y profesional para el cliente."
    )
    fuentes: list[int] = Field(
        default_factory=list,
        description="Lista de números enteros correspondientes a los fragmentos del contexto utilizados (ej. [1, 3])."
    )
    encontrado: bool = Field(
        default=True,
        description="True si la información solicitada se encontró en las políticas; False si no está disponible."
    )


# ===========================================================================
# 3. Esquemas para Clasificación y Pedidos (Fases 2 y 3)
# ===========================================================================

class ItemPedido(BaseModel):
    """Renglón de producto solicitado por el cliente tal como lo escribió."""
    sku: Optional[str] = Field(
        default=None,
        description="Código de producto formato XX-0000 si fue provisto; None si no."
    )
    descripcion: str = Field(description="Nombre o descripción textual provista por el cliente.")
    cantidad: int = Field(gt=0, description="Cantidad solicitada (número entero positivo).")
    aclaracion: str = Field(default="", description="Detalles adicionales, marca o medidas.")

    @field_validator("sku")
    @classmethod
    def normalizar_sku(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper().replace(" ", "")
        return v or None


class PedidoExtraido(BaseModel):
    """Salida estructurada del LLM al extraer ítems de un mensaje de pedido."""
    items: list[ItemPedido] = Field(default_factory=list, description="Lista de renglones pedidos.")
    descuento_solicitado: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Porcentaje de descuento especial que solicita el cliente, si lo menciona explícitamente."
    )
    pago_anticipado: bool = Field(
        default=False,
        description="True si el cliente menciona que paga por adelantado, transferencia previa o contado."
    )
    destino: Optional[str] = Field(default=None, description="Localidad de entrega si la menciona.")
    observaciones: Optional[str] = Field(default=None, description="Cualquier indicación adicional.")


class ClasificacionMensaje(BaseModel):
    """Salida estructurada del LLM para clasificar la intención del cliente."""
    intencion: Literal["consulta", "pedido", "derivar"] = Field(
        description="Intención principal: 'consulta', 'pedido' o 'derivar'."
    )
    confianza: float = Field(
        ge=0.0,
        le=1.0,
        description="Nivel de certidumbre de la clasificación (0.0 a 1.0)."
    )
    justificacion: str = Field(
        default="",
        description="Breve explicación de por qué se asignó esta intención."
    )
    area_sugerida: Optional[str] = Field(
        default=None,
        description="Área sugerida si intención es derivar (ej. 'reclamos_y_logistica', 'administracion')."
    )


class DecisionHumana(BaseModel):
    """Decisión ingresada por Gerencia Comercial en el nodo Human-in-the-Loop."""
    tipo: Literal["aprobar", "editar", "rechazar"] = Field(
        description="Acción decidida por el operador: 'aprobar', 'editar' o 'rechazar'."
    )
    razon: str = Field(default="", description="Motivo de la aprobación, edición o rechazo.")
    ediciones: dict[str, int] = Field(
        default_factory=dict,
        description="En caso de editar: mapa {sku: nueva_cantidad} (cantidad 0 elimina el renglón)."
    )
    descuento_autorizado: Optional[float] = Field(
        default=None,
        description="Descuento especial autorizado en caso de edición."
    )


# ===========================================================================
# 4. Esquemas de Valorización y Lógica Comercial (Fase 2)
# ===========================================================================

class RenglonValorizado(BaseModel):
    """Un ítem del pedido tras cruzarlo contra el catálogo y aplicar descuentos."""
    sku: str
    nombre: str
    marca: str
    rubro: str
    cantidad: int
    precio_unitario: float
    subtotal_lista: float
    descuento_pct: float
    subtotal_neto: float
    alicuota_iva: float
    stock_disponible: int
    hay_stock: bool
    deposito: str = ""
    nota: str = ""


RutaComercial = Literal[
    "confirmar_pedido",
    "aprobacion_humana",
    "rechazar_pedido",
    "derivar_humano",
]


class ResultadoValidacion(BaseModel):
    """Resultado determinístico de validar un pedido comercial en Python puro."""
    ruta: RutaComercial
    motivo: str = ""
    alertas: list[str] = Field(default_factory=list)
    observaciones: list[str] = Field(default_factory=list)
    renglones: list[RenglonValorizado] = Field(default_factory=list)
    no_resueltos: list[dict] = Field(default_factory=list)
    bruto: float = 0.0
    descuento_volumen_pct: float = 0.0
    descuento_aplicado_pct: float = 0.0
    descuento_financiero: float = 0.0
    neto: float = 0.0
    iva: float = 0.0
    flete: float = 0.0
    total: float = 0.0
    credito: dict[str, Any] = Field(default_factory=dict)
    cotizacion_flete: dict[str, Any] = Field(default_factory=dict)
    pago_anticipado: bool = False


class ResumenPedido(BaseModel):
    """Documento formal emitido al finalizar el ciclo de un pedido (confirmado o rechazado)."""
    numero: str
    mensaje_id: str
    cuit: str
    razon_social: str
    estado: Literal["confirmado", "rechazado"]
    renglones: list[RenglonValorizado] = Field(default_factory=list)
    neto: float = 0.0
    descuento_aplicado_pct: float = 0.0
    flete: float = 0.0
    total: float = 0.0
    autorizado_por: Optional[str] = None
    motivo: str = ""


class Ticket(BaseModel):
    """Incidente derivado para atención humana (reclamos, casos fuera de catálogo, etc.)."""
    numero: str
    mensaje_id: str
    area: str
    motivo: str
    origen: Literal["clasificacion", "validacion"]
    resumen: str
    cliente: Optional[str] = None


