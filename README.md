# Mesa de Pedidos — Desafío 1 (IA Engineering)

**Diplomatura Universitaria en IA Engineering** · ICARO · UNC · FCEFyN

Sistema inteligente de procesamiento y validación de pedidos para **Ferretería Industrial Andina S.R.L.** implementado con **LangGraph**, **RAG sobre políticas** y **validación determinística de reglas comerciales en Python**.

---

## 📌 1. ¿De qué se trata el sistema?

El sistema automatiza el procesamiento de mensajes que ingresan por WhatsApp o correo electrónico:
1. **Clasifica** la intención del mensaje: `consulta`, `pedido` o `derivar`.
2. **Responde consultas** buscando en las políticas de la empresa (4 documentos PDF) citando de forma verificable el **documento y la página**.
3. **Extrae pedidos** con *Structured Output* (Pydantic).
4. **Valida stock, precios, mora y crédito disponible** consultando bases de datos locales (`catalogo.csv` y `clientes.csv`) **sin delegar cálculos matemáticos al LLM**.
5. **Human-in-the-Loop (`interrupt`)**: Si el pedido excede el crédito, presenta mora o requiere descuentos especiales, frena el flujo, guarda el estado en SQLite y solicita aprobación o rechazo a Gerencia Comercial antes de confirmar.

---

## 🏛️ 2. Arquitectura del Grafo (LangGraph)

El flujo cuenta con 9 nodos, 3 routers condicionales y 5 hojas. Las flechas punteadas son aristas condicionales (su destino se conoce recién cuando se ejecuta):

```mermaid
graph TD
	__start__([__start__]):::terminal
	cargar_cliente(cargar_cliente):::nodo
	clasificar(clasificar):::llm
	responder_consulta(responder_consulta):::llm
	extraer_pedido(extraer_pedido):::llm
	validar_pedido(validar_pedido):::nodo
	aprobacion_humana(aprobacion_humana):::hitl
	confirmar_pedido(confirmar_pedido):::ok
	rechazar_pedido(rechazar_pedido):::error
	derivar_humano(derivar_humano):::error
	__end__([__end__]):::terminal

	__start__ --> cargar_cliente
	cargar_cliente --> clasificar
	clasificar -. consulta .-> responder_consulta
	clasificar -. pedido .-> extraer_pedido
	clasificar -. derivar .-> derivar_humano
	extraer_pedido --> validar_pedido
	validar_pedido -. confirmar .-> confirmar_pedido
	validar_pedido -. aprobar .-> aprobacion_humana
	validar_pedido -. rechazar .-> rechazar_pedido
	validar_pedido -. derivar .-> derivar_humano
	aprobacion_humana -. aprueba .-> confirmar_pedido
	aprobacion_humana -. rechaza .-> rechazar_pedido
	responder_consulta --> __end__
	confirmar_pedido --> __end__
	rechazar_pedido --> __end__
	derivar_humano --> __end__

	classDef default fill:#21252b,stroke:#abb2bf,stroke-width:1.5px,color:#ffffff
	classDef terminal fill:#1e2227,stroke:#5c6370,stroke-width:2px,stroke-dasharray: 4 4,color:#e5c07b
	classDef nodo fill:#1f2937,stroke:#38bdf8,stroke-width:2px,color:#ffffff
	classDef llm fill:#2e1065,stroke:#c084fc,stroke-width:2px,color:#ffffff
	classDef hitl fill:#451a03,stroke:#fb923c,stroke-width:2px,color:#ffffff
	classDef ok fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ffffff
	classDef error fill:#4c0519,stroke:#f87171,stroke-width:2px,color:#ffffff
```

**9 nodos · 3 routers · 5 hojas:**

| Nodo | Usa LLM | Qué hace |
|---|---|---|
| `cargar_cliente` | no | Resuelve el CUIT contra `clientes.csv` |
| `clasificar` | sí | Consulta / pedido / derivar (structured output) |
| `responder_consulta` | sí | RAG: recupera fragmentos de políticas y responde citando |
| `extraer_pedido` | sí | Extrae ítems y cantidades del texto (structured output) |
| `validar_pedido` | **no** | Tools + reglas en Python: precio, descuento, crédito, stock → ruta |
| `aprobacion_humana` | no | `interrupt()`: frena y espera decisión de Gerencia |
| `confirmar_pedido` | redacta | Registra el pedido confirmado; el LLM solo redacta el mensaje |
| `rechazar_pedido` | redacta | Registra el rechazo; el LLM solo redacta el mensaje |
| `derivar_humano` | no | Genera ticket de derivación para atención humana |

---

## ⚙️ 3. Puesta en Marcha

### Prerrequisitos
- Python instalado en Windows (ejecutable mediante `py`).
- Clave de API de Google Gemini (o de OpenAI opcional).

### Instalación paso a paso

1. **Crear y activar el entorno virtual:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Instalar dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno:**
   Copia la plantilla `.envmodelo` a un nuevo archivo `.env` y coloca tu API Key:
   ```powershell
   copy .envmodelo .env
   ```
   Dentro de `.env`:
   ```env
   GOOGLE_API_KEY=tu_clave_real_aqui
   ```
   *(Nota de seguridad: el archivo `.env` está estrictamente ignorado en `.gitignore` para no filtrar credenciales).*

### 💡 Guía de Modelos y Cuotas en Google AI Studio (Free Tier vs. Paga)

Google actualiza frecuentemente las cuotas diarias y los nombres de los modelos en su API. Para verificar exactamente cuántas peticiones tenés disponibles en tu proyecto:

👉 **Consultar cuotas en tu cuenta:** [Panel de Rate Limits en Google AI Studio](https://aistudio.google.com/app/rate-limit)

| Tarea | Modelo API | Equivalente en Dashboard | Cuenta Paga | Capa Gratuita (Free Tier) |
|---|---|---|---|---|
| **Embeddings (RAG)** | `models/gemini-embedding-001` | `Gemini Embedding 1` / `2` | Ilimitado | **100 RPM** · **1.000 RPD** |
| **LLM (Cuenta Paga)** | `gemini-3.8-flash` | Gemini 3.8 Flash | Ilimitado | *(Limitado a 20 RPD)* |
| **LLM (Capa Gratuita)** | `gemini-3.1-flash-lite` / `gemini-3.5-flash-lite` | Flash Lite | Ilimitado | **15 RPM** · **500 RPD** (ideal para pruebas) |

**Recomendaciones para configurar según tu tipo de cuenta:**
- **Cuenta Paga (Pay-as-you-go):** Usar `gemini-3.8-flash` y `models/gemini-embedding-001` en `config.py`.
- **Cuenta Gratuita (Free Tier):** Elegir en `config.py` un modelo de texto con alto cupo diario (500 RPD como `gemini-3.1-flash-lite`), ya que modelos como 3.8 Flash tienen un tope de 20 consultas por día y se agotan antes de terminar los 20 casos de prueba.
- **Herramienta de diagnóstico:** Ejecutá `py -m mesa_pedidos.ver_modelos` en cualquier momento para ver la lista exacta de modelos activos con tu clave.

---

## 🎛️ 4. Configuración del Proyecto (`config.py`)

Todo lo relativo a modelos, rutas y reglas de negocio vive centralizado en `config.py`:

- **Proveedor y Modelos:**
  - Proveedor activo: `PROVEEDOR = "google"` (o `"openai"`).
  - LLM: `gemini-1.5-flash` (rápido y estructurado con temperatura 0.0).
  - Embeddings: `models/text-embedding-004`.
- **RAG:**
  - Estrategia de chunking: por artículos de políticas.
  - Búsqueda: Híbrida (`rank-bm25` + denso en FAISS).
- **Reglas de Negocio:**
  - Pedidos mínimos: Zona A ($250.000), Zona B ($250.000), Zona C ($120.000).
  - Topes de descuento por volumen y tope del 15% en marcas eléctricas (Dewalt, Makita, Bosch).
  - Días de mora (>30 requiere autorización, >60 cuenta suspendida).

---

## 📂 5. Estructura del Repositorio

```text
Mesa de Pedidos/
├── datos/                  # Corpus de RAG (politicas/) y CSVs de clientes/catálogo
├── mesa_pedidos/           # Paquete Python del sistema (RAG, tools, nodos, grafo)
├── artefactos/             # Índices FAISS y base SQLite checkpoints (ignorado en git)
├── config.py               # Configuración centralizada y reglas de negocio
├── correr.py               # Script de ejecución CLI y gestión de aprobaciones
├── evaluar.py              # Evaluación del Golden Set (20 mensajes de prueba)
├── requirements.txt        # Dependencias fijadas del proyecto
├── .envmodelo              # Plantilla segura de variables de entorno
├── .gitignore              # Protección estricta de secretos y cachés
├── AGENTS.md               # Instrucciones permanentes para el asistente IA
├── PLAN_DE_TRABAJO.md      # Plan detallado de fases del proyecto
└── README.md               # Esta documentación
```

---

## 🧪 6. Ejecución y Métricas

Una vez implementado, el sistema se opera mediante:

* **Indexar políticas:** `py -m mesa_pedidos.indexar`
* **Procesar un mensaje individual:** `py correr.py --mensaje MSG-009`
* **Reanudar pedido en espera de aprobación:** `py correr.py --mensaje MSG-013 --decision aprobar`
* **Medir rendimiento contra el Golden Set (20 casos):**
  ```powershell
  py evaluar.py
  ```
  El reporte evalúa:
  1. **Tasa de acierto de rama:** Porcentaje de casos que concluyen en la hoja esperada.
  2. **Presencia de dato clave:** Verificación de montos, plazos y condiciones recuperadas por el RAG.
