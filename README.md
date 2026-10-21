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
| `clasificar` | sí | Consulta / pedido / derivar (structured output con Pydantic) |
| `responder_consulta` | sí | RAG: recupera fragmentos de políticas y responde citando fuente |
| `extraer_pedido` | sí | Extrae ítems y cantidades del texto (structured output) |
| `validar_pedido` | **no** | Tools + reglas en Python: precio, descuento, crédito, stock → ruta |
| `aprobacion_humana` | no | `interrupt()`: frena y espera decisión de Gerencia |
| `confirmar_pedido` | redacta | Registra el pedido confirmado; el LLM solo redacta el mensaje |
| `rechazar_pedido` | redacta | Registra el rechazo; el LLM solo redacta el mensaje |
| `derivar_humano` | no | Genera ticket de derivación para atención humana |

---

## ⚙️ 3. Puesta en Marcha

### Prerrequisitos
- Python 3.10+ en Windows.
- Clave de API de Google Gemini (o de OpenAI opcional).

### Instalación paso a paso

1. **Crear y activar el entorno virtual:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

---

### 💡 Gestión de Cuotas y Modelos en Google AI Studio (Free Tier vs. Paga)

Google actualiza frecuentemente las cuotas diarias y los nombres de los modelos en su API. Para verificar exactamente cuántas peticiones tenés disponibles en tu proyecto:

👉 **Consultar cuotas en tu cuenta:** [Panel de Rate Limits en Google AI Studio](https://aistudio.google.com/app/rate-limit)

| Tarea | Modelo API | Equivalente en Dashboard | Cuenta Paga | Capa Gratuita (Free Tier) |
|---|---|---|---|---|
| **Embeddings (RAG)** | `models/gemini-embedding-001` | `Gemini Embedding 1` / `2` | Ilimitado | **100 RPM** · **1.000 RPD** |
| **LLM (Cuenta Paga)** | `gemini-3.8-flash` | Gemini 3.8 Flash | Ilimitado | *(Limitado a 20 RPD)* |
| **LLM (Capa Gratuita)** | `gemini-3.1-flash-lite` | Flash Lite | Ilimitado | **15 RPM** · **500 RPD** (recomendado para tests) |

**Recomendaciones para configurar según tu tipo de cuenta:**
- **Cuenta Paga (Pay-as-you-go):** Configurar `MODELO_LLM = "gemini-3.8-flash"` y `MODELO_EMBEDDINGS = "models/gemini-embedding-001"` en `config.py`.
- **Cuenta Gratuita (Free Tier):** Elegir en `config.py` un modelo de texto con alto cupo diario como `gemini-3.1-flash-lite` (500 RPD), ya que modelos como 3.8 Flash tienen un tope de 20 consultas por día y se agotan antes de terminar los 20 casos de prueba del Golden Set.
- **Herramienta de diagnóstico:** Ejecutá `.\.venv\Scripts\python.exe -m mesa_pedidos.ver_modelos` en cualquier momento para ver la lista exacta de modelos activos con tu clave.

---

## 🎛️ 4. Configuración del Proyecto (`config.py`)

Todo lo relativo a modelos, rutas y reglas de negocio vive centralizado en `config.py`:

- **Proveedor y Modelos Activos:**
  - Proveedor activo: `PROVEEDOR = "google"` (respaldo disponible para `"openai"`).
  - LLM: `gemini-3.8-flash` (cuenta paga) o `gemini-3.1-flash-lite` (cuenta gratuita, 500 RPD), con `temperatura: 0.0`.
  - Embeddings: `models/gemini-embedding-001` (100 RPM / 1.000 RPD).
- **RAG de Políticas:**
  - Estrategia de chunking: `articulos` (59 chunks estructurados por artículo y SKU, preservando excepciones legales).
  - Búsqueda: Híbrida (`rank-bm25` con peso 0.4 + denso en FAISS con peso 0.6).
- **Reglas Comerciales Determinísticas (Python puro):**
  - Pedidos mínimos por zona: Zona A ($250.000), Zona B ($250.000), Zona C ($120.000).
  - Escala de descuentos por volumen y tope estricto del 15% en herramientas eléctricas (Dewalt, Makita, Bosch).
  - Bonificación financiera por pago anticipado: 3% (opera fuera del tope).
  - Días de mora: 30 a 60 días requiere autorización gerencial; >60 días cuenta suspendida (solo habilitado con pago anticipado).

---

## 📂 5. Estructura del Repositorio

```text
Mesa de Pedidos/
├── datos/                          # Datos duros y corpus de negocio
│   ├── politicas/                  # 4 PDFs corporativos (corpus del RAG: 10 páginas)
│   ├── catalogo.csv                # 40 productos industriales con precio, IVA, stock y depósito
│   ├── clientes.csv                # 12 clientes con CUIT, límite, saldo y mora
│   ├── bandeja.jsonl               # 20 mensajes de entrada reales (simulación de bandeja)
│   └── golden_set.json             # 20 casos de referencia para evaluación final
├── mesa_pedidos/                   # Paquete Python del sistema
│   ├── __init__.py
│   ├── esquemas.py                 # Modelos Pydantic para Structured Output y datos duros
│   ├── estado.py                   # TypedDict (EstadoMesa) con reducer Annotated[list[str], add]
│   ├── tools.py                    # Tools comerciales (@tool con args_schema y tolerancia tipográfica)
│   ├── reglas.py                   # Política comercial en Python puro (cero matemática en LLM)
│   ├── indexar.py                  # Ingesta, chunking semántico por artículo e índice FAISS
│   ├── recuperacion.py             # Retriever híbrido (BM25 + FAISS) y armado fidedigno de citas
│   ├── llm.py                      # Fábrica del cliente Google Gemini con Structured Output
│   ├── prompts.py                  # Prompts protegidos contra inyecciones y sin operaciones matemáticas
│   ├── nodos.py                    # Implementación de los 9 nodos del grafo
│   ├── grafo.py                    # Ensamblado del StateGraph, 3 routers condicionales y SqliteSaver
│   └── ver_modelos.py              # Script de diagnóstico para consultar cuotas y modelos de Gemini
├── tests/                          # Suite de pruebas unitarias determinísticas (sin consumo de API)
│   ├── test_tools.py               # Pruebas de consultar_producto y verificar_credito
│   ├── test_reglas.py              # Pruebas de descuentos, topes, mora, fletes y decidir_ruta
│   ├── test_estado.py              # Validación de EstadoMesa, reducers de logs y serialización
│   └── test_grafo.py               # Verificación de compilación, topología y routers pasamanos
├── docs/                           # Documentación técnica y manual de estudio
│   ├── CONSIGNA.md                 # Enunciado y requerimientos formales de la diplomatura
│   └── GUIA_CURADA.md              # Bitácora técnica y guía pedagógica de IA Engineering
├── artefactos/                     # Salidas generadas por el sistema (ignorado en git)
│   ├── indice_faiss/               # Vectorstore FAISS persistido con su manifiesto
│   ├── checkpoints.sqlite          # Base transaccional para pausar y reanudar pedidos (HITL)
│   └── salidas/                    # Registro de pedidos confirmados, rechazados y tickets
├── config.py                       # Configuración centralizada única (rutas, modelos, parámetros)
├── correr.py                       # Consola CLI para procesar mensajes y gestionar aprobaciones
├── evaluar.py                      # Evaluación automatizada contra el Golden Set (Fase 6)
├── requirements.txt                # Dependencias fijadas del proyecto
├── .envmodelo                      # Plantilla segura de variables de entorno
├── .gitignore                      # Protección estricta de secretos, bases y cachés
├── AGENTS.md                       # Reglas permanentes para el asistente IA
├── PLAN_DE_TRABAJO.md              # Plan de trabajo estructurado por fases
└── README.md                       # Documentación principal del proyecto
```

---

## 🧪 6. Ejecución, Pruebas y Métricas

Todas las ejecuciones deben realizarse con el entorno virtual activo:

### 1. Pruebas Unitarias (Sin costo de API)
Verifican el 100% de la lógica determinística de negocio en milisegundos:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 2. Indexación de Políticas (RAG)
Genera el índice vectorial FAISS a partir de los 4 documentos PDF (se persiste en disco):
```powershell
.\.venv\Scripts\python.exe -m mesa_pedidos.indexar
```
*Para inspeccionar los 59 chunks generados sin llamar a la API:*
```powershell
.\.venv\Scripts\python.exe -m mesa_pedidos.indexar --ver
```

### 3. Procesamiento y Consola Interactiva (`correr.py`)
| Comando | Qué hace |
|---|---|
| `.\.venv\Scripts\python.exe correr.py --mensaje MSG-009` | Procesa un pedido estándar: valida y confirma automáticamente. |
| `.\.venv\Scripts\python.exe correr.py --mensaje MSG-013` | Pedido que supera el crédito: se frena en `interrupt()` y guarda checkpoint. |
| `.\.venv\Scripts\python.exe correr.py --pendientes` | Lista todos los pedidos frenados esperando decisión de Gerencia. |
| `.\.venv\Scripts\python.exe correr.py --mensaje MSG-013 --decision aprobar --razon "Autorizado"` | Reanuda el hilo en SQLite y confirma el pedido. |
| `.\.venv\Scripts\python.exe correr.py --texto "Cuánto sale el flete a Salta?" --cuit 30-71455982-2` | Procesa una consulta informal escrita a mano. |
| `.\.venv\Scripts\python.exe correr.py --todos` | Procesa en lote los 20 mensajes de `bandeja.jsonl`. |
| `.\.venv\Scripts\python.exe correr.py --grafo` | Imprime el diagrama del grafo en sintaxis Mermaid. |

### 4. Evaluación Formal contra el Golden Set (`evaluar.py`)
Mide el rendimiento integral del sistema contra los 20 casos de prueba de referencia:
```powershell
.\.venv\Scripts\python.exe evaluar.py
```

Resultados consolidados de la evaluación benchmark:
```text
==========================================================================================
METRICAS GLOBALES DEL BENCHMARK (20 CASOS - GOLDEN SET):
  1. Ruteo Correcto:           20/20 (100.0%)
  2. Consultas con Dato Clave:   8/8 (100.0%)
  3. Intencion Correcta:       20/20 (100.0%)
  4. Documento Correcto:         8/8 (100.0%)
  5. Extraccion de Items:      Precision: 1.000 | Recall: 1.000
==========================================================================================
```

El reporte estructurado se persiste automáticamente en:
* `artefactos/salidas/evaluacion.json` (reporte estructurado oficial de entrega).
* `artefactos/salidas/evaluacion_hibrido.json` (versión con detalle técnico del retriever).

Para un análisis detallado de la arquitectura, decisiones de diseño, casos de estudio y justificación teórica, consultar el **Informe Final de Entrega** en:
👉 [docs/INFORME_FINAL.md](docs/INFORME_FINAL.md)
