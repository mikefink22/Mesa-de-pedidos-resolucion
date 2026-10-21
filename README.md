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

El flujo cuenta con 9 nodos, 3 routers condicionales y 5 hojas:

```text
                        START
                          │
                   [cargar_cliente]          ← resuelve CUIT contra clientes.csv
                          │
                     [clasificar]            ← LLM + structured output
                          │
              ┌───────────┼───────────┐
        (consulta)    (pedido)    (derivar)
              │           │           │
   [responder_consulta]   │   [derivar_humano]
         ↑ RAG            │           │
              │     [extraer_pedido]  │      ← LLM + structured output
             END          │          END
                   [validar_pedido]          ← Tools + reglas comerciales (Python puro)
                          │
        ┌────────┬────────┴────────┬──────────┐
  (confirmar) (humano)      (rechazar)   (derivar)
        │        │                │           │
        │ [aprobacion_humana]     │           │
        │    │        │           │           │
        │ aprueba   rechaza       │           │
        │    │        │           │           │
   [confirmar_pedido] │           │           │
        │             └──→ [rechazar_pedido]  │
       END                        │    [derivar_humano]
                                 END          │
                                             END
```

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
