# 📘 Guía Curada de Aprendizaje: Mesa de Pedidos
**Diplomatura en IA Engineering (ICARO · UNC · FCEFyN)**
*Sistema inteligente de atención, validación comercial y ruteo con LangGraph, RAG y Human-in-the-Loop.*

---

## 🧭 1. Introducción y Fundamentos del Proyecto

### ¿Qué problema resolvemos?
En una distribuidora industrial (*Ferretería Industrial Andina S.R.L.*), los clientes se comunican por canales no estructurados (WhatsApp, correos) enviando:
1. **Consultas sobre políticas** (fletes por zona, garantías de máquinas, medios de pago, bonificaciones).
2. **Pedidos de mercadería** (mezclando descripciones coloquiales, marcas, cantidades).
3. **Casos especiales** (reclamos airados, cancelaciones, solicitudes de cuentas corrientes).

### El gran error de la IA tradicional vs. IA Engineering
Un error común en principiantes es pedirle a un LLM:
> *"Lee el catálogo, valida el stock, calcula el límite de crédito con la tasa de descuento y confírmame el pedido"*.

Esto **falla en producción** por tres razones:
1. **Alucinación matemática:** Los LLMs no son calculadoras; predicen el siguiente token estadísticamente. Confiarles sumas, porcentajes y límites crediticios produce errores financieros graves.
2. **Falta de determinismo:** Las reglas de negocio (mora >30 días requiere autorización, marcas con tope del 15% de descuento) deben ser 100% predecibles y auditables.
3. **Problema de contexto (Context Window y Ruidos):** Inyectar miles de líneas de catálogos y políticas en cada prompt es lento, costoso y propenso a omitir cláusulas finas.

### Los Tres Pilares de Nuestra Arquitectura
1. **LLM solo para tareas de Lenguaje Natural:**
   - Clasificar la intención del cliente (`consulta`, `pedido`, `derivar`).
   - Extraer items estructurados a partir de texto informal (*Structured Output* con Pydantic).
   - Redactar respuestas a consultas citando fuentes explícitas del RAG.
2. **Python Puro para Lógica de Negocio (Tools Determinísticas):**
   - Stock, precios de lista, límites de crédito, análisis de mora y cálculo de fletes se ejecutan en código Python exacto.
3. **Orquestación con LangGraph y Human-in-the-Loop:**
   - Modelamos el flujo como una máquina de estados finita (grafo dirigido cíclico/acíclico).
   - Si un pedido excede el crédito o el cliente presenta mora, el grafo **guarda su estado en SQLite y se frena** (`interrupt()`) esperando la decisión de un supervisor humano, que luego reanuda el proceso (`resume`).

---

## 📁 2. Fase 0: Estructura Modular y Configuración Centralizada

### Organización de Carpetas
```text
Mesa de Pedidos/
├── datos/                  <- Datos de entrada del negocio
│   ├── politicas/          <- 4 PDFs de políticas corporativas (corpus para RAG)
│   ├── catalogo.csv        <- 40 productos con SKU, descripción, precio, stock
│   ├── clientes.csv        <- 12 clientes con CUIT, límite, saldo, mora
│   ├── bandeja.jsonl       <- 20 mensajes reales para simulación
│   └── golden_set.json     <- Casos de prueba para evaluación
├── mesa_pedidos/           <- Paquete de Python del sistema
│   ├── __init__.py
│   ├── indexar.py          <- (Fase 1) Ingesta, chunking y vectorización de PDFs
│   ├── recuperacion.py     <- (Fase 1) Búsqueda vectorial/híbrida y LLM con citas
│   ├── tools.py            <- (Fase 2) Funciones comerciales determinísticas
│   ├── esquemas.py         <- (Fase 3) Modelos Pydantic para Structured Output
│   ├── estado.py           <- (Fase 3) TypedDict y reducers de LangGraph
│   ├── nodos.py            <- (Fase 3) Los 9 nodos del grafo
│   └── grafo.py            <- (Fase 4) Ensamblado de nodos, routers y SqliteSaver
├── artefactos/             <- Directorio de salidas generadas (ignorado en git)
│   ├── indice_faiss/       <- Índice vectorial persistido
│   └── checkpoints.sqlite  <- Base de persistencia de LangGraph
├── config.py               <- Archivo único de verdad (rutas, modelos, reglas)
├── correr.py               <- Script CLI para procesar mensajes e interactuar
└── evaluar.py              <- Suite de evaluación contra Golden Set
```

### ¿Por qué centralizar en `config.py`?
- **Evitar magic numbers y hardcoding:** Valores como `PEDIDO_MINIMO = {"A": 250_000, "B": 250_000, "C": 120_000}` o `MORA_AUTORIZACION_DIAS = 30` están en un solo lugar. Si la empresa cambia una política comercial, se ajusta aquí sin tocar el grafo ni los prompts.
- **Flexibilidad de proveedores:** Permite cambiar fácilmente entre Google Gemini y OpenAI.
- **Rutas seguras con `pathlib.Path`:** Resuelve rutas absolutas independientes del directorio desde el que se invoque el comando en Windows o Linux.

---

## 🔍 3. Fase 1: RAG de Políticas y Citación Verificable (Próximo Paso)

### Conceptos Clave
1. **¿Qué es RAG (Retrieval-Augmented Generation)?**
   En lugar de entrenar o hacer fine-tuning al modelo con los documentos de la empresa, RAG busca los fragmentos pertinentes en tiempo real y se los pasa al LLM como contexto en el prompt.

2. **Chunking por Artículos vs. Chunking Fijo:**
   - *Chunking fijo (por caracteres o tokens):* Corta arbitrariamente cada 500 caracteres con un solapamiento (overlap). En documentos legales o políticas, esto puede cortar un artículo por la mitad, separando la regla de su excepción.
   - *Chunking por Artículos (Estructural):* Detecta expresiones como `Artículo 1°`, `Artículo 2°` o secciones temáticas. Cada chunk es una unidad semántica autosuficiente.

3. **Embeddings (`models/text-embedding-004`):**
   Convierten fragmentos de texto en vectores matemáticos de alta dimensión (768 dimensiones). Textos con significados conceptualmente similares quedan cerca en el espacio vectorial.

4. **FAISS (Facebook AI Similarity Search):**
   Una librería ultra eficiente en C++ (con bindings para Python) para realizar búsquedas por similitud de cosenos o producto interno a escala.

5. **Búsqueda Híbrida (Dense + Sparse con BM25):**
   - **Embeddings densos:** Capturan la semántica general ("¿cómo me envían las cosas?" -> busca "fletes y logística").
   - **BM25 (Sparse / Léxico):** Busca palabras clave exactas y códigos específicos (por ejemplo: "Zona B", "garantía de 12 meses", "SKU").
   - La combinación ponderada de ambos evita que el modelo pierda detalles específicos mientras conserva la flexibilidad del lenguaje natural.

6. **Citación Verificable (Anti-Alucinaciones):**
   El prompt del RAG exige al modelo:
   - Contestar **únicamente** con la información presente en el contexto.
   - Incluir la cita en formato: `[Nombre del Documento, Página X]`.
   - Si la información no se encuentra en las políticas, indicar explícitamente que no está disponible y sugerir consultar a un operador humano.

### Implementación del Módulo de Indexación (`mesa_pedidos/indexar.py`)
El archivo implementa las tres etapas fundamentales:
1. `cargar_paginas()`: Recorre `datos/politicas/*.pdf` con `PyPDFLoader`, asignando a cada página su documento y número de página normalizado (base 1).
2. `partir_en_chunks()`:
   - Filtra encabezados ("FERRETERÍA INDUSTRIAL ANDINA...") y pies repetitivos.
   - Agrupa el contenido por artículo (`ART. \d+ - `), punto numerado (`\d+\. TITULO`) o SKU (`HE-4330`).
   - Concatena el título formal al inicio del chunk (`Condiciones Comerciales 2026 | ART. 4...`).
   - Aplica una red de seguridad con `RecursiveCharacterTextSplitter` solo si algún artículo supera los 1200 caracteres.
3. `construir_indice()`:
   - Calcula una huella digital (`_huella()`) con los hashes SHA-256 de los 4 PDFs y los parámetros de configuración.
   - Si el índice ya existe y la huella coincide, lo carga al instante desde disco (`FAISS.load_local`) sin consumir cuota de API.
   - Si cambió algo o no existe, genera los embeddings mediante Google Gemini (`models/text-embedding-004`) o OpenAI, crea el vectorstore y lo persiste en `artefactos/indice_faiss/` junto a `manifiesto.json`.

### 🔬 Metodología de Validación del Chunking (El Caso Crítico MSG-003)
¿Cómo sabe un Ingeniero de IA que el particionado no tiene fallos sin leer todos los fragmentos a mano?
1. **Suma de Control Estructural:**
   - 9 artículos en condiciones comerciales.
   - 40 fichas de productos (SKUs).
   - 5 puntos en logística.
   - 5 puntos en devoluciones y garantías.
   - **Total exacto esperado:** $9 + 40 + 5 + 5 = 59\text{ chunks}$. Ni un fragmento huérfano ni fusiones espurias.
2. **Prueba de Excepciones Legales (Edge Case MSG-003):**
   - El mensaje MSG-003 de la bandeja pregunta si un cliente puede devolver zapatos de seguridad por arrepentimiento.
   - Si se usa un splitter recursivo estándar por longitud fija (ej: 900 caracteres), la regla general (*"Se aceptan devoluciones dentro de los 15 días"*) cae en un chunk y la lista de exclusiones (*"NO se aceptan devoluciones de elementos de protección personal: guantes, zapatos de seguridad..."*) cae en el siguiente chunk. El LLM respondería afirmativamente con total elocuencia, cometiendo un error costoso.
   - **Verificación:** El chunk `[56]` (`politica_devoluciones_2026.pdf | Sección: 2. DEVOLUCION POR ARREPENTIMIENTO`) tiene 642 caracteres y reúne ambas partes inseparables en una sola ventana de contexto.
3. **Control de Densidad Semántica:**
   - Todos los chunks se ubican entre 150 y 808 caracteres, rango óptimo para evitar tanto la pérdida de contexto (chunks microscópicos) como la dilución de la señal vectorial (chunks kilométricos).

### 💡 Gestión de Cuotas en Google AI Studio (Free Tier vs. Paga)
Para chequear los límites en tiempo real de tu cuenta:
👉 [Panel de Rate Limits en Google AI Studio](https://aistudio.google.com/app/rate-limit)

1. **Embeddings:** El modelo activo es **`models/gemini-embedding-001`** (en dashboard: *Gemini Embedding 1 / 2*), con **100 RPM** y **1.000 RPD** en la capa gratuita. Cubre holgadamente los 59 chunks del índice en una sola corrida.
2. **LLM para Cuenta Paga:** `gemini-3.8-flash` (sin límites diarios).
3. **LLM para Cuenta Gratuita:** Seleccionar en `config.py` un modelo con alto cupo diario como `gemini-3.1-flash-lite` (500 RPD y 15 RPM). Evitar usar modelos con tope de 20 RPD (como el 3.8 en free tier) porque no alcanzan a procesar los 20 casos de prueba del Golden Set.
4. **Herramienta de diagnóstico:** Ejecutar `py -m mesa_pedidos.ver_modelos` para inspeccionar los modelos disponibles de cualquier API key.

### Implementación del Módulo de Recuperación y Citas (`mesa_pedidos/recuperacion.py`)
1. **Tokenizador BM25 con preservación de SKUs:**
   `re.findall(r"[a-z]{2}-\d{4}|[a-z0-9]+", normalizar_texto(texto))` asegura que códigos como `HE-4330` o `¿HE-4330?` se capturen como el token limpio `he-4330`.
2. **MMR (Maximal Marginal Relevance) en FAISS:**
   `fetch_k=20`, `k=4`, `lambda_mult=0.5`. Evita que el retriever traiga cuatro fragmentos redundantes que dicen lo mismo, diversificando la información para el LLM.
3. **EnsembleRetriever:**
   Combina BM25 (0.4) y FAISS denso (0.6) aplicando Reciprocal Rank Fusion (RRF).
4. **Citación Verificable:**
   El modelo devuelve el esquema Pydantic `RespuestaConsulta(respuesta=..., fuentes=[1, 2], encontrado=True)`. La función `citas_desde_fuentes()` toma esos enteros y busca los metadatos reales del documento (`documento`, `pagina`), garantizando cero alucinaciones de archivos o páginas inexistentes.

---

## 🏛️ 4. Matriz de Decisiones de Arquitectura (Consenso de Diseño)

A partir de la exploración del código y de las pruebas del sistema de referencia, dejamos formalmente asentadas las 6 decisiones críticas del proyecto:

| Dimensión | Decisión Acordada | Justificación Técnica |
|---|---|---|
| **Estrategia de Chunking** | Semántico por Artículos y SKUs (`_chunks_por_articulo`) | Evita la fragmentación arbitraria. Mantiene las excepciones y listas de exclusión (ej. calzado de seguridad en devoluciones) unidas a sus reglas principales. Genera 59 chunks estructurados. |
| **Generación de Citas** | Índices numéricos del LLM traducidos por Python | El LLM solo devuelve `fuentes: [1, 3]`; Python lee los metadatos reales del chunk (`documento`, `pagina`). Imposibilita alucinaciones de archivos o páginas inexistentes. |
| **Lógica Comercial** | Funciones puras determinísticas en `reglas.py` | La matemática, las escalas de descuento por volumen, los topes por marca (15% en herramientas eléctricas) y el cálculo de crédito no se delegan al LLM. Routers del grafo operan como pasamanos de estado. |
| **Diseño Human-in-the-Loop** | Evitar la doble ejecución de `interrupt()` | Como LangGraph reinicia el nodo desde la primera línea al reanudar el checkpoint, solo se ubica lectura del payload arriba de `interrupt()`. Las validaciones con Pydantic y posibles recálculos corren abajo. Si la decisión es inválida, se rechaza por defecto. |
| **Evaluación del Golden Set** | Registro previo de interrupción (`hubo_interrupt`) | Al evaluar el Golden Set, se registra si el grafo se frenó en `aprobacion_humana` *antes* de reanudar, garantizando que el sistema sea evaluado por saber cuándo pedir aprobación. |
| **Recuperación RAG** | Híbrida ponderada (0.4 BM25 / 0.6 Denso) | BM25 con tokenizador regex preserva códigos técnicos enteros (ej. `he-4330`) y normaliza tildes; el denso captura similitud semántica y paráfrasis. |

---

## 📈 Bitácora de Progreso
- [x] **Fase 0: Setup y Estructura:**
  - Estructura de carpetas `datos/`, `mesa_pedidos/`, `artefactos/` creada.
  - Archivos movidos a `datos/`.
  - Creación de `.venv` y preparación de dependencias.
- [x] **Fase 1 - Paso A (Indexación RAG completada):**
  - Módulo `mesa_pedidos/datos.py` con normalización de caracteres y carga de CSVs.
  - Módulo `mesa_pedidos/indexar.py` con chunking semántico por artículo/SKU.
  - 59 vectores generados e indexados con éxito en `artefactos/indice_faiss/`.
  - Script de diagnóstico `mesa_pedidos/ver_modelos.py` documentado.
- [ ] **Fase 1 - Paso B (Recuperación y Citas - Próximo paso):**
  - Retriever híbrido (BM25 + FAISS denso con MMR).
  - Generación de respuestas con LLM citando documento y página exactos.
- [ ] **Fase 2: Tools y Lógica Comercial en Python**
- [ ] **Fase 3: Estado, Esquemas Pydantic y Nodos**
- [ ] **Fase 4: Grafo y Routers Condicionales**
- [ ] **Fase 5: Human-in-the-Loop y Consola CLI**
- [ ] **Fase 6: Evaluación contra Golden Set**
