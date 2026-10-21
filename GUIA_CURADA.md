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

---

## 📈 Bitácora de Progreso
- [x] **Fase 0: Setup y Estructura:**
  - Estructura de carpetas `datos/`, `mesa_pedidos/`, `artefactos/` creada.
  - Archivos movidos a `datos/`.
  - Creación de `.venv` y preparación de dependencias.
- [ ] **Fase 1: RAG y Búsqueda sobre Políticas (En progreso)**
- [ ] **Fase 2: Tools y Lógica Comercial en Python**
- [ ] **Fase 3: Estado, Esquemas Pydantic y Nodos**
- [ ] **Fase 4: Grafo y Routers Condicionales**
- [ ] **Fase 5: Human-in-the-Loop y Consola CLI**
- [ ] **Fase 6: Evaluación contra Golden Set**
