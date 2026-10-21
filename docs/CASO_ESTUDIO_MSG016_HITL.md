# 🔬 Caso de Estudio Técnico: Negociación HITL, Guardrail de Stock (Art. 9) y Ciclo de Reclamos (MSG-016)

**Mesa de Pedidos — Ferretería Industrial Andina S.R.L.**  
*Diplomatura en IA Engineering (ICARO · UNC · FCEFyN)*

---

## 📌 1. Resumen Ejecutivo

Este documento detalla el análisis de ingeniería, diagnóstico de fallas y resolución arquitectónica surgidos a partir del procesamiento del caso **MSG-016** (Luciana - Distribuidora Cuyo Insumos S.A.). 

El caso expone tres dimensiones críticas en el diseño de agentes comerciales con **Human-in-the-Loop (HITL)**:
1. La soberanía de Gerencia Comercial frente a topes generales vs. convenios de marca.
2. La obligación de implementar **guardrails defensivos en Python** para evitar facturación de mercadería fantasma ante quiebres de stock (Art. 9).
3. El enrutamiento y derivación automática de reclamos de postventa mediante tickets auditables.

---

## 💬 2. El Escenario Inicial: Mensaje MSG-016

El cliente envía el siguiente mensaje por WhatsApp:
> *"Pedido grande: 500 discos AB-1035, 300 discos AB-1120 y 60 amoladoras HE-4110. Necesito que me hagan 25% de descuento total, es una licitacion que ganamos y estamos muy justos."*

### A. Extracción Estructurada (LLM)
El nodo `extraer_pedido` estructura la intención:
* 500 x `AB-1035` (Disco de corte metal 180 mm).
* 300 x `AB-1120` (Disco de desbaste 115 mm).
* 60 x `HE-4110` (Amoladora angular Dewalt 115 mm).
* `descuento_solicitado`: 25.0%.

### B. Validación Determinística en Python (`validar_pedido`)
El motor de reglas cruza el pedido contra `catalogo.csv` y `clientes.csv` sin intervención del LLM y dispara **3 alertas comerciales**:
1. **Límite de crédito superado (Art. 7):** Saldo actual \$3.200.000 + Pedido \$18.731.000 = \$21.931.000 (Límite: \$5.500.000).
2. **Exceso de descuento general (Art. 5):** Solicita 25% (Tope general delegable: 20%).
3. **Quiebre de stock (Art. 9):** Solicita 60 amoladoras `HE-4110`, pero en depósito solo existen **52 unidades**.

**Veredicto del Router:** Se frena el flujo con `interrupt()` y se desvía a `aprobacion_humana`.

---

## ⚠️ 3. Diagnóstico del Bug: Facturación de Mercadería Inexistente

### La Primera Corrida (Falla Silenciosa)
Durante la reanudación interactiva en `correr.py`:
```text
Decision ([A]probar / [E]ditar / [R]echazar): E
Cantidades nuevas (SKU=cant; vacio = igual): <Enter>
Descuento autorizado en %: 22
Razon o justificacion: cliente importante
```

Al dar **Enter** en cantidades nuevas, el sistema original ejecutó:
```python
nueva_cantidad = decision.ediciones.get(r["sku"], r["cantidad"])
```
Como no se ingresó una edición manual de cantidades, el software mantuvo ciegamente las **60 unidades** de amoladoras, emitiendo una confirmación por:
* Neto mercadería: **\$18.631.200,00**
* Total c/IVA: **\$22.543.752,00**
* Renglón amoladora: `HE-4110 x 60 u. | Stock: NO (stock 52)`

### Debate de Ingeniería: ¿Error de Negocio o Bug de Software?
Un sistema de IA Engineering robusto **no puede depender de que el operador humano recuerde tipear a mano `HE-4110=52`**:
* El sistema **ya sabía** que el stock físico era 52 (lo había detectado en `validar_pedido`).
* Facturar 60 unidades genera un faltante de remito ilegal y fraudulento según la ley comercial argentina, obligando a emitir Notas de Crédito y frenando la cobranza.
* **Conclusión:** Permitir que una orden se confirme con cantidades superiores al stock disponible en entrega inmediata constituye una **falla de diseño por ausencia de guardrails defensivos**.

---

## 🛠️ 4. La Solución Implementada: Guardrail Automático del Artículo 9

Se modificó [`mesa_pedidos/nodos.py`](file:///c:/Proyectos/Icaro/Mesa%20de%20Pedidos/mesa_pedidos/nodos.py) en `_aplicar_edicion_humana` y `aprobacion_humana`:

```python
# Guardrail Art. 9: jamás facturar por encima del stock disponible en entrega inmediata.
# Si el operador no ingresó una cantidad manual explícita y la solicitada supera el stock,
# se factura lo disponible físicamente y la diferencia queda como entrega parcial pendiente.
if r["sku"] not in decision.ediciones and cant_solicitada > r["stock_disponible"]:
    nueva_cantidad = r["stock_disponible"]
    pendientes = cant_solicitada - nueva_cantidad
    faltantes_pendientes.append(
        f"Queda pendiente por falta de stock: {pendientes} x {r['sku']} {r['nombre']}"
    )
```

### Comportamiento Resultante:
1. **Facturación sobre lo Real:** La amoladora se clava automáticamente en **52 unidades**.
2. **Coherencia Matemática:**
   - 500 x `AB-1035` al 22%: **\$2.301.000,00**
   - 300 x `AB-1120` al 22%: **\$1.591.200,00**
   - 52 x `HE-4110` al 15% (tope Art. 5): **\$12.773.800,00**
   - **Neto Facturado:** **\$16.666.000,00** *(diferencia exacta de \$1.965.200 no cobrados indebidamente)*.
   - **IVA (21%):** **\$3.499.860,00**
   - **Total:** **\$20.165.860,00** *(o \$21.249.150,00 con IIBB San Juan)*.
3. **Registro de Pendientes:** Se asienta formalmente en las observaciones:  
   `"Queda pendiente por falta de stock: 8 x HE-4110 Amoladora angular 115 mm 900 W"`.
4. **Notificación al Cliente:** El prompt de `confirmar_pedido` recibe el bloque de pendientes y le aclara al cliente en el correo que las 8 unidades se despacharán ni bien ingrese stock.

### B. Evidencia de Ejecución en Consola (Validación Real del Guardrail)

A continuación se transcribe la salida real de la terminal al reanudar la ejecución con la decisión de **`APROBAR`** (`"es la jefa"`):

```text
Decision ([A]probar / [E]ditar / [R]echazar): a
Razon o justificacion (queda en auditoria): es la jefa  

Reanudando ejecucion con decision: APROBAR...
-> aprobacion_humana      Entrega parcial por stock (Art. 9) - Nuevo neto: $16.765.800,00
-> confirmar_pedido       Guardado en MSG-016.json

================================================================================
MENSAJE: MSG-016   |   RUTA FINAL: confirmar_pedido
================================================================================

Items valorizados:
SKU      Cant.  Precio Lista  Desc.  Subtotal Neto   Stock
-------  -----  ------------  -----  --------------  -----
AB-1035  500    $5.900,00     20%    $2.360.000,00   SI
AB-1120  300    $6.800,00     20%    $1.632.000,00   SI
HE-4110  52     $289.000,00   15%    $12.773.800,00  SI

Neto: $16.765.800,00  |  IVA: $3.520.818,00  |  Flete: $0,00  |  TOTAL: $20.286.618,00

Alertas comerciales:
   ! Saldo actual $3.200.000,00 + pedido $18.731.000,00 = $21.931.000,00 supera el límite de crédito de $5.500.000,00 (Art. 7).
   ! Pide 25% de descuento: supera el tope máximo de 20% (Art. 5)
   ! Sin stock suficiente de HE-4110: solicitan 60 u., hay disponibles 52 u. (Art. 9)

Observaciones:
   · Descuento habitual 12% + volumen 12% (Art. 4) = 24%
   · Topeado al 20% general (Art. 5)
   · Zona 3 (San Juan): $95.000,00, 4 a 6 días hábiles, Cruz del Sur; Flete bonificado al 100% por monto (Punto 3)
   · Queda pendiente por falta de stock: 8 x HE-4110 Amoladora angular 115 mm 900 W
   · Pedido ajustado por Gerencia Comercial previo a la confirmación

Respuesta al cliente:
   Estimados de Distribuidora Cuyo Insumos S.A., confirmamos el ingreso de su pedido PED-2026-1660.
   Detalle confirmado: 500 discos AB-1035 (desc. 20%), 300 discos AB-1120 (desc. 20%) y 52 amoladoras HE-4110 (desc. 15% por tope de marca). Quedan 8 amoladoras pendientes por falta de stock.
   - Neto de mercadería: $16.765.800,00
   - Descuento comercial aplicado: 20%
   - Flete: Bonificado al 100%
   - Total con IVA: $20.286.618,00
   - Plazo estimado de entrega: 4 a 6 días hábiles.
   Saludos cordiales.

Log de auditoria (reducer acumulativo por nodo):
   Mensaje MSG-016 ingresado por canal whatsapp
   19:51:39 | cargar_cliente     | Distribuidora Cuyo Insumos S.A. | Cat. A | Estado: activo | Mora: 0 d | Crédito disp.: $2.300.000,00
   19:51:49 | clasificar         | pedido (confianza 0.95) - El cliente solicita mercadería detallando códigos de producto específicos y cantidades concretas, adjuntando una solicitud de condición comercial por volumen.
   19:51:53 | extraer_pedido     | 3 ítems extraídos (AB-1035 x500, AB-1120 x300, HE-4110 x60) | solicita 25% desc.
   19:51:53 | validar_pedido     | Neto: $18.731.000,00 | Desc: 20% | 3 alertas comerciales -> Sugiere: aprobacion_humana
   19:52:26 | aprobacion_humana  | APROBAR por Gerencia Comercial - Motivo: es la jefa | Entrega parcial por stock (Art. 9) - Nuevo neto: $16.765.800,00
   19:52:39 | confirmar_pedido   | PED-2026-1660 emitido | Total: $20.286.618,00 | Autorizado por Gerencia Comercial | Guardado en MSG-016.json
```

---

## 🔄 5. Simulación de Postventa: El Reclamo del Cliente

Se simuló la reacción de un cliente disconforme ante la entrega parcial:
```powershell
py correr.py --cuit 30-71340098-6 --texto "Buen dia, el pedido llego incompleto, me faltan 8 amoladoras y no me avisaron nada. Exijo una solucion urgente o cancelo la cuenta."
```

### Evidencia de Ejecución en Consola:
```text
Procesando LIBRE-1021-194743 [WEB · Consola CLI]:
"Buen dia, el pedido llego incompleto, me faltan 8 amoladoras y no me avisaron nada. Exijo una solucion urgente o cancelo la cuenta...."

-> cargar_cliente         Crédito disp.: $2.300.000,00
-> clasificar             derivar (confianza 0.98) - El cliente realiza un rec
-> derivar_humano         Ticket TCK-2026-7738 -> Reclamos y Logística (origen

================================================================================
MENSAJE: LIBRE-1021-194743   |   RUTA FINAL: derivar_humano
================================================================================

Log de auditoria (reducer acumulativo por nodo):
   Mensaje LIBRE-1021-194743 ingresado por canal web
   19:47:43 | cargar_cliente     | Distribuidora Cuyo Insumos S.A. | Cat. A | Estado: activo | Mora: 0 d | Crédito disp.: $2.300.000,00
   19:47:47 | clasificar         | derivar (confianza 0.98) - El cliente realiza un reclamo formal por un faltante físico de mercadería (8 amoladoras) en una entrega ya realizada y exige una solución urgente.
   19:47:47 | derivar_humano     | Ticket TCK-2026-7738 -> Reclamos y Logística (origen: clasificacion)
```

### Corrección Adicional: Bug de Saludo ("Hola Consola")
* **Problema:** En mensajes libres por terminal, el sistema tomaba la primera palabra de `remitente="Consola CLI"` y redactaba: *"Hola Consola, gracias por comunicarte..."*.
* **Solución:** Se corrigió en `derivar_humano` de [`mesa_pedidos/nodos.py`](file:///c:/Proyectos/Icaro/Mesa%20de%20Pedidos/mesa_pedidos/nodos.py) para priorizar la Razón Social de la empresa (`cliente.get("razon_social")`), respondiendo:
  > *"Hola Distribuidora Cuyo Insumos S.A., gracias por comunicarte con Ferretería Industrial Andina S.R.L.. Tu mensaje fue derivado al sector de Reclamos y Logística..."*

---

## 📊 6. Matriz Comparativa de Resultados

| Dimensión | Repo de Referencia | Nuestro Sistema (Inicial) | Nuestro Sistema (Final con Guardrail) |
|---|---|---|---|
| **Descuento en Discos** | 20.0% *(bug Art. 5)* | 22.0% *(soberanía Gerencia)* | **22.0%** *(soberanía Gerencia)* |
| **Descuento Dewalt** | 15.0% *(tope marca)* | 15.0% *(tope marca)* | **15.0%** *(tope marca)* |
| **Cant. Amoladoras Facturadas** | 60 u. | 60 u. | **52 u. (entrega parcial física)** |
| **Unidades Pendientes Registradas** | Ninguna | Ninguna | **8 u. (Art. 9 backorder)** |
| **Neto Facturado** | \$18.731.000,00 | \$18.631.200,00 | **\$16.666.000,00 (exacto)** |
| **Contradicción Factura vs Mail** | Sí (la carta decía 25%, cobraba 20%) | Parcial (cobró 60 amoladoras sin stock) | **Cero contradicción** |
