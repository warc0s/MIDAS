# Midas Plot

## 1. Descripción General

**MIDAS Plot** es el componente MIDAS que genera visualizaciones de datos a partir de un CSV y descripciones en lenguaje natural. Este sistema utiliza un flujo basado en CrewAI Flow para gestionar todo el proceso, el cual se compone de los pasos: 

1. Recolectar el CSV que sube el usuario.
2. El agente genera el codigo matplotlib a partir del prompt de usuario, la petición de su gráfica.
3. Ejecutar dicho codigo de forma segura en un entorno e2b, devolviendo el grafico en base64.
4. Renderizar dicho base64 para que aparezca la gráfica en el Streamlit, y pueda descargarse.

---

## 2. Arquitectura Técnica

### 2.1 Backend – `flow.py`

El backend se organiza mediante un **CrewAI Flow** que gestiona el proceso completo de generación y ejecución del código. Los componentes clave son:

- **Clase Principal: `FlowPlotV1`**
  - **Herencia:** Extiende de la clase `Flow` de CrewAI, permitiendo la definición de un flujo modular con pasos encadenados.
  - **Atributos:**
    - `api_input`: Entrada opcional desde API.
    - `_custom_state`: Diccionario que almacena información a lo largo del flujo (prompt, código generado, código limpio, etc.).
    - `model`: Modelo LLM (por ejemplo, `"gemini/gemini-2.0-flash"`) usado para la generación del código.

- **Pasos del Flujo:**
  1. **Inicio (`inicio`):**
     - Recibe el prompt y el contenido CSV.
     - Prepara el estado con la solicitud del usuario y datos adicionales (como el año actual).
     - Llama al modelo LLM usando LiteLLM (a través de `litellm.completion`) para generar el código Python (**raw_code**) basado en la descripción del usuario.
  2. **Limpieza de Código (`limpiar_codigo`):**
     - Elimina caracteres o backticks adicionales generados por el LLM, dejando el código listo para ejecución.
  3. **Ejecución del Código (`ejecutar_codigo`):**
     - Ejecuta el código limpio dentro de un entorno sandbox (usando `e2b_code_interpreter.Sandbox`).
     - Se escribe en el sandbox que el CSV sea utilizado en la ejecución.
     - Captura la salida estándar y extrae la imagen en formato base64 (se espera que sea la única salida impresa).

- **Funciones Auxiliares:**
  - **`_generate_plot_code`:** Construye el prompt para el LLM, especificando:
    - Uso obligatorio de matplotlib y pandas (si se requiere).
    - La necesidad de codificar la imagen como base64.
    - La impresión exclusiva del string base64 en la salida.
  - **`_extraer_base64`:** Analiza la salida del sandbox para identificar y extraer el string base64 correspondiente a la imagen (se asume que comienza con `iVBORw0KGgo`).

### 2.2 Frontend – `flow_gui.py`

- **Interfaz Web con Streamlit:**
  - Permite la carga y previsualización de archivos CSV.
  - Ofrece un área de entrada para prompts en lenguaje natural.
  - Muestra los resultados (visualizaciones) generados en formato de imagen (PNG codificado en base64).

---

## 3. Funcionalidades Clave

- **Generación Automática de Código Python:** Transforma descripciones en lenguaje natural en código para generar gráficos mediante matplotlib.
- **Ejecución Segura en Sandbox:** El código generado se ejecuta en un entorno aislado, previniendo riesgos de seguridad.
- **Soporte para Datos CSV:** Permite cargar y utilizar datasets en formato CSV, integrándolos en el proceso de visualización.
- **Manejo de Errores:** Implementa un sistema de validación y mensajes amigables para informar sobre posibles errores en la generación o ejecución del código.

---

## 4. Guía de Uso

1. **Carga de Datos:** El usuario puede cargar un archivo CSV para proveer datos al proceso de visualización.
2. **Descripción de la Visualización:** Se introduce un prompt en lenguaje natural describiendo el gráfico deseado.
3. **Generación y Ejecución del Código:** El sistema genera el código Python, lo sanitiza y lo ejecuta en el sandbox.
4. **Visualización e Iteración:** Se muestra el resultado (imagen en formato PNG codificada en base64) y se permite al usario descargar la imagen.

---

## 5. Referencias y Recursos

- **[CrewAI](https://www.crewai.com):** Framework utilizado para orquestar el flujo de generación y ejecución del código.
- **[Streamlit](https://streamlit.io):** Framework para la creación de la interfaz web interactiva.
- **[E2B Sandbox](https://e2b.dev):** Entorno de ejecución seguro para la ejecución del código generado.

---

## 6. Limitaciones Actuales

- **Dependencia de la Calidad del Prompt:** La precisión del resultado depende en gran medida de la claridad y calidad del prompt proporcionado por el usuario.
- **Formatos de Salida Limitados:** Actualmente, la salida se limita a imágenes en formato PNG codificadas en base64.
