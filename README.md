<div align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/logo1.png" alt="MIDAS Logo" width="50%">
  <h1>MIDAS - Multi-agent Intelligent Data Automation System 🤖</h1>
  <p><em>Convierte tus ideas en modelos ML listos para producción</em></p>
</div>

<!-- Enlaces Rápidos -->
<div align="center">
  <div class="quick-links">
    <div class="quick-link">
      <a href="https://github.com/warc0s/MIDAS" target="_blank">
        <img src="https://img.shields.io/badge/📦_REPOSITORIO-GitHub-181717?style=for-the-badge&labelColor=111111&logo=github&logoColor=white" alt="Repositorio"/>
      </a>
    </div>
    <div class="quick-link">
      <a href="Extra/Documentacion/docs/index.md" target="_blank">
        <img src="https://img.shields.io/badge/📚_DOCUMENTACIÓN-Local-22A699?style=for-the-badge&labelColor=15756C&logo=markdown&logoColor=white" alt="Documentación (local)"/>
      </a>
    </div>
    <div class="quick-link">
      <a href="Extra/Webs/MIDASTFM-Triptico-Final.pdf" target="_blank">
        <img src="https://img.shields.io/badge/🎯_PRESENTACIÓN-PDF-FF9E00?style=for-the-badge&labelColor=D97F00&logo=googledrive&logoColor=white" alt="Presentación (PDF)"/>
      </a>
    </div>
    <div class="quick-link">
      <a href="https://youtu.be/G5KMC8kFZEY" target="_blank">
        <img src="https://img.shields.io/badge/🎥_VIDEO_EXPLICATIVO-Video Youtube-FF5757?style=for-the-badge&labelColor=D63030&logo=youtube&logoColor=white" alt="Video Explicativo"/>
      </a>
    </div>
  </div>
</div>

## 📑 Índice

0. [Visión General](#-visión-general)
1. [Justificación y Descripción del Proyecto](#sección-1-justificación-y-descripción-del-proyecto)
2. [Obtención de Datos](#sección-2-obtención-de-datos)
3. [Limpieza de Datos](#sección-3-limpieza-de-datos)
4. [Exploración y Visualización de Datos](#sección-4-exploración-y-visualización-de-los-datos)
5. [Preparación de Datos para ML](#sección-5-preparación-de-los-datos-para-los-algoritmos-de-machine-learning)
6. [Entrenamiento y Evaluación de Modelos](#sección-6-entrenamiento-del-modelo-y-comprobación-del-rendimiento)
7. [Procesamiento de Lenguaje Natural](#sección-7-procesamiento-de-lenguaje-natural)
8. [Aplicación Web](#sección-8-aplicación-web)
9. [Conclusiones](#sección-9-conclusiones)
10. [Creadores](#-creadores)

## 🌟 Visión General

![Midas Main Website](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Main.png?raw=true)

MIDAS es un proyecto de Trabajo Final de Máster (TFM) que propone un sistema innovador para automatizar el desarrollo de modelos de machine learning. A través de una arquitectura multiagente compuesta por 8 componentes especializados, MIDAS aborda los desafíos técnicos y las barreras de entrada que existen en el campo de la ciencia de datos. El sistema cubre todo el ciclo de desarrollo ML: desde la generación de datos y visualizaciones, pasando por el entrenamiento y validación de modelos, hasta su despliegue final, permitiendo que profesionales con diversos niveles de experiencia puedan crear e implementar soluciones ML efectivas de manera más ágil y accesible.

---

## Sección 1: Justificación y descripción del proyecto

MIDAS es un sistema multiagente multicomponente que automatiza integralmente el proceso de desarrollo de modelos de machine learning, desde la creación o ingesta de datos, hasta su despliegue en producción. El proyecto nace para resolver un problema crítico en la industria: el desarrollo de modelos de ML, el cual tradicionalmente requiere múltiples herramientas, conocimientos especializados y procesos manuales que consumen tiempo y recursos. Inspirado en la leyenda del Rey Midas, nuestro sistema actúa como un "toque dorado" moderno que transforma datos o ideas sin procesar en soluciones de ML listas para usar.

### ✨ Fundamentos del Proyecto

La necesidad de MIDAS se fundamenta en tres pilares principales:

- 🔍 La creciente demanda de automatización en procesos de ML.
- 🔗 La escasez de soluciones integrales que cubran todo el pipeline de datos.
- 🚪 La importancia de hacer accesible el ML a usuarios con diferentes niveles de experiencia técnica.

### 🏗️ Arquitectura Modular

El sistema implementa una arquitectura modular innovadora a través de 8 componentes especializados:

| Componente | Descripción |
|------------|-------------|
| **🔄 Midas Dataset** | Genera conjuntos de datos sintéticos personalizados según las especificaciones del usuario en términos de temática, dimensiones y características. |
| **📊 Midas Plot** | Genera gráficos a partir de un dataset proporcionado por el usuario, interpretando solicitudes en lenguaje natural. |
| **✨ Midas Touch** | Ejecuta la limpieza, entrenamiento y optimización de modelos, automatizando las tareas más complejas del proceso. |
| **🧪 Midas Test** | Implementa validación rigurosa y métricas de rendimiento, asegurando la calidad del modelo obtenido. |
| **🚀 Midas Deploy** | Facilita el despliegue mediante interfaces web automatizadas para predicciones. |
| **🗣️ Midas Assistant** | Interfaz central que guía al usuario en la utilización efectiva de cada componente. |
| **🏗️ Midas Architect** | Guía el diseño del sistema multiagente. |
| **❓ Midas Help** | Proporciona soporte técnico contextual de nuestro TFM basado en RAG. |

![Midas Diagrama](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Diagrama_ArquitecturaModular.png?raw=true)

Profesional, escalable y orientado a resultados, MIDAS redefine la automatización en proyectos de Machine Learning. Su arquitectura modular, donde cada componente está estratégicamente diseñado y optimizado, establece un nuevo paradigma en el desarrollo de modelos ML. El sistema demuestra que la verdadera "transformación en oro" va más allá de convertir datos en modelos precisos - consiste en hacer accesible todo el proceso de ML a través de interacciones naturales e intuitivas, democratizando así el desarrollo de modelos para equipos de cualquier tamaño y experiencia.

---

## Sección 2: Obtención de datos

MIDAS implementa múltiples estrategias de obtención de datos, alineadas con las diferentes necesidades que pueden surgir a lo largo del ciclo de vida de un proyecto de machine learning:

### 2.1 Generación sintética mediante Midas Dataset 🧬

![Midas Dataset](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Dataset_2_1.png?raw=true)

**El componente Midas Dataset** constituye una herramienta efectiva para la generación automatizada de conjuntos de datos sintéticos. Esta funcionalidad es fundamental en escenarios donde:

> 🔹 Se necesitan datos para pruebas de concepto sin exponer información sensible  
> 🔹 Se requiere crear datasets de prueba con datos realistas  
> 🔹 Se desea generar información estructurada para desarrollo y testing

**Mecanismo de funcionamiento:** Midas Dataset implementa un sistema multi-agente basado en AG2 que coordina tres agentes especializados:

- **Input Agent:** Recibe y procesa las solicitudes iniciales del usuario
- **Validation Agent:** Verifica que los parámetros proporcionados sean válidos
- **Column Classifier Agent:** Clasifica automáticamente los nombres de columnas para mapearlos a tipos de datos apropiados

El sistema utiliza la biblioteca Faker para generar datos realistas en español (es_ES), con soporte para diversas categorías de información:

- Datos personales (nombres, apellidos, edad)
- Información de contacto (correo, teléfono)
- Direcciones (calle, ciudad, país)
- Datos financieros (precios, porcentajes)
- Identificadores únicos (IDs, códigos)
- Y muchos más tipos predefinidos

El proceso de generación es **simple pero potente**:
1. El usuario especifica el número de registros y los nombres de columnas
2. El sistema detecta automáticamente los tipos de datos adecuados basándose en los nombres
3. Para columnas numéricas, se pueden definir valores mínimos y máximos
4. Se genera el dataset completo que puede ser modificado posteriormente

<p align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Dataset_2_1_2.png?raw=true" alt="Midas Dataset Resultado" width="500">
</p>

### 2.2 Carga directa desde fuentes externas 📂

Además, **los componentes Midas Touch y Midas Plot** permiten a los usuarios cargar sus propios conjuntos de datos en múltiples formatos:

| Formato | Compatibilidad | Características |
|---------|----------------|----------------|
| **CSV** | Midas Touch & Plot | Formato principal, universalmente soportado |
| **XLSX** | Midas Touch | Facilita integración con herramientas empresariales |
| **Parquet** | Midas Touch | Formato columnar optimizado para análisis |
| **JSON** | Midas Touch | Para estructuras de datos más complejas |

Los datos son subidos a través de interfaces intuitivas implementadas en Streamlit, que permiten la previsualización inmediata y validación básica antes del procesamiento. De igual forma, recomendamos usar siempre CSV.

### 2.3 Adquisición de conocimiento para módulos concretos 🧠

Por último, **los componentes Midas Help y Midas Architect** implementan sistemas de Recuperación Aumentada Generativa (RAG) para proporcionar asistencia contextualizada. Para estos módulos hemos obtenido sus datos mediante:

- **Web crawling:** En Midas Architech, para obtener la documentación de cada framework. Usamos <a href="https://github.com/unclecode/crawl4ai" target="_blank" rel="noopener noreferrer">crawl4ai</a>
 para extraer documentación técnica en formato Markdown
- **Fine-tuning en el Bert:** Midas Help incorpora un modelo BERT específicamente afinado para clasificar las consultas de los usuarios. El dataset de este Bert fue obtenido de forma sintética, puedes verlo en: <a href="https://github.com/warc0s/MIDAS/blob/main/7midas_help/Cuadernos_PredecirDificultad/Bert_Spanish__Predecir_Dificultad_Help.ipynb" target="_blank" rel="noopener noreferrer">
    BERT Spanish - Predecir Dificultad Prompt
</a>
- **Midas Help:** La documentación en la que se basa (RAG) para responder está extraida de este repositorio. Este readme y la carpeta "Documentación", dentro de "Extras".

---

## Sección 3: Limpieza de datos
La limpieza y preparación de datos constituye una fase crítica en cualquier proyecto de machine learning. **El componente Midas Touch** aborda este reto a través de un enfoque automatizado y adaptativo.

### 3.1 Procesamiento adaptativo según tipo de problema 🔄
**El agente DataAlchemistAgent de Midas Touch** implementa un pipeline inteligente de limpieza que se adapta automáticamente al tipo de problema detectado:
- ✅ **Detección automática del objetivo:** El sistema extrae la columna objetivo directamente del prompt del usuario
- ✅ **Identificación del tipo de problema:** Determina si se trata de clasificación o regresión mediante análisis semántico de la descripción
- ✅ **Ajuste dinámico de estrategias:** Aplica diferentes enfoques de preprocesamiento según el tipo de datos (numéricos o categóricos)

### 3.2 Tratamiento de valores nulos 🧩
**Midas Touch** implementa estrategias específicas para la gestión de valores **faltantes**:
<table>
  <tr>
    <th>Tipo de Variable</th>
    <th>Estrategia de Imputación</th>
  </tr>
  <tr>
    <td><strong>Numéricas</strong></td>
    <td>Imputación con la mediana</td>
  </tr>
  <tr>
    <td><strong>Categóricas</strong></td>
    <td>Imputación con la moda (valor más frecuente)</td>
  </tr>
  <tr>
    <td><strong>Columnas con alta tasa de valores faltantes</strong></td>
    <td>Eliminación de columnas con más del 70% de valores faltantes</td>
  </tr>
</table>

El sistema documenta el proceso de preprocesamiento en el notebook generado, incluyendo las transformaciones aplicadas a cada tipo de variable. Concretamente, se vería así tomando como dataset el famoso del Titanic de Kaggle:
<p align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Preprocesamiento_3_2.png?raw=true" alt="Midas Touch Preprocesamiento" width="500">
</p>

<sub><i>Nota: Aquí se puede ver una de las debilidades de Midas Touch, aplica one-hot encoding a las columnas categóricas a pesar de tener alta dimensionalidad.</i></sub>

### 3.3 Detección y procesamiento de fechas 📅
**El DataAlchemistAgent** incluye capacidades específicas para el manejo de columnas temporales:

- 🔍 **Detección automática**: Identifica columnas que parecen contener fechas mediante **expresiones regulares**
- 🔍 **Extracción de componentes**: Convierte fechas detectadas en características útiles como año, mes, día y día de la semana
- 🔍 **Transformación estructurada**: Reemplaza las fechas originales con componentes numéricos que pueden ser utilizados por los modelos

Este procesamiento permite que la información temporal sea aprovechada efectivamente por los algoritmos de machine learning, que típicamente requieren entradas numéricas.

### 3.4 Validación de calidad de datos ✓
**Midas Touch** también incluye validaciones básicas para garantizar la viabilidad del análisis:

- 📋 **Validación de la columna objetivo**: Verifica que exista, no tenga demasiados valores faltantes y contenga suficiente variabilidad
- 📋 **Detección de columnas problemáticas**: Identifica y elimina columnas con más del 70% de valores faltantes
- 📋 **Verificación de tamaño mínimo**: Comprueba que el dataset tenga suficientes filas para el entrenamiento
- 📋 **Alerta sobre columnas de alta cardinalidad**: Detecta variables categóricas con gran número de valores únicos. Solo alerta.

Estas verificaciones se registran en el log del sistema y se documentan en el notebook generado, permitiendo entender las decisiones tomadas durante el preprocesamiento.

### 3.5 Descripción detallada del proceso 📝

Cada conjunto de datos procesado por **Midas Touch** es documentado automáticamente por el agente **NotebookScribeAgent**, generando:

| Tipo de documentación | Descripción |
|----------------------|-------------|
| 📊 **Resumen del dataset** | Información sobre dimensiones y estructura de los datos |
| 📈 **Estadísticas descriptivas** | Tipos de datos, valores faltantes y valores únicos |
| 🔍 **Análisis de columnas** | Información básica sobre cada columna del dataset |
| 🔄 **Mapeo de transformaciones** | Documentación de los cambios aplicados durante el preprocesamiento |

Esta documentación se integra en el notebook generado, facilitando la comprensión y trazabilidad del proceso completo.

### 3.6 Resiliencia ante fallos 🛡️

**El agente PhoenixAgent de Midas Touch** está específicamente diseñado para gestionar situaciones excepcionales durante el procesamiento:

- 🚨 **Respuesta a errores:** Actúa cuando otros agentes reportan fallos durante el proceso
- 🔄 **Estrategias adaptativas específicas:** Implementa soluciones según el tipo de error:
  - Para errores en DataGuardianAgent: Selección de columna alternativa (última columna)
  - Para errores en DataAlchemist: Simplificación del preprocesamiento
  - Para errores en ModelShaman: Utilización de modelos fallback más simples
- 📋 **Registro de recuperación:** Documenta las acciones tomadas para recuperar el workflow

Esta arquitectura garantiza que el proceso sea robusto incluso ante datasets particularmente desafiantes o errores inesperados.

---

## Sección 4: Exploración y visualización de los datos

La exploración y visualización de datos constituye una fase fundamental para comprender patrones, correlaciones y características inherentes en los conjuntos de datos. **El componente Midas Plot** potencia este proceso revolucionando la forma en que se generan visualizaciones.

### 4.1 Generación de visualizaciones mediante lenguaje natural 💬

**Midas Plot** implementa un enfoque innovador que permite a los usuarios solicitar visualizaciones complejas utilizando simplemente lenguaje natural:

- 🔤 **Interpretación semántica:** Transforma descripciones textuales en una gráfica real, en segundos
- 🔄 **Flexibilidad expresiva:** Permite especificar desde simples histogramas hasta gráficos complejos multivariados
- 🚀 **Abstracción de complejidad técnica:** Elimina la necesidad de conocer detalles de implementación en Python

Este enfoque democratiza la creación de visualizaciones, haciéndolas accesibles tanto a cientificos de datos experimentados como a analistas de negocio con conocimientos técnicos limitados.

![Midas Plot](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Plot_4_1.png?raw=true)

### 4.2 Arquitectura Midas PLot ⚙️

**Midas Plot** emplea una arquitectura de flujo secuencial basada en CrewAI Flow que comprende cuatro pasos fundamentales:

1. **Inicio:** Recibe el prompt del usuario y el CSV, preparando el estado inicial
2. **Generación de código:** Invoca el modelo LLM para traducir la petición en código matplotlib
3. **Limpieza de código:** Sanitiza el código generado eliminando artefactos o errores comunes
4. **Ejecución segura:** Utiliza un entorno sandbox e2b para ejecutar el código sin riesgos

Esta arquitectura garantiza tanto la flexibilidad como la seguridad del proceso de visualización.

### 4.3 Tipos de visualizaciones soportadas 📊

**Midas Plot** es capaz de generar una amplia variedad de representaciones visuales:

| Categoría | Tipos de Gráficos | Ejemplos |
|-----------|-------------------|----------|
| **Univariantes** | Distribuciones, conteos | Histogramas, gráficos de densidad, diagramas de caja |
| **Bivariantes** | Relaciones entre dos variables | Gráficos de dispersión, mapas de calor, gráficos de barras agrupadas |
| **Multivariantes** | Patrones complejos | Matrices de correlación, gráficos de coordenadas paralelas |
| **Temporales** | Evolución cronológica | Series temporales, descomposiciones estacionales |
| **Categóricas** | Relaciones entre categorías | Diagramas de Sankey, gráficos de radar, diagramas aluviales |

Básicamente, cualquier gráfica que matplotlib soporte, Midas Plot lo soporta.
Además, el sistema optimiza automáticamente aspectos como paletas de colores, escalas, leyendas y anotaciones para maximizar la legibilidad y el impacto visual.

### 4.4 Integración en el flujo de trabajo 🔄

Las visualizaciones generadas por **Midas Plot** se integran perfectamente en el flujo de trabajo más amplio de MIDAS:

- 📥 **Exportación en formato PNG:** Permite incorporar las visualizaciones en informes o presentaciones
- 📓 **Integración con notebooks:** Una vez generada tu gráfica, puedes añadirla a cualquier cuaderno jupyter para completarlo
- 🔄 **Retroalimentación para modelos:** Proporciona información visual sobre tu dataset, para así comprenderlo mejor y decidir el siguiente paso en tu entrenamiento del modelo

Esta integración asegura que las visualizaciones no sean un fin en sí mismas, sino herramientas valiosas para mejorar la comprensión de los datos y la calidad de los modelos resultantes.

---

## Sección 5: Preparación de los datos para los algoritmos de Machine Learning

La preparación adecuada de los datos constituye un elemento crítico para el éxito de cualquier algoritmo de machine learning. **El componente Midas Touch** aborda esta fase a través de procesos automatizados e inteligentes implementados principalmente en sus agentes especializados.

### 5.1 Ingeniería de características adaptativa 🛠️

**El DataAlchemistAgent de Midas Touch** implementa estrategias básicas de ingeniería de características que se adaptan al tipo de datos, como ya explicamos en el punto 3:

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #4caf50;">
<b>Características Implementadas:</b><br>
- Extracción de componentes temporales de fechas (año, mes, día, día de semana)<br>
- Detección automática de columnas con formato de fecha<br>
- Eliminación de columnas con alta tasa de valores faltantes (>70%)<br>
- Construcción de pipeline de transformación con sklearn
</div>

### 5.2 Normalización y escalado de datos 📏

**Midas Touch** implementa técnicas específicas de normalización según el tipo de datos:

| Tipo de Variable | Transformación Aplicada | Implementación |
|---------|-----------------|----------------|
| **Variables numéricas** | Estandarización (Z-score) | `sklearn.preprocessing.StandardScaler` |
| **Variables categóricas** | Codificación one-hot | `sklearn.preprocessing.OneHotEncoder` |
| **Valores faltantes numéricos** | Imputación con mediana | `sklearn.impute.SimpleImputer(strategy='median')` |
| **Valores faltantes categóricos** | Imputación con valor más frecuente | `sklearn.impute.SimpleImputer(strategy='most_frequent')` |

Estas transformaciones se aplican automáticamente dentro de un pipeline de scikit-learn, que maneja adecuadamente los diferentes tipos de columnas presentes en el dataset.

### 5.3 Implementación de pipelines de transformación 🔄
**El DataAlchemistAgent** construye pipelines estructurados utilizando la API Pipeline de scikit-learn, proporcionando:
- ✅ **Reproducibilidad:** Las transformaciones se aplican consistentemente a los datos
- 🔄 **Preprocesamiento modular:** Separación de transformaciones para columnas numéricas y categóricas
- 📝 **Documentación detallada:** Los pasos del pipeline quedan documentados en el notebook generado

### 5.4 Manejo de diferentes tipos de columnas
**El DataAlchemistAgent** identifica y procesa diferentes tipos de datos:
- 🔢 **Variables numéricas:** Detectadas automáticamente y procesadas con escalado apropiado
- 🔤 **Variables categóricas:** Codificadas mediante one-hot encoding
- 📅 **Variables de fecha:** Detectadas por patrones y convertidas en componentes temporales útiles
- ⚠️ **Columnas problemáticas:** Identificación de columnas con alta proporción de valores faltantes

Esto permite que el sistema funcione con una amplia variedad de datasets sin requerir preprocesamiento manual previo.

### 5.5 Estrategias de validación 🧩
**Midas Touch** implementa técnicas específicas para la división y validación de datos:
- 📊 **Estratificación en división de datos:** Para problemas de clasificación, preserva la distribución de clases en los conjuntos de entrenamiento y prueba (cuando hay suficientes ejemplos de cada clase)
- 🔄 **Validación cruzada (5-fold):** Evalúa la robustez del modelo mediante validación cruzada con 5 particiones
- 🛡️ **Prevención de fugas de datos:** División explícita de conjuntos de entrenamiento y prueba antes de la evaluación del modelo

El sistema adapta sus estrategias de validación según el tipo de problema (clasificación/regresión) y las características del dataset.

### 5.6 Implementación técnica a través de agentes especializados 🤖
El proceso de preparación de datos se implementa a través de dos agentes clave de **Midas Touch**:
- **DataGuardianAgent:** Identifica la columna objetivo mencionada en el prompt y analiza sus características estadísticas
- **DataAlchemistAgent:** Ejecuta las transformaciones específicas y construye los pipelines de preprocesamiento

El proceso completo queda documentado en el notebook generado automáticamente por el **NotebookScribeAgent**, incluyendo:
- Código para cada transformación aplicada
- Explicaciones en formato markdown de cada decisión tomada
- Visualizaciones de resumen de los datos antes y después del preprocesamiento
- Información sobre el impacto de las transformaciones en la estructura del dataset

---

## Sección 6: Entrenamiento del modelo y comprobación del rendimiento

El entrenamiento de modelos y la evaluación exhaustiva de su rendimiento constituyen fases determinantes para garantizar la efectividad de las soluciones de machine learning. **MIDAS** implementa un enfoque integral a través de los componentes **Midas Touch** y **Midas Test**. Concretamente, **Midas Touch** se vería así:

![Midas Touch Interfaz](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Interfaz_6_0.png?raw=true)

### 6.1 Selección inteligente de algoritmos 🧠

**El agente ModelShamanAgent de Midas Touch** implementa un sistema de selección automática de algoritmos basado en criterios específicos:

Criterios de Selección:
- Tipo de problema (clasificación o regresión)<br>
- Tamaño del dataset (número de muestras)<br>
- Complejidad de las características (número de variables)

| Criterio | Algoritmo Seleccionado |
|------------------|--------------------------|
| **Datasets pequeños (<1000 muestras) o con muchas características (>50)** | RandomForest (Classifier/Regressor) |
| **Datasets más grandes con pocas características** | GradientBoosting (Classifier/Regressor) |
| **Casos de fallback (tras errores)** | DummyClassifier/DummyRegressor |

El sistema selecciona automáticamente entre estos algoritmos de scikit-learn según las características del dataset, y en caso de fallos repetidos durante el entrenamiento, utiliza modelos baseline como mecanismo de recuperación.

### 6.2 Evaluación mediante agentes especializados 📊

<p align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Test_Interfaz_6_2.png?raw=true" alt="Midas Test Interfaz" width="500">
</p>

**El componente Midas Test** coordina un análisis colaborativo mediante múltiples agentes especializados basados en LLM:

**Arquitectura de agentes:**
- **Model Analyzer**: Examina estructura y características generales del modelo
- **Performance Tester**: Analiza rendimiento computacional y uso de recursos
- **Robustness Checker**: Evalúa comportamiento ante datos anómalos
- **Output Validator**: Verifica la consistencia y validez de las predicciones

El sistema realiza pruebas técnicas fundamentales sin depender del tipo de problema:

| Aspecto Evaluado | Pruebas Realizadas |
|------------------|----------------------|
| **Validez del modelo** | Verificación de compatibilidad con Scikit-learn |
| **Robustez** | Comportamiento ante valores nulos, extremos y tipos incorrectos |
| **Predicciones** | Formato correcto (array NumPy), rango de valores, consistencia |
| **Rendimiento** | Carga, latencia, memoria, CPU, throughput |

Los agentes LLM analizan los resultados de estas pruebas para proporcionar interpretaciones, contexto y recomendaciones en lenguaje natural.

### 6.3 Validación cruzada y evaluación del modelo 🛡️

**El ModelShamanAgent de Midas Touch** implementa estrategias de validación para evaluar el rendimiento de los modelos:

> 🔄 **K-Fold Cross Validation:** Implementa validación cruzada con k=5 para estimaciones robustas de rendimiento  
> 📊 **Estratificación condicional:** Aplica estratificación en la división train/test cuando hay al menos 2 ejemplos por clase  
> 🧮 **Métricas específicas según problema:**  
>   +Clasificación: accuracy, f1-score (weighted), precision, recall  
>   +Regresión: R², MSE, RMSE  
> 🛑 **Validación contra umbrales mínimos:** El OracleAgent verifica que las métricas superen los umbrales configurados

El sistema captura y maneja adecuadamente las advertencias de métricas indefinidas en situaciones con clases minoritarias, garantizando resultados fiables incluso en condiciones complejas.

### 6.4 Análisis de latencia y rendimiento computacional ⚡

**El componente Midas Test** evalúa aspectos críticos para la implementación práctica del modelo mediante mediciones precisas:

<table>
  <tr>
    <th>Tipo de Evaluación</th>
    <th>Métricas</th>
  </tr>
  <tr>
    <td><strong>Tiempo de carga</strong></td>
    <td>Segundos para deserializar el modelo desde archivo joblib</td>
  </tr>
  <tr>
    <td><strong>Latencia</strong></td>
    <td>Tiempos de respuesta en milisegundos para diferentes tamaños de batch (1, 100, 1000, 10000)</td>
  </tr>
  <tr>
    <td><strong>Throughput</strong></td>
    <td>Predicciones por segundo calculadas con un batch de 1000 muestras</td>
  </tr>
  <tr>
    <td><strong>Recursos</strong></td>
    <td>Incremento de uso de CPU (%) y memoria (MB) durante la fase de predicción</td>
  </tr>
</table>

Estas métricas se obtienen mediante pruebas directas sobre el modelo cargado utilizando datos sintéticos generados automáticamente y la biblioteca psutil para monitoreo de recursos.

### 6.5 Generación de reportes detallados 📝

<p align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Test_Reporte_6_5.png?raw=true" alt="Midas Test Reporte" width="500">
</p>

**Midas Test** produce documentación estructurada en español que sintetiza el análisis completo:

- 📄 **Informes en formato Markdown:** Organizados en secciones claramente definidas con emojis distintivos
- 🔄 **Traducción automática:** Conversión del análisis técnico generado por los agentes LLM del inglés al español
- ✅ **Clasificación binaria:** Etiquetado del modelo como "APTO" o "NO APTO" basado en su validez y consistencia de predicciones
- 🔍 **Desglose de resultados:** Presentación detallada de todas las pruebas realizadas y sus resultados

El informe se estructura en siete secciones principales:
1. Resumen del Modelo
2. Métricas de Rendimiento
3. Análisis de Latencia
4. Validez de Predicciones
5. Pruebas de Robustez
6. Recomendación Final
7. Sugerencias de Mejora

Los informes se pueden descargar desde la interfaz Streamlit o se generan automáticamente como "informe_analisis_modelo.md" al usar la interfaz de línea de comandos.

### 6.6 Serialización y persistencia de modelos 💾

**El componente Midas Touch** implementa un sistema completo para la serialización y persistencia de modelos:

- 💾 **Serialización mediante joblib** con versionado automático por timestamp
- 📝 **Guardado de metadatos en JSON** incluyendo:
  - Tipo de modelo y columna objetivo
  - Lista de características utilizadas
  - Métricas de rendimiento detalladas
  - Mapeo entre valores numéricos y etiquetas originales (para clasificación)
- 📊 **Generación de reportes de rendimiento** en formato texto
- 🗃️ **Creación de archivos ZIP** con todos los resultados para facilitar la distribución

El sistema maneja automáticamente la conversión de tipos de datos complejos (como arrays NumPy) a formatos serializables, garantizando la integridad de toda la información del modelo para su posterior uso o análisis.

---

## Sección 7: Procesamiento de Lenguaje Natural

El Procesamiento de Lenguaje Natural (NLP) constituye una tecnología fundamental que atraviesa transversalmente todos los componentes de **MIDAS**, actuando como el mecanismo central que permite la interacción intuitiva mediante lenguaje humano y proporciona capacidades avanzadas de análisis textual.

### 7.1 Arquitectura multimodelo para procesamiento lingüístico 🧠

**MIDAS** implementa una arquitectura sofisticada que emplea múltiples modelos de lenguaje para diferentes tareas:

Modelos Generativos Principales:
- <b>Meta Llama 3.3 (70B):</b> Utilizado en Midas Dataset, Deploy, Help y Test<br>
- <b>Gemini 2.0 Flash:</b> Implementado en Midas Touch, Architech, Plot y Help<br>
- <b>Deepseek V3:</b> Empleado anteriormente en Midas Help para consultas técnicas avanzadas. Fue eliminado por su alta latencia y tiempo de respuesta.

| Modelo Especializado | Uso Principal | Componente |
|----------------------|---------------|------------|
| **BERT Fine-tuned** | Clasificación de consultas | Midas Help |
| **OpenAI 4o-mini** | Generación de resúmenes de chunks | Midas Architect |
| **text-embedding-3-small** | Embeddings para RAG | Midas Architech |
| **BGE-M3** | Embeddings para RAG | Midas Help |
| **BGE V2 M3** | Reranking de resultados | Midas Help |

### 7.2 Sistemas RAG (Retrieval-Augmented Generation) 📚

Además, **MIDAS** implementa arquitecturas RAG sofisticadas en sus componentes de documentación:

**🏗️ MIDAS ARCHITECT (Sistema RAG Agéntico)**
- Segmentación inteligente de textos
- Embeddings mediante text-embedding-3-small
- Base de datos vectorial Supabase
- Herramientas de recuperación y razonamiento

![Midas Architech Interfaz](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Architech.png?raw=true)

**❓ MIDAS HELP (Arquitectura LLM+RAG+Reranker)**
- Clasificador BERT fine-tuned
- Selector de LLM automatizado, aunque puedes "forzar" el que prefieras
- Embeddings BGE-M3
- Reranker BGE V2 M3

![Midas Help RAG](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Help_7_3.png?raw=true)

### 7.3 Generación automática de código 💻

Por último, múltiples componentes de **MIDAS** implementan generación de código mediante NLP (prompt redactado por el usuario):

<table>
  <tr>
    <th>Componente</th>
    <th>Tipo de Código Generado</th>
    <th>Tecnología Base</th>
  </tr>
  <tr>
    <td><strong>Midas Plot</strong></td>
    <td>Visualizaciones matplotlib</td>
    <td>CrewAI Flow + LLM</td>
  </tr>
  <tr>
    <td><strong>Midas Deploy</strong></td>
    <td>Interfaces Streamlit</td>
    <td>AG2 Multiagente</td>
  </tr>
  <tr>
    <td><strong>Midas Touch</strong></td>
    <td>Notebooks completos</td>
    <td>Agentes Python vanilla</td>
  </tr>
</table>

---

## Sección 8: Aplicación Web

**MIDAS** implementa múltiples interfaces web que facilitan la interacción intuitiva con cada componente del sistema, priorizando la accesibilidad y experiencia de usuario mediante tecnologías modernas.

### 8.1 Arquitectura multi-interfaz 🖥️

El sistema adopta un enfoque modular en el desarrollo de interfaces, con implementaciones específicas para cada componente:

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #fd7e14;">
<b>Interfaces Principales:</b><br>
- <b>Streamlit:</b> Para componentes con manipulación directa de datos<br>
- <b>Flask:</b> Para interfaces conversacionales tipo chat<br>
- <b>Tailwind CSS:</b> Para diseño moderno y responsivo
</div>

<table>
  <tr>
    <th>Componente</th>
    <th>Framework Web</th>
    <th>Características Principales</th>
  </tr>
  <tr>
    <td><strong>Midas Dataset</strong></td>
    <td>Streamlit</td>
    <td>Generación de datos sinteticos</td>
  </tr>
  <tr>
    <td><strong>Midas Touch</strong></td>
    <td>Streamlit</td>
    <td>Carga de datos, creación de modelos ML</td>
  </tr>
  <tr>
    <td><strong>Midas Plot</strong></td>
    <td>Streamlit</td>
    <td>Generación de visualizaciones mediante texto</td>
  </tr>
  <tr>
    <td><strong>Midas Test</strong></td>
    <td>Streamlit</td>
    <td>Evaluación de modelos, métricas</td>
  </tr>
  <tr>
    <td><strong>Midas Deploy</strong></td>
    <td>Streamlit</td>
    <td>Generación de interfaces para modelos</td>
  </tr>
  <tr>
    <td><strong>Midas Help</strong></td>
    <td>Flask</td>
    <td>Chat con capacidades RAG</td>
  </tr>
  <tr>
    <td><strong>Midas Assistant</strong></td>
    <td>Flask</td>
    <td>Orientación conversacional</td>
  </tr>
</table>

### 8.2 Interfaces generadas dinámicamente por Midas Deploy 🚀

**El componente Midas Deploy** representa la culminación del pipeline MIDAS, generando automáticamente aplicaciones web funcionales para modelos entrenados:

1. **Model_Analyzer**: Extrae información del modelo
2. **UI_Designer**: Diseña la interfaz adaptada
3. **Code_Generator**: Implementa código Streamlit
4. **Resultado final**: Aplicación Streamlit ejecutable

Este componente transforma modelos joblib estáticos en aplicaciones interactivas listas para usuarios finales, completando el ciclo "de datos a aplicación".

<p align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Deploy_8_2.png?raw=true" alt="Midas Deplot Resultado" width="500">
</p>

### 8.3 Tecnologías y frameworks frontend 🛠️

**MIDAS** implementa un conjunto diverso de tecnologías frontend:

> 📊 **Streamlit:** Framework principal para aplicaciones interactivas de datos  
> 🎨 **Tailwind CSS:** Para interfaces modernas y responsivas en componentes Flask  
> 💻 **JavaScript:** Para interactividad avanzada en interfaces web  
> 📝 **Marked.js:** Para renderizado de Markdown en respuestas de modelos  
> 🌐 **HTML5/CSS3:** Para estructuración y estilizado base

Esta combinación permite experiencias ricas y accesibles desde cualquier navegador moderno.

### 8.4 Despliegue y accesibilidad 🌐

Las interfaces web de **MIDAS** están diseñadas para máxima accesibilidad:

- 📱 **Responsive design** para diferentes dispositivos
- 🌍 **Localización completa** en español
- ♿ **Consideraciones WCAG** para accesibilidad
- 🚀 **Opciones flexibles** de despliegue

Esta capa de aplicación web constituye la interfaz principal entre **MIDAS** y sus usuarios, transformando capacidades técnicas complejas en interacciones intuitivas y productivas.

---

## Sección 9: Conclusiones

El desarrollo e implementación de **MIDAS** representa un avance significativo en la automatización y democratización de los procesos de machine learning, aportando innovaciones sustanciales tanto en el plano técnico como en su impacto potencial en la industria y academia.

### 9.1 Logros principales ✅

**MIDAS** ha alcanzado objetivos ambiciosos que transforman el panorama de la automatización en ML:

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745;">
<b>Principales Hitos:</b><br>
- Automatización integral end-to-end<br>
- Arquitectura multiagente funcional<br>
- Democratización efectiva del ML<br>
- Integración coherente de múltiples frameworks
</div>

### 9.2 Reflexiones sobre el desarrollo 🤔

El proceso de diseño e implementación de **MIDAS** nos ha dado reflexiones valiosas:

* 🔄 **Balance agente-herramienta:** La combinación de agentes con herramientas especializadas dio buen resultado 
* 🎯 **Especialización vs. generalización:** Los agentes especializados mostraron mejor desempeño  
* 📝 **Importancia de la documentación:** La generación automática de documentación (ipynb en Touch) fue muy útil 
* 🧩 **Valor de la arquitectura modular:** El diseño desacoplado facilitó evolución y mantenimiento, así como reparto de tareas 
* 🧠 **Capacidades de LLMs:** Los modelos, bien guiados, demostraron aptitudes sorprendentes en tareas técnicas complejas

### 9.3 Limitaciones actuales ⚠️

A pesar de sus logros, **MIDAS** presenta limitaciones que deben reconocerse:

- 🔌 **Dependencia de servicios externos** de LLM
- 🔄 **Diversidad de frameworks** que aumenta complejidad de mantenimiento
- 📊 **No tan óptimo** en datasets de gran tamaño tanto en filas como columnas
- 🧮 **Soporte limitado** de algoritmos ML
- 🔄 **Ausencia de un orquestador central** completo

### 9.4 Impacto potencial 🌟

**MIDAS** tiene el potencial de generar impacto significativo en múltiples ámbitos:

<table>
  <tr>
    <th>Ámbito</th>
    <th>Impacto</th>
  </tr>
  <tr>
    <td><strong>Educativo</strong></td>
    <td>Herramienta para introducir conceptos ML sin programación avanzada</td>
  </tr>
  <tr>
    <td><strong>Empresarial</strong></td>
    <td>Prototipos rápidos y pruebas de concepto en contextos de negocio</td>
  </tr>
  <tr>
    <td><strong>Investigación</strong></td>
    <td>Plataforma para experimentación ágil con nuevos enfoques</td>
  </tr>
  <tr>
    <td><strong>Democratización</strong></td>
    <td>Extensión de capacidades ML a profesionales no técnicos</td>
  </tr>
</table>

### 9.5 Líneas futuras de desarrollo 🔮

No obstante, el proyecto establece bases sólidas para evoluciones posteriores:

- 🔄 **Integración completa:** Desarrollo de un orquestador central para flujos end-to-end
- 🧠 **Expansión de algoritmos:** Incorporación de deep learning y más modelos ML
- 📊 **Optimización para grandes datos:** Adaptaciones para datasets masivos
- 👥 **Personalización interactiva:** Implementación de modo "semi-manual" consultivo en Midas Touch
- 🏠 **Independencia de APIs:** Exploración de despliegues locales de LLMs más ligeros

### 9.6 Reflexión final 💭

**MIDAS** demuestra que estamos en un punto de inflexión donde la conjunción de sistemas multiagente, modelos de lenguaje avanzados y técnicas tradicionales de ML puede transformar radicalmente cómo concebimos el desarrollo de soluciones de datos. El proyecto no solo automatiza procesos técnicos, sino que reimagina la interacción humano-máquina en contextos altamente especializados, avanzando hacia un paradigma donde la tecnología se adapta a las capacidades humanas, y no al revés.

<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; font-style: italic;">
La metáfora del "toque de Midas" refleja adecuadamente esta visión: convertir algo abundante pero en bruto (datos) en algo valioso y útil (conocimiento accionable) mediante un proceso que, aunque complejo en su interior, se presenta ante el usuario de forma natural e intuitiva.
</div>

---

## 👥 Creadores
<table>
  <tr>
    <td align="center" width="400">
      <div style="border: 3px solid #FFD700; border-radius: 15px; padding: 20px; background-color: rgba(255, 215, 0, 0.05);">
        <div style="border: 2px solid #FFD700; border-radius: 50%; padding: 3px; margin: 0 auto;">
          <a href="https://warcos.dev">
            <img src="https://github.com/warc0s.png" width="220" alt="Marcos García" style="border-radius: 50%; border: 2px solid #FFF; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);">
          </a>
        </div>
        <div style="margin: 10px auto; background-color: rgba(255, 215, 0, 0.1); border-radius: 10px; padding: 5px; width: 80%;">
          <div style="background-color: #FFD700; width: 65%; height: 10px; border-radius: 5px;"></div>
          <p style="color: #FFD700; margin: 5px 0; font-weight: bold; font-size: 14px; text-align: center;">Contribución: 65%</p>
        </div>
        <h2 style="color: #FFD700; margin: 15px 0; font-family: 'Helvetica Neue', sans-serif;">Marcos García Estévez</h2>
        <div style="display: flex; gap: 10px; justify-content: center; margin-top: 15px;">
          <a href="https://github.com/warc0s">
            <img src="https://custom-icon-badges.demolab.com/badge/-GitHub-1a1a1a?style=for-the-badge&logo=github&logoColor=FFD700" alt="GitHub">
          </a>
          <a href="https://warcos.dev">
            <img src="https://custom-icon-badges.demolab.com/badge/-Portfolio-1a1a1a?style=for-the-badge&logo=browser&logoColor=FFD700" alt="Portfolio">
          </a>
        </div>
      </div>
    </td>
    
  <td align="center" width="400">
    <div style="border: 3px solid #FFD700; border-radius: 15px; padding: 20px; background-color: rgba(255, 215, 0, 0.05);">
      <div style="border: 2px solid #FFD700; border-radius: 50%; padding: 3px; margin: 0 auto;">
        <a href="https://github.com/jesusact">
          <img src="https://github.com/jesusact.png" width="220" alt="Jesús Aceituno" style="border-radius: 50%; border: 2px solid #FFF; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);">
        </a>
      </div>
      <div style="margin: 10px auto; background-color: rgba(255, 215, 0, 0.1); border-radius: 10px; padding: 5px; width: 80%;">
        <div style="background-color: #FFD700; width: 35%; height: 10px; border-radius: 5px;"></div>
        <p style="color: #FFD700; margin: 5px 0; font-weight: bold; font-size: 14px; text-align: center;">Contribución: 35%</p>
      </div>
      <h2 style="color: #FFD700; margin: 15px 0; font-family: 'Helvetica Neue', sans-serif;">Jesús Aceituno Valero</h2>
      <div style="display: flex; gap: 10px; justify-content: center; margin-top: 15px;">
        <a href="https://github.com/jesusact">
          <img src="https://custom-icon-badges.demolab.com/badge/-GitHub-1a1a1a?style=for-the-badge&logo=github&logoColor=FFD700" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/jesus-aceituno-valero/">
          <img src="https://custom-icon-badges.demolab.com/badge/-LinkedIn-1a1a1a?style=for-the-badge&logo=linkedin&logoColor=FFD700" alt="LinkedIn">
        </a>
      </div>
    </div>
  </td>
  </tr>
</table>
