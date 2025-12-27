================================================
File: README.md
================================================
<div align="center">
  <img src="https://github.com/warc0s/MIDAS/blob/main/Extra/logo1.png" alt="MIDAS Logo" width="50%">
  <h1>MIDAS - Multi-agent Intelligent Data Automation System 🤖</h1>
  <p><em>Convierte tus ideas en modelos ML listos para producción</em></p>
</div>

<!-- Enlaces Rápidos -->
<div align="center">
  <p>
    <a href="https://github.com/warc0s/MIDAS" target="_blank">
      <img width="300" src="https://img.shields.io/badge/📦_REPOSITORIO-GitHub-181717?style=for-the-badge&labelColor=111111&logo=github&logoColor=white" alt="Repositorio"/>
    </a>
    &nbsp;&nbsp;
    <a href="../Extra/Documentacion/docs/index.md" target="_blank">
      <img width="300" src="https://img.shields.io/badge/📚_DOCUMENTACIÓN-Local-22A699?style=for-the-badge&labelColor=15756C&logo=markdown&logoColor=white" alt="Documentación (local)"/>
    </a>
  </p>
  <p>
    <a href="../Extra/Webs/MIDASTFM-Triptico-Final.pdf" target="_blank">
      <img width="300" src="https://img.shields.io/badge/🎯_PRESENTACIÓN-PDF-FF9E00?style=for-the-badge&labelColor=D97F00&logo=googledrive&logoColor=white" alt="Presentación (PDF)"/>
    </a>
    &nbsp;&nbsp;
    <a href="https://youtu.be/G5KMC8kFZEY" target="_blank">
      <img width="300" src="https://img.shields.io/badge/🎥_VIDEO_EXPLICATIVO-Video Youtube-FF5757?style=for-the-badge&labelColor=D63030&logo=youtube&logoColor=white" alt="Video Explicativo"/>
    </a>
  </p>
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

> 🔄 **Balance agente-herramienta:** La combinación de agentes con herramientas especializadas dio buen resultado 
> 🎯 **Especialización vs. generalización:** Los agentes especializados mostraron mejor desempeño  
> 📝 **Importancia de la documentación:** La generación automática de documentación (ipynb en Touch) fue muy útil 
> 🧩 **Valor de la arquitectura modular:** El diseño desacoplado facilitó evolución y mantenimiento, así como reparto de tareas 
> 🧠 **Capacidades de LLMs:** Los modelos, bien guiados, demostraron aptitudes sorprendentes en tareas técnicas complejas

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
          <p style="color: #FFD700; margin: 5px 0; font-weight: bold; font-size: 14px; text-align: center;">Contribución: 70%</p>
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
        <p style="color: #FFD700; margin: 5px 0; font-weight: bold; font-size: 14px; text-align: center;">Contribución: 30%</p>
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


================================================
File: Extra/Documentacion/docs/architecture.md
================================================
# Arquitectura del Sistema MIDAS

## Visión General

MIDAS (Multi-agent Intelligent Data Automation System) es una plataforma multiagente diseñada para automatizar y optimizar el ciclo completo de ciencia de datos, desde la generación de datasets hasta el despliegue de modelos. El nombre MIDAS hace referencia al Rey Midas, cuyo toque convertía objetos en oro, simbolizando cómo este sistema transforma datos crudos (CSV) en valiosos modelos predictivos (joblib).

La arquitectura de MIDAS implementa un enfoque modular y desacoplado, donde cada componente especializado se comunica a través de interfaces bien definidas y formatos estándar. El sistema aprovecha múltiples frameworks de IA conversacional (AG2, CrewAI, LiteLLM) y modelos de lenguaje de gran escala (LLMs) para proporcionar capacidades avanzadas de automatización, razonamiento y generación.

## Componentes Principales

MIDAS está compuesto por ocho módulos especializados que pueden funcionar de manera independiente o como parte de un flujo de trabajo integrado:

1. **Midas Dataset**: Generador de datasets sintéticos basado en agentes AG2
2. **Midas Touch**: Motor de procesamiento automático de ML que transforma datos en modelos
3. **Midas Test**: Evaluador de calidad y rendimiento de modelos ML
4. **Midas Deploy**: Generador de interfaces para modelos entrenados
5. **Midas Plot**: Creador de visualizaciones mediante instrucciones en lenguaje natural
6. **Midas Architect**: Sistema RAG agéntico para documentación técnica
7. **Midas Help**: Asistente de documentación con RAG mejorado y reranking
8. **Midas Assistant**: Chatbot inteligente para navegación y orientación

## Diagrama de Arquitectura Conceptual

La arquitectura de MIDAS sigue un patrón de flujo de trabajo lineal con múltiples puntos de entrada y retroalimentación:

![Midas Completo Diagrama](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Diagrama_ArquitecturaModular.png?raw=true)

## Tecnologías y Frameworks

MIDAS integra múltiples tecnologías de vanguardia:

### Frameworks de IA Multi-agente:
- **AG2**: Utilizado en Midas Dataset, Midas Deploy y Midas Test para orquestar conversaciones entre agentes especializados
- **CrewAI**: Implementado en Midas Plot para gestionar flujos de trabajo de generación visual
- **Python "vanilla"**: Sistema de agentes personalizado en Midas Touch

### Modelos de Lenguaje (LLMs):
- **Meta Llama 3.3 (70B)**: Utilizado principalmente en Midas Dataset, Midas Deploy y Midas Test
- **Gemini 2.0 Flash**: Implementado en Midas Touch y como opción en Midas Help
- **Deepseek V3**: Utilizado en ciertos casos de Midas Help
- **OpenAI 4o-mini**: Para generación de resúmenes en Midas Architect

### Bases de Datos y Almacenamiento:
- **Supabase**: Como base de datos vectorial en Midas Architect
- **Sistemas de archivos locales**: Para almacenamiento de modelos y datasets

### Interfaces de Usuario:
- **Streamlit**: Implementado en todos los componentes con interfaz gráfica
- **Flask**: Utilizado en versiones web de Midas Assistant y Midas Help

### Procesamiento de Datos y ML:
- **Pandas**: Para manipulación y análisis de datos
- **Scikit-learn**: Para creación y evaluación de modelos
- **Matplotlib**: Para generación de visualizaciones

### Otros Componentes:
- **Faker**: Para generación de datos sintéticos
- **LiteLLM**: Como abstracción para interacción con diferentes LLMs
- **e2b Sandbox**: Para ejecución segura de código en Midas Plot
- **Embeddings**: Diversos modelos como text-embedding-3-small y BGE-M3

## Flujos de Datos y Comunicación

MIDAS implementa varios flujos de trabajo principales:

1. **Flujo de Generación de Modelos**:
   - Midas Dataset → Midas Touch → Midas Test → Midas Deploy
   
2. **Flujo de Visualización**:
   - Midas Dataset/Datos existentes → Midas Plot
   
3. **Flujos de Soporte**:
   - Usuario → Midas Help/Architect/Assistant → Usuario

Cada componente produce artefactos específicos que pueden servir como entradas para otros componentes:

- **Midas Dataset**: Produce archivos CSV con datos sintéticos
- **Midas Touch**: Genera modelos ML en formato joblib
- **Midas Test**: Crea informes de evaluación en Markdown
- **Midas Deploy**: Produce aplicaciones Streamlit ejecutables
- **Midas Plot**: Genera visualizaciones en formato PNG

## Consideraciones de Diseño

La arquitectura de MIDAS se basa en varios principios clave:

1. **Modularidad**: Cada componente está diseñado para funcionar de forma independiente
2. **Especialización**: Los componentes se centran en resolver tareas específicas del flujo de ML
3. **Interoperabilidad**: Uso de formatos estándar (CSV, joblib) para facilitar la integración
4. **Automatización**: Minimización de intervención manual en procesos complejos
5. **Explicabilidad**: Generación automática de documentación y visualizaciones para mejorar la comprensión
6. **Extensibilidad**: Arquitectura que permite añadir nuevos componentes o mejorar los existentes

## Limitaciones de la Arquitectura Actual

La arquitectura actual presenta algunas limitaciones que podrían abordarse en versiones futuras:

1. **Integración parcial**: Aunque conceptualmente forman un sistema, los componentes no están completamente integrados en una plataforma unificada
2. **Diversidad de frameworks**: El uso de diferentes frameworks (AG2, CrewAI) puede complicar el mantenimiento
3. **Dependencia de servicios externos**: Varios componentes dependen de APIs externas para acceder a LLMs
4. **Ausencia de orquestación central**: No existe un componente que coordine automáticamente el flujo completo
5. **Limitaciones de escalabilidad**: Algunos componentes están optimizados para datasets de tamaño pequeño a mediano

[Empezar →](/modules/midas_assistant)

================================================
File: Extra/Documentacion/docs/faq.md
================================================
# Preguntas Frecuentes (FAQ)

## Preguntas Generales

### ¿Qué es MIDAS?
MIDAS (Multi-agent Intelligent Data Automation System) es un sistema multiagente diseñado para automatizar y optimizar el ciclo completo de ciencia de datos, desde la generación de datasets hasta el despliegue de modelos, utilizando tecnologías de IA conversacional y LLMs.

### ¿Por qué se llama MIDAS?
El nombre hace referencia al Rey Midas de la mitología griega, cuyo toque convertía objetos en oro. De manera similar, este sistema transforma datos crudos (datasets CSV) en "oro" (modelos de ML bien entrenados y precisos).

### ¿Cuáles son los componentes principales de MIDAS?
MIDAS consta de ocho componentes principales:
- Midas Dataset: Generador de datasets sintéticos
- Midas Touch: Automatización de flujo completo de ML
- Midas Test: Evaluador de calidad de modelos
- Midas Deploy: Generador de interfaces para modelos
- Midas Plot: Creador de visualizaciones desde lenguaje natural
- Midas Architect: Sistema RAG para documentación técnica
- Midas Help: Asistente de documentación con RAG+Reranker
- Midas Assistant: Chatbot de orientación sobre el sistema

### ¿MIDAS es un único programa o varios independientes?
MIDAS es un sistema compuesto por múltiples componentes independientes que pueden funcionar de forma autónoma o como parte de un flujo de trabajo integrado. Cada componente está diseñado para resolver una parte específica del proceso de ciencia de datos.

### ¿Qué tecnologías utiliza MIDAS?
MIDAS utiliza diversas tecnologías, incluyendo:
- Frameworks de agentes: AG2 (fork mejorado de AutoGen), CrewAI, Pydantic AI
- Modelos de lenguaje: Llama 3.3, Gemini 2.0, Deepseek V3...
- Interfaces: Streamlit, Flask
- Procesamiento de datos: Pandas, Scikit-learn
- Visualización: Matplotlib
- Bases de datos: Supabase
- Otros: LiteLLM, Faker, e2b Sandbox...

## Uso y Funcionalidad

### ¿Cómo empiezo a usar MIDAS?
Para comenzar, debe instalar los componentes que desee utilizar y configurar las credenciales necesarias para acceder a los servicios de LLM. Luego puede ejecutar cada componente individualmente según sus necesidades.

### ¿Necesito conocimientos de programación para usar MIDAS?
Los componentes de MIDAS están diseñados con interfaces intuitivas que reducen la necesidad de programación. Sin embargo, cierto conocimiento básico de ciencia de datos y ML ayudará a comprender mejor los resultados y a formular prompts efectivos.

### ¿Qué tipos de modelos de ML puede crear MIDAS?
Actualmente, Midas Touch se centra en modelos de clasificación y regresión utilizando algoritmos de Scikit-learn, específicamente RandomForest y GradientBoosting.

### ¿Qué formatos de datos acepta MIDAS?
MIDAS puede trabajar con diversos formatos:
- Midas Touch: CSV, Excel, Parquet, JSON
- Midas Plot: CSV
- Midas Test/Deploy: Modelos en formato joblib

### ¿Puedo integrar MIDAS con mis flujos de trabajo existentes?
Sí, los componentes de MIDAS están diseñados para ser modulares. Puede utilizar Midas Dataset para generar datos, procesar estos datos con sus propias herramientas, y luego usar Midas Test para evaluar los modelos resultantes.

## Capacidades y Limitaciones

### ¿Qué tamaño de datasets puede manejar MIDAS?
Midas Touch está optimizado para datasets de tamaño pequeño a mediano (recomendable hasta ~25K filas). Datasets muy grandes pueden causar problemas de rendimiento.

### ¿MIDAS requiere conexión a internet?
Sí, la mayoría de los componentes dependen de servicios externos de LLM como DeepInfra o Google AI, por lo que requieren conexión a internet para funcionar.

### ¿Qué credenciales API necesito para usar MIDAS?
Dependiendo de los componentes que utilice, puede necesitar:
- API key de DeepInfra (para componentes que usan Llama 3.3)
- API key de Google AI (para componentes que usan Gemini)

### ¿MIDAS Touch puede explicar sus decisiones?
Sí, un enfoque clave de MIDAS Touch es la explicabilidad. Genera notebooks detallados que documentan cada paso del proceso, proporciona informes completos, y incluso Midas Deploy incluye comentarios en el código generado.

### ¿Cuáles son las limitaciones actuales más importantes?
Algunas limitaciones importantes incluyen:
- Soporte limitado de modelos ML (principalmente Scikit-learn)
- Optimización para datasets de tamaño pequeño a mediano
- Ausencia de optimización avanzada de hiperparámetros
- Falta de integración completa entre todos los componentes
- Dependencia de servicios externos para LLMs

### ¿Cómo se genera el .ipynb en Midas Touch?
Primero se genera un json nbformat vacio y se va llenando. Cada agente llama a "NotebookScribeAgent" cada vez que realiza una acción, de forma que queda reflejado en tiempo real en este cuaderno jupyter. Se realiza en pares de markdown-python para que cada código utilizado tenga su explicación en markdown.

### ¿Cuándo se usa un LLM en Midas Touch? ¿O algún tipo de IA generativa?
Solo al inicio, para extraer la columna a predecir y si es un problema de regresion o clasificacion a partir del prompt del usuario. El resto de la ejecución es un conjunto de reglas expertas, definidas a nivel de código, con varios agentes cada uno con tareas específicas.
Probamos a usar un enfoque full-LLM pero tenia demasiada latencia, costes (+100 llamadas para entrenar un pequeño modelo), y un pequeño fallo en el formato del json o similares, rompía el flujo de ejecución. Por eso decidimos usar este enfoque híbrido.

## Problemas Comunes

### El LLM no responde o da errores de timeout
Asegúrese de que sus credenciales API estén correctamente configuradas y que tenga una conexión estable a internet. Los servicios de LLM pueden tener límites de velocidad o períodos de mantenimiento que afecten la disponibilidad.

### El modelo generado no tiene buena precisión
La calidad del modelo depende en gran medida de los datos de entrada. Asegúrese de que su dataset tenga suficientes ejemplos, características relevantes y esté correctamente preparado. Puede probar con diferentes prompts en Midas Touch para especificar mejor el objetivo.

### Midas Plot no genera la visualización que esperaba
Las descripciones en lenguaje natural pueden ser interpretadas de diferentes maneras. Intente ser más específico en su prompt, mencionando el tipo exacto de gráfico, las variables a utilizar y cualquier personalización deseada.

### Los agentes parecen "atascarse" en una conversación infinita
En raras ocasiones, los sistemas multiagente pueden entrar en bucles de conversación. Si observa que un componente no avanza después de varios minutos, puede intentar reiniciar el proceso con un prompt más claro o directivas más específicas.

## Desarrollo y Contribución

### ¿MIDAS es de código abierto?
Sí, MIDAS es un proyecto de código abierto desarrollado como Trabajo Fin de Máster (TFM). Puede encontrar el código fuente en [GitHub](https://github.com/warc0s/MIDAS).

### ¿Cómo puedo contribuir al proyecto?
Las contribuciones son bienvenidas. Puede contribuir reportando problemas, sugiriendo mejoras o enviando pull requests al repositorio GitHub.


================================================
File: Extra/Documentacion/docs/index.md
================================================
# MIDAS: Multi-agent Intelligent Data Automation System

![MIDAS Logo](https://github.com/warc0s/MIDAS/raw/main/Extra/logo1.png)

## Transformando Ideas en Oro

MIDAS es un sistema multiagente diseñado para automatizar y optimizar el ciclo completo de ciencia de datos. Su nombre proviene de la figura mitológica del Rey Midas, cuyo toque convertía objetos en oro, simbolizando cómo este sistema transforma datos crudos en valiosos modelos predictivos y visualizaciones.

## Capacidades Principales

MIDAS ofrece un conjunto completo de herramientas para científicos de datos, desarrolladores y analistas:

- **Generación de Datos Sintéticos**: Creación automática de datasets realistas para testing y desarrollo
- **Automatización de ML**: Transformación de datos en modelos predictivos sin intervención manual
- **Evaluación de Modelos**: Análisis exhaustivo de calidad, rendimiento y robustez
- **Visualización Inteligente**: Creación de gráficos mediante descripciones en lenguaje natural
- **Despliegue Rápido**: Generación automática de interfaces para modelos
- **Asistencia y Documentación**: Sistemas avanzados de soporte basados en RAG

## Componentes del Sistema

![Midas Main Website](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Main.png?raw=true)

MIDAS está compuesto por ocho módulos especializados que pueden funcionar de manera independiente o como parte de un flujo de trabajo integrado:

### [Midas Dataset](./modules/midas_dataset.md)
Generador de datasets sintéticos que utiliza agentes conversacionales basados en AG2 para interpretar requisitos y crear datos realistas mediante la biblioteca Faker.

### [Midas Touch](./modules/midas_touch.md)
Motor de automatización de ML que transforma datasets en modelos entrenados, implementando un enfoque multigente con Python vanilla y Gemini 2.0 Flash para gestionar el proceso completo.

### [Midas Test](./modules/midas_test.md)
Evaluador de modelos que analiza la calidad, rendimiento y robustez mediante agentes especializados basados en AG2, generando informes detallados en formato Markdown.

### [Midas Deploy](./modules/midas_deploy.md)
Generador de interfaces que crea aplicaciones Streamlit personalizadas para modelos ML, utilizando agentes conversacionales para analizar y diseñar la mejor experiencia de usuario.

### [Midas Plot](./modules/midas_plot.md)
Creador de visualizaciones que transforma descripciones en lenguaje natural en gráficos utilizando CrewAI Flow y ejecución segura de código en un entorno sandbox.

### [Midas Architect](./modules/midas_architect.md)
Sistema RAG agéntico que proporciona acceso inteligente a documentación técnica de frameworks como Pydantic AI, LlamaIndex, CrewAI y AG2, utilizando Supabase como base de datos vectorial.

### [Midas Help](./modules/midas_help.md)
Asistente de documentación que implementa una arquitectura LLM+RAG+Reranker para resolver consultas sobre el sistema MIDAS mediante lenguaje natural.

### [Midas Assistant](./modules/midas_assistant.md)
Chatbot inteligente que proporciona orientación, recomendaciones y soporte informativo sobre todos los componentes del ecosistema MIDAS.

## Primeros Pasos

Para comenzar a utilizar MIDAS, siga estos pasos:

1. **Instalación**.

2. **Configuración**: Configure las credenciales necesarias en los example.env

3. **Flujos de trabajo recomendados**:
   - Para crear y entrenar un modelo desde cero: Dataset → Touch → Test → Deploy
   - Para visualizar datos existentes: Plot
   - Para obtener ayuda y documentación: Assistant o Help

## Propósito y Filosofía

MIDAS nace de la visión de democratizar y automatizar los procesos de ciencia de datos mediante el uso de tecnologías de IA conversacional. El sistema busca:

1. **Reducir la barrera de entrada** para tareas complejas de ML
2. **Aumentar la productividad** de científicos de datos experimentados
3. **Mejorar la calidad** mediante evaluaciones estandarizadas
4. **Facilitar la documentación** y comprensión de procesos técnicos
5. **Promover las mejores prácticas** en el desarrollo de modelos

## Recursos Adicionales

- [Arquitectura del Sistema](./architecture.md)
- [Preguntas Frecuentes](./faq.md)
- [Repositorio GitHub](https://github.com/warc0s/MIDAS)

## Agradecimientos

MIDAS ha sido desarrollado como un Trabajo Fin de Máster (TFM) y se beneficia de múltiples frameworks y tecnologías de código abierto como AG2, CrewAI, Streamlit, Pandas, Scikit-learn y otros.

[Empezar →](/modules/midas_assistant)

================================================
File: Extra/Documentacion/docs/modules/midas_architect.md
================================================
# Midas Architect

## Descripción General

Midas Architect es un componente de Recuperación Aumentada Generativa (RAG) que utiliza Supabase como base de datos vectorial para almacenar y consultar documentación técnica de cuatro frameworks de desarrollo: Pydantic AI, LlamaIndex, CrewAI y AG2.

Este sistema implementa un enfoque de RAG asistido por agentes, permitiendo navegar inteligentemente por la documentación técnica mediante el uso de herramientas específicas de consulta. Utiliza modelos de lenguaje grandes (LLM), específicamente Gemini 2.0 Flash, para procesar consultas y generar respuestas contextualizadas basadas en la documentación oficial de estos frameworks.

## Arquitectura Técnica

### Backend:

- **Sistema de Ingesta de Documentación**:
  - Utiliza *Crawl4AI* para extraer automáticamente contenido en formato Markdown de los sitemaps oficiales de cada framework.
  - Procesa cada página web recuperada y la convierte a un formato optimizado para su posterior procesamiento.

- **Procesamiento de Texto**:
  - Implementa una *segmentación inteligente* que divide el texto en chunks de máximo 5000 caracteres.
  - La segmentación respeta las siguientes estructuras para mantener la coherencia contextual:
    - *Bloques de código*: Detecta marcadores "```" después del 30% del chunk.
    - *Párrafos*: Identifica saltos de línea dobles "\n\n" después del 30% del chunk.
    - *Oraciones*: Localiza finales de oración ". " después del 30% del chunk.
  - Esta estrategia garantiza chunks de tamaño óptimo para el procesamiento por LLMs.

- **Sistema de Embeddings**:
  - Utiliza el modelo *text-embedding-3-small* de OpenAI (1536 dimensiones) para generar representaciones vectoriales del texto.
  - Implementa el modelo *gpt-4o-mini* para la generación automática de títulos y resúmenes de cada chunk.

- **Base de Datos Vectorial**:
  - *Supabase* como infraestructura para almacenar embeddings y metadatos.
  - Estructura de tabla SQL optimizada para consultas vectoriales mediante índices IVFFlat.
  - Cada registro incluye: *embedding vectorial*, *URL de origen*, *título*, *resumen*, *contenido completo* y *metadatos* (incluyendo la fuente del documento).

- **Sistema de Consulta Basado en Herramientas**:
  - Implementa tres herramientas principales mediante Pydantic AI:
    - *retrieve_relevant_documentation*: Recuperación basada en similitud de embeddings.
    - *list_documentation_pages*: Listado de todas las URLs disponibles para un framework específico.
    - *get_page_content*: Recuperación de todos los chunks de una página específica mediante URL exacta.

### Frontend:
- Implementado en Streamlit con diseño responsivo y experiencia de usuario mejorada.
- Interfaz con estilos personalizados y animaciones para una mejor experiencia.
- Selector de framework que permite cambiar dinámicamente entre las diferentes fuentes de documentación.
- Sistema de streaming de respuestas en tiempo real.

## Funcionalidad

- Proporciona respuestas precisas a consultas técnicas sobre los frameworks Pydantic AI, LlamaIndex, CrewAI y AG2.
- Ofrece capacidad de comprensión y contextualización profunda de la documentación técnica.
- Permite la recuperación selectiva e inteligente de información relevante mediante enfoque agéntico.
- Facilita el acceso a información técnica compleja sin necesidad de navegar manualmente por la documentación.
- Responde en español a pesar de que la documentación original está en inglés.
- Dirigido principalmente a desarrolladores que trabajan con estos frameworks y buscan resolver dudas técnicas de forma rápida.

## Guía de Uso

Para interactuar con Midas Architect:

1. **Seleccionar el framework** sobre el que se desea consultar información mediante el selector en la barra lateral.

2. **Formular consultas específicas** en español sobre el framework seleccionado.
  
   *Ejemplo de consulta:* "¿Cómo puedo implementar un RAG básico con LlamaIndex?"

3. El sistema procesará la consulta a través de su pipeline:
   - Analizará la consulta para entender qué información se necesita.
   - Recuperará chunks relevantes de la documentación mediante similitud vectorial.
   - Si es necesario, consultará páginas completas o listará recursos disponibles.
   - Generará una respuesta detallada en español basada en la documentación original.

## Referencias y Recursos

- Modelo de embeddings: [OpenAI text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- Modelo para resúmenes y títulos: OpenAI gpt-4o-mini
- Modelo principal de LLM: Gemini 2.0 Flash
- Base de datos vectorial: [Supabase Vector](https://supabase.com/docs/guides/ai)
- Frameworks documentados:
  - [Pydantic AI](https://docs.pydantic.dev/)
  - [LlamaIndex](https://docs.llamaindex.ai/)
  - [CrewAI](https://docs.crewai.com/)
  - [AG2](https://docs.ag2.ai/docs/user-guide/basic-concepts/installing-ag2)
- Librería de crawling: [Crawl4AI](https://github.com/unclecode/crawl4ai)

## Limitaciones Actuales

- La documentación de LlamaIndex está incompleta debido a su extensión (más de 1650 páginas), lo que puede afectar a la capacidad del sistema para responder algunas consultas específicas sobre este framework.
- No se ha implementado un sistema de citas de fuentes para las respuestas. Los intentos de incluir fuentes mediante prompting resultaron en la generación de URLs inexistentes (alucinadas).
- El modelo Gemini 2.0 Flash puede tener limitaciones en el procesamiento de consultas muy específicas o complejas.
- Sistema diseñado para consultas en español únicamente a pesar de que la documentación original está en inglés.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Architech.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Arch_Full_Captura.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_assistant.md
================================================
# Midas Assistant

## Descripción General
MIDAS Assistant es el componente del sistema MIDAS que proporciona orientación, recomendaciones y soporte informativo sobre todos los componentes del ecosistema MIDAS. Actúa como un chatbot inteligente utilizando tecnología LLM para responder consultas relacionadas con el sistema MIDAS, sus componentes individuales y flujos de trabajo óptimos.

Este componente se basa en LiteLLM como framework de abstracción, permitiendo la integración con diferentes modelos de lenguaje como Gemini, dependiendo de la configuración del usuario. Básicamente, a grandes rasgos, es un LLM con un gran system prompt con información acerca de cada componente Midas para así resolver dudas sobre el mismo.

## Arquitectura Técnica

### Backend
- **Lenguaje y framework:** 
 - *Python* como lenguaje principal
 - *LiteLLM* como framework de abstracción para interactuar con LLMs
 - *Flask* para la versión web

- **Componentes clave:**
 - *Módulo de configuración:* Gestiona las variables de entorno y la configuración del modelo a utilizar
 - *Gestor de contexto:* Mantiene el historial de conversación para proporcionar respuestas contextualizadas
 - *Sistema de prompts:* Utiliza un prompt de sistema extenso con información detallada sobre todos los componentes MIDAS
 - *API REST:* En la versión Flask, proporciona endpoints para consultas y gestión de conversaciones

- **Flujo de procesamiento:**
 1. Recepción de la consulta del usuario
 2. Consulta al LLM configurado vía LiteLLM
 3. Formateo y entrega de la respuesta al usuario

### Frontend
- **Versión CLI:**
 - Terminal interactiva con *Colorama* para destacar elementos visuales
 - Formato de texto para mejorar la legibilidad de las respuestas

- **Versión Web:**
 - *HTML/CSS* con *Tailwind CSS* para una interfaz moderna y responsiva
 - *JavaScript* para la gestión del chat y efectos visuales
 - *Marked.js* para renderizar Markdown de las respuestas del LLM

## Funcionalidad
- Proporciona información completa sobre los ocho componentes del sistema MIDAS
- Genera recomendaciones de flujos de trabajo adaptados a las necesidades del usuario
- Sugiere prompts efectivos para interactuar con cada componente específico
- Direcciona consultas técnicas específicas hacia Midas Help - Dado que la idea es sugerir usos de los componentes Midas, no responder dudas sobre el TFM.
- Mantiene un tono profesional y conciso, enfocado en proporcionar valor práctico
- Presenta la información en formato Markdown para una mejor legibilidad

## Guía de Uso

### Versión CLI
1. Configura tus credenciales en el archivo `.env` (siguiendo el formato de `example.env`)
2. Ejecuta el script `Midas_Assistant_cli.py`
3. Inicia el diálogo con preguntas como:
  - "¿Qué componente MIDAS debo usar para visualizar datos?"
  - "Dame un prompt efectivo para Midas Plot"
  - "¿Cómo implemento un flujo de trabajo para crear un modelo predictivo?"

### Versión Web
1. Configura tus credenciales en el archivo `.env`
2. Ejecuta `Midas_Assitant_flask.py` para iniciar el servidor
3. Accede a la interfaz web desde tu navegador
4. Interactúa con el chatbot mediante el campo de texto
5. Utiliza el panel de componentes para acceder rápidamente a información específica

**Ejemplo de interacción:**
- Usuario: "Necesito crear un dataset y visualizarlo para analizar tendencias"
- MIDAS Assistant: "Para ese flujo de trabajo te recomiendo usar MIDAS DATASET para generar tus datos sintéticos, especificando el número de filas y columnas necesario. Luego, utiliza MIDAS PLOT para visualizar las tendencias. Para MIDAS PLOT, un prompt efectivo sería: 'Genera una gráfica de líneas temporal que muestre la evolución de [variable] agrupada por [categoría]'."

## Referencias y Recursos
- Repositorio GitHub: [MIDAS](https://github.com/warc0s/MIDAS)
- Website de LiteLLM: [LiteLLM Documentation](https://litellm.ai/)

## Limitaciones Actuales
- El componente está optimizado para responder sobre el ecosistema MIDAS, rechazando educadamente consultas fuera de este ámbito
- La calidad de respuesta depende del modelo LLM configurado, siendo gemini-2.0-flash el mejor calidad/precio de todos los que hemos probado
- La versión CLI no conserva el historial de conversación entre sesiones (aunque la versión web sí lo hace)
- No existe integración directa con otros componentes MIDAS, es puramente informativo
- La idea original era implementarlo como un agente que tuviera como herramientas cada componente MIDAS, de forma que con un prompt simple como "hazme un modelo ML que prediga X" fuera capaz de invocar automáticamente estas herramientas con los mejores prompts posibles que el agente conoce y devolviera exactamente lo que el usuario necesita. Sin embargo, debido a limitaciones de tiempo, esta funcionalidad no pudo ser implementada.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Assistant.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Assistant_Full_Captura.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_dataset.md
================================================
# Midas Dataset

## Descripción General

MIDAS Dataset es el componente MIDAS diseñado para la generación automatizada de conjuntos de datos sintéticos. Su objetivo principal es facilitar la creación de datos de prueba realistas para desarrollo y testing, sin necesidad de exponer información sensible o real.

El sistema utiliza tecnologías de Inteligencia Artificial, específicamente Large Language Models (LLM), para interpretar solicitudes del usuario, validar parámetros y clasificar columnas de datos. Se basa en la biblioteca Faker para generar datos sintéticos convincentes y ofrece tanto una interfaz de línea de comandos como una interfaz web mediante Streamlit.

## Arquitectura Técnica

### Tecnologías Utilizadas
- **Python**: Lenguaje de programación principal
- **AG2**: Framework para sistema multi-agente
- **Pandas**: Biblioteca para manipulación de datos
- **Faker**: Generación de datos sintéticos realistas
- **Streamlit**: Interfaz gráfica de usuario web
- **Meta Llama 3.3 70B Instruct Turbo**: Modelo LLM (a través de DeepInfra API)

### Componentes Clave
- **Input Agent**: Recibe y procesa las solicitudes iniciales del usuario
- **Validation Agent**: Verifica que los parámetros proporcionados sean válidos
- **Column Classifier Agent**: Clasifica nombres de columnas para mapearlos a atributos de Faker
- **User Proxy**: Coordina el flujo de trabajo entre los diferentes agentes
- **Sistema de Detección de Tipos**: Analiza nombres de columnas para inferir el tipo de datos a generar

### Flujo de Procesamiento
1. El usuario proporciona los parámetros (número de registros y nombres de columnas)
2. Para cada columna numérica, se pueden especificar valores mínimos y máximos
3. El sistema detecta automáticamente el tipo de datos para cada columna basándose en su nombre
4. Se genera el dataset sintético utilizando Faker con localización es_ES (español)
5. El usuario puede modificar el dataset generado (eliminar o añadir columnas)
6. El resultado puede ser descargado en formato CSV o Excel

## Funcionalidad

### Detección Automática de Tipos
El sistema analiza los nombres de columnas e intenta determinar el tipo de datos más apropiado para generar:

1. Busca coincidencias exactas (ej: "nombre" → name)
2. Busca coincidencias parciales (ej: "email_cliente" → email)
3. Utiliza algoritmos de coincidencia aproximada para nombres similares
4. Si no hay coincidencia, usa "text" como valor predeterminado

## Tipos de Datos Soportados
El sistema soporta una amplia variedad de tipos de datos a través del mapeo de nombres de columnas a métodos de Faker:

#### Información Personal
- **Nombres**: nombre, primer_nombre, segundo_nombre, apellido, apellido_paterno, apellido_materno, nombre_completo
- **Identidad**: genero, sexo, edad, fecha_nacimiento
- **Documentos**: dni, cedula, pasaporte, curp, rfc

#### Información de Contacto
- **Comunicación**: correo, email, telefono, celular, movil, whatsapp
- **Perfiles**: red_social, usuario, nickname
- **Seguridad**: contraseña, password

#### Direcciones
- **Ubicación**: direccion, calle, numero_exterior, numero_interior
- **Localidad**: colonia, municipio, ciudad, estado, region, pais
- **Códigos**: codigo_postal, zip

#### Empresa y Trabajo
- **Organizaciones**: empresa, compania, negocio
- **Posiciones**: puesto, cargo, departamento
- **Compensación**: sueldo, salario

#### Información Financiera
- **Valores**: precio, costo, descuento, cantidad, total
- **Transacciones**: ingreso, gasto, deuda, credito
- **Indicadores**: porcentaje, tasa

#### Información Temporal
- **Fechas**: fecha, fecha_nacimiento, fecha_registro, fecha_creacion, fecha_modificacion, fecha_actualizacion
- **Unidades**: hora, tiempo, mes, año, semana, dia

#### Identificadores Únicos
- **Claves**: id, identificador, folio, referencia, codigo, hash

#### Información Web y Técnica
- **Redes**: ip, ipv6, mac
- **Internet**: url, dominio, navegador, sistema_operativo

#### Texto y Descripciones
- **Contenido**: descripcion, comentario, notas, mensaje, resumen, detalle, observaciones

#### Misceláneos
- **Varios**: color, emoji, serie, numero, valor, cantidad_articulos, probabilidad, ranking, puntuacion, nivel, factor

## Interfaces de Usuario

### Interfaz de Línea de Comandos
La aplicación puede ejecutarse desde la terminal:

*python agents_dataset.py*

El usuario proporciona:
- Número de registros a generar
- Nombres de columnas separados por comas

Después de la generación, se presentan opciones para:
- Eliminar columnas
- Añadir nuevas columnas
- Finalizar el proceso

### Interfaz Web (Streamlit)
Una interfaz gráfica más amigable implementada con Streamlit:

*streamlit run app.py*

Características:
- Formulario para especificar número de registros y columnas
- Campos para definir valores mínimos/máximos para columnas numéricas
- Previsualización del dataset generado
- Opciones para modificar el dataset (eliminar/añadir columnas)
- Botones para descargar en formato CSV o Excel

## Implementación Técnica

### Detección de Tipos de Columnas
La función `detect_column_type()` utiliza varias estrategias para mapear nombres de columnas a métodos de Faker:

1. Compara con un diccionario de mapeos predefinidos
2. Busca palabras clave dentro del nombre de columna
3. Utiliza `difflib` para encontrar coincidencias aproximadas
4. Devuelve "text" como valor predeterminado

### Generación de Datos
La función `generate_synthetic_data()` crea un DataFrame de Pandas con datos sintéticos:

- Utiliza Faker con localización es_ES
- Respeta restricciones de valores mínimos/máximos para datos numéricos
- Genera datos apropiados según el tipo detectado para cada columna

### Sistema Multi-Agente
La función `start_conversation()` orquesta la interacción entre agentes:

1. Input_Agent procesa los requisitos del usuario
2. Validation_Agent verifica los parámetros
3. Column_Classifier_Agent clasifica las columnas
4. User_Proxy coordina el flujo de trabajo

## Limitaciones Actuales

- El mapeo de tipos de columnas está predefinido y podría no cubrir todos los casos de uso
- Las relaciones entre columnas no están soportadas (cada columna se genera independientemente)
- No hay validación exhaustiva de las entradas del usuario ni manejo robusto de errores
- La generación de datos está limitada a los tipos soportados por Faker

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Dataset_2_1.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Dataset_2_1_2.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_deploy.md
================================================
# Midas Deploy

## Descripción General
MIDAS Deploy es el componente MIDAS que genera interfaces de usuario personalizadas para modelos de ML. Utilizando tecnologías de IA, específicamente LLMs, MIDAS Deploy analiza modelos guardados en formato joblib y crea aplicaciones Streamlit que permiten a los usuarios interactuar con estos modelos sin necesidad de programación adicional.

El sistema utiliza AG2 para orquestar una conversación entre agentes de IA especializados que analizan el modelo, diseñan una interfaz y generan código ejecutable.

## Arquitectura Técnica

### Backend:
- **Lenguaje y Frameworks:** 
  - *Python* como lenguaje base
  - *AG2* para la orquestación de agentes de IA
  - *Scikit-learn* para procesamiento de modelos ML
  - *Joblib* para carga y manipulación de modelos

- **Componentes clave:**
  - *Model_Analyzer*: Agente especializado que analiza modelos ML y extrae información relevante (características, parámetros, estructura)
  - *UI_Designer*: Agente encargado de diseñar la interfaz de usuario basada en el análisis del modelo
  - *Code_Generator*: Agente que implementa código funcional de Streamlit basado en el diseño de UI
  - *User_Proxy*: Orquestador del flujo de trabajo entre agentes especializados
  - *process_joblib*: Función utilitaria para extraer información de archivos joblib
  - *load_json*: Función para cargar archivos JSON que contienen información sobre las features y la columna objetivo del modelo

- **Modelo LLM utilizado:** 
  - Meta-Llama/Llama-3.3-70B-Instruct-Turbo a través de la API de DeepInfra

- **Flujo de procesamiento:**
  1. Carga del modelo desde archivo joblib
  2. Extracción de metadatos (características, número de features, tipo de modelo)
  3. Carga de información desde un archivo JSON para definir features y la columna objetivo
  4. Análisis del modelo por agentes de IA
  5. Diseño de interfaz adaptada al modelo específico
  6. Generación de código Streamlit ejecutable
  7. Entrega del código para implementación

### Frontend:
- **Tecnología:** Aplicación web Streamlit
- **Componentes de UI:**
  - Cargador de archivos para modelos joblib
  - Cargador de archivos JSON con información sobre features
  - Campo de texto para descripción del modelo
  - Botón de generación de interfaz
  - Visualizador de código generado
  - Funcionalidad de descarga de código

## Funcionalidad
- Análisis automatizado de modelos de aprendizaje automático compatibles con scikit-learn
- Diseño inteligente de interfaces adaptadas a las especificaciones del modelo
- Carga de archivos JSON para definir features y la columna objetivo
- Generación de código Streamlit listo para usar
- Soporte para diversos tipos de modelos ML (clasificadores, regresores, pipelines)
- Creación de interfaces que tienen en cuenta los requisitos de entrada del modelo
- Capacidades de exportación y descarga de código
- Interacción con múltiples agentes de IA para optimizar la experiencia del usuario

## Guía de Uso
1. **Iniciar la aplicación:**
   - Ejecutar *streamlit run app.py*
   - Se abrirá la interfaz web de MIDAS Deploy en el navegador

2. **Cargar un modelo:**
   - Utilizar el cargador de archivos para subir un modelo .joblib
   - Cargar un archivo .json con información de features y la columna objetivo
   - Proporcionar una breve descripción del propósito del modelo (ej. "Predicción de satisfacción del cliente basada en datos demográficos")

3. **Generar la interfaz:**
   - Hacer clic en el botón "🚀 Iniciar generación de interfaz"
   - Esperar mientras el sistema analiza el modelo y genera la interfaz

4. **Implementar el resultado:**
   - Descargar el código generado mediante el botón "📥 Descargar código generado"
   - Guardar el código como archivo .py
   - Ejecutar *streamlit run generated_interface.py*
   - La interfaz personalizada para el modelo estará disponible a través del navegador

**Ejemplo práctico:**
Para un modelo que predice la probabilidad de una condición médica basada en edad, altura y peso:
- Cargar el archivo model.joblib
- Opcionalmente, cargar un archivo model_features.json con las features edad, altura y peso, y la columna objetivo diagnostico
- Describir como "Modelo de predicción de condición médica basado en factores biométricos"
- MIDAS Deploy generará una aplicación Streamlit con campos de entrada para edad, altura y peso
- La aplicación permitirá a los usuarios ingresar estos datos y obtener predicciones en tiempo real

## Implementación Técnica
MIDAS Deploy utiliza ConversableAgent de AG2 para crear agentes especializados:

1. **Model_Analyzer**: Analiza el modelo joblib y extrae metadatos como:
   - Tipo de modelo
   - Número de características
   - Nombres de características (si están disponibles)
   - Parámetros del modelo
   - Estructura del pipeline (si aplica)

2. **UI_Designer**: Diseña una interfaz adaptada al modelo basándose en:
   - El número de características requeridas
   - La descripción del propósito del modelo
   - El tipo de predicción (clasificación o regresión)

3. **Code_Generator**: Crea código Streamlit funcional que:
   - Carga correctamente el modelo joblib
   - Implementa campos de entrada para todas las características necesarias
   - Procesa adecuadamente los datos de entrada
   - Muestra el resultado de la predicción del modelo
   
4. **User_Proxy**: Orquesta la conversación entre los agentes, siguiendo un flujo secuencial de análisis, diseño y generación.

## Referencias y Recursos
- Documentación de AG2: https://docs.ag2.ai/docs/home/home
- Documentación de Streamlit: https://docs.streamlit.io/
- DeepInfra (para acceso a LLM): https://deepinfra.com/
- Scikit-learn (para modelos ML): https://scikit-learn.org/

## Limitaciones Actuales
- Solo soporta modelos compatibles con scikit-learn guardados en formato joblib
- Opciones de personalización limitadas para la interfaz generada
- Puede generar interfaces que necesiten ajustes menores para modelos complejos

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Deploy_8_2.png?raw=true)


================================================
File: Extra/Documentacion/docs/modules/midas_help.md
================================================
# Midas Help

## Descripción General
MIDAS Help constituye el componente de asistencia y documentación interactiva del sistema MIDAS, más a nivel de implementación. Se trata de un chatbot inteligente basado en una arquitectura LLM+RAG+Reranker que permite a los usuarios resolver dudas sobre la implementación del sistema MIDAS mediante lenguaje natural. 

Esta arquitectura utiliza una aproximación RAG mejorada, gracias a incorporar un reranker y un selector de LLM inteligente, pero sin llegar a características avanzadas como "Agentic RAG" o bases de datos vectoriales. Todo el flujo está basado en el framework Llama-Index.

## Arquitectura Técnica

### Backend
El backend está desarrollado en Python utilizando el framework Flask y se encarga de procesas las consultas de los usuarios. Los componentes principales son:

- **Clasificador de Preguntas (Fine-tuned BERT):** Un modelo BERT afinado que *analiza la pregunta del usuario (prompt)* y la clasifica en una de tres categorías:
    -   **Pregunta fácil:** Requiere una respuesta sencilla y directa.
    -   **Pregunta difícil:** Implica una respuesta más compleja y elaborada.
    -   **Pregunta no relacionada:** No tiene relación con la documentación de MIDAS. *En este caso, el sistema no genera una respuesta.*
- Framework **Llama Index** para la generación y gestión del índice documental.
- Modelo de **embeddings BGE-M3** de BAAI para la representación vectorial de los textos (tanto de la consulta como de los documentos). Para cada consulta, se seleccionan los 30 chunks mas relevantes según su similitud vectorial.
- **Reranker BGE V2 M3:** Este componente reordena los resultados obtenidos por la búsqueda inicial basada en embeddings.  El reranker evalúa la relevancia de cada documento recuperado *con respecto a la consulta específica del usuario*, utilizando un modelo de lenguaje más sofisticado que la simple comparación de embeddings. Esto ayuda a filtrar el ruido y a asegurar que los documentos más relevantes sean presentados al LLM para la generación de la respuesta final. Toma los 30 chunks que salen del proceso de embedding, y los "filtra" para pasarle al LLM solo los 10 realmente mas relevantes.
- **Selector de LLM:** Permite elegir entre diferentes modelos de lenguaje, o usar el modo automatico para usar un modelo u otro dependiendo de la clasificación del BERT Fine-tuneado:
    -   **Modo Automático:** Utiliza el clasificador de preguntas (BERT) para seleccionar el LLM óptimo (Llama o Gemini).
    -   **Llama 3.3 70B:** Un modelo de lenguaje eficiente, ideal para preguntas fáciles.  *(Usado por defecto en el modo automático si la pregunta se clasifica como "fácil").*
    -   **Gemini 2.0 Flash:** Un modelo más potente, diseñado para preguntas difíciles que requieren mayor capacidad de razonamiento. *(Usado por defecto en el modo automático si la pregunta se clasifica como "difícil").*

### Frontend
La interfaz de usuario está construida con HTML, JavaScript y Tailwind CSS, proporcionando una experiencia moderna y responsive.

## Funcionalidad
MIDAS Help facilita:

- Acceso interactivo a la documentación técnica del sistema
- Resolución de consultas sobre implementación y arquitectura
- Comprensión de la integración entre componentes
- Soporte tanto a desarrolladores como usuarios finales

## Guía de Uso
El sistema se ejecuta en local (Flask). Para iniciarlo:

1. Instala dependencias: `pip install -r requirements.txt`
2. Ejecuta: `python app.py`
3. Abre: `http://127.0.0.1:5008`

Los usuarios pueden realizar consultas como:

- "¿Qué componentes integran MIDAS?"
- "¿Qué tipo de gráficos soporta MIDAS Plot?"
- "¿Cuál es el flujo de interacción entre componentes en MIDAS Hub?"
- "¿Qué framework utiliza MIDAS Deploy para generar interfaces Streamlit?"

Las respuestas se presentan y renderizan en formato Markdown para optimizar la legibilidad.
Mientras el sistema procesa la consulta, se muestra información en tiempo real sobre la etapa actual del proceso (por ejemplo, "Clasificando pregunta...", "Extrayendo embeddings...", "Aplicando reranking...", "Redactando respuesta..."). Se visualiza en todo momento qué LLM fue usado para la respuesta, ya sea si lo escogió automáticamente o si el usuario forzó su uso a través del selector.

## Referencias y Recursos

- Aplicación (local): http://127.0.0.1:5008
- Repositorio: [github.com/warc0s/MIDAS](https://github.com/warc0s/MIDAS)
- Sitio Web Llama Index: [llamaindex.ai](https://www.llamaindex.ai)

## Limitaciones Actuales

La implementación actual no incluye:

- Sistema de RAG Agéntico
- Bases de datos vectoriales para optimización de la velocidad de búsqueda

La expansión de estas capacidades fue contemplada, pero no implementada por falta de tiempo.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Help_7_3.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Help_Full_Captura.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_plot.md
================================================
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
    - `model`: Modelo LLM (en este caso, `"gemini/gemini-2.0-flash"`) usado para la generación del código.

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
  - **`_extraer_base64`:** Analiza la salida del sandbox para identificar y extraer el string base64 correspondiente a la imagen (se asume que comienza con `iVBORw0KGgo` - así comienza el base64 de cualquier png).

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

- **[CrewAI](https://www.crewai.com) (En su version Flow):** Framework utilizado para orquestar el flujo de generación y ejecución del código.
- **[Streamlit](https://streamlit.io):** Framework para la creación de la interfaz web interactiva.
- **[E2B Sandbox](https://e2b.dev):** Entorno de ejecución seguro para la ejecución del código generado.

---

## 6. Limitaciones Actuales

- **Dependencia de la Calidad del Prompt:** La precisión del resultado depende en gran medida de la claridad y calidad del prompt proporcionado por el usuario.
- **Formatos de Salida Limitados:** Actualmente, la salida se limita a imágenes en formato PNG codificadas en base64.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Plot_4_1.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Plot_Full_Captura.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_test.md
================================================
# Midas Test

## Descripción General

MIDAS Test es el componente MIDAS especializado en la evaluación exhaustiva de modelos de machine learning almacenados en formato joblib. Su propósito principal es analizar la calidad, rendimiento y robustez de modelos ML mediante una arquitectura de agentes conversacionales basados en IA.

El sistema utiliza Large Language Models (LLM) para coordinar múltiples agentes especializados que evalúan diferentes aspectos de los modelos ML. MIDAS Test implementa un enfoque de colaboración multi-agente donde cada agente aporta su perspectiva especializada para generar un informe completo.

MIDAS Test se basa en el framework AG2 para la gestión de agentes conversacionales y utiliza Streamlit para proporcionar una interfaz de usuario accesible.

## Arquitectura Técnica

### Backend:

- **Lenguaje y Bibliotecas:** 
  - Python 3.x
  - AG2 para la gestión de agentes IA
  - Scikit-learn para manipulación de modelos ML
  - Joblib para carga/guardado de modelos
  - DeepInfra API para acceder a modelos LLM
  - deep_translator para traducir informes al español

- **Componentes Clave:**
  - *Agentes Especializados*:
    - **Model Analyzer**: Examina la estructura y características del modelo ML.
    - **Performance Tester**: Evalúa métricas de rendimiento como latencia, uso de memoria y CPU.
    - **Robustness Checker**: Verifica la resistencia del modelo ante entradas anómalas.
    - **Output Validator**: Confirma la validez y formato de las predicciones del modelo.
  
  - *Gestor de Comunicación*:
    - **GroupChat**: Facilita la comunicación entre agentes.
    - **GroupChatManager**: Coordina el flujo de la conversación y turno de los agentes.
  
  - *Modelo LLM Base*:
    - Utiliza *meta-llama/Llama-3.3-70B-Instruct-Turbo* a través de la API de DeepInfra.
    - Configuración personalizable de temperatura y seed para resultados consistentes.
  
  - *Módulos de Procesamiento*:
    - **load_model**: Carga modelos joblib y mide tiempo de carga.
    - **check_model_validity**: Verifica si el modelo es compatible con Scikit-learn.
    - **measure_latency**: Evalúa tiempos de respuesta en diferentes tamaños de batch.
    - **measure_memory_usage**: Mide el uso de memoria.
    - **measure_memory_and_cpu_during_prediction**: Evalúa el uso de recursos durante predicciones.
    - **validate_predictions**: Verifica la consistencia y formato de las predicciones.
    - **check_robustness**: Prueba comportamiento ante valores nulos, extremos y tipos incorrectos.
    - **translate_to_spanish**: Traduce el informe al español.
    - **generate_markdown_report**: Compila los hallazgos en formato Markdown estructurado.

- **Flujo de Procesamiento**:
  1. Carga del modelo joblib.
  2. Validación inicial del modelo (compatibilidad con Scikit-learn).
  3. Generación de datos de muestra para pruebas.
  4. Ejecución de pruebas de rendimiento, robustez y validación.
  5. Compilación de métricas y resultados.
  6. Activación de agentes IA para análisis especializado.
  7. Generación de informe final en formato Markdown en español.

### Frontend:

- **Tecnologías:**
  - Streamlit para la interfaz web interactiva
  - Componentes UI de Streamlit: file_uploader, expanders, download_button

- **Estructura de la Interfaz:**
  - Sección de carga de archivos
  - Panel de progreso y estado
  - Visualización de resultados en secciones expandibles
  - Botones para iniciar evaluación y descargar informes

## Funcionalidad

- **Análisis de Modelos ML**: Evalúa múltiples aspectos del modelo incluyendo validez, rendimiento y robustez.

- **Métricas de Rendimiento**: 
  - Tiempo de carga del modelo
  - Uso de memoria durante predicciones
  - Utilización de CPU
  - Latencia en diferentes tamaños de batch (1, 100, 1000, 10000)
  - Throughput (predicciones por segundo)

- **Pruebas de Robustez**:
  - Comportamiento ante valores nulos
  - Resistencia a valores fuera de rango
  - Manejo de tipos de datos incorrectos
  - Consistencia de predicciones

- **Validación de Salidas**:
  - Verificación de formato correcto (array NumPy)
  - Validación de rangos de valores
  - Comprobación de suma de probabilidades igual a 1 (cuando aplica)

- **Recomendación Automatizada**: Clasificación del modelo como "APTO" o "NO APTO" basada en la validez del modelo y la consistencia de sus predicciones.

- **Reporte Markdown**: Generación automática de documentación estructurada en español con los hallazgos y recomendaciones.

## Guía de Uso

### A través de la Interfaz Web (Streamlit):

1. Inicie la aplicación ejecutando:
   *streamlit run app.py*

2. En la interfaz web, haga clic en el cargador de archivos y seleccione el modelo joblib a evaluar.

3. Una vez cargado el modelo, pulse el botón "🔄 Iniciar Evaluación con los Agentes" para comenzar el análisis.

4. El sistema mostrará un mensaje indicando que la evaluación está en proceso.

5. Después de unos 90 segundos, pulse "📄 Finalizar Análisis y Descargar Reporte" para ver y descargar los resultados.

6. Explore los resultados en las secciones expandibles:
   - "📌 Información del Modelo": Datos básicos como tiempo de carga y tamaño
   - "📈 Métricas de Rendimiento": Detalles sobre uso de recursos
   - "⚠️ Pruebas de Robustez": Resultados de las pruebas de resistencia

7. Descargue el informe completo en formato Markdown utilizando el botón "⬇️ Descargar Reporte".

### Mediante Línea de Comandos:

1. Ejecute el script principal:
   *python agents_test.py*

2. Cuando se solicite, ingrese la ruta completa al archivo joblib que desea analizar.

3. El sistema ejecutará automáticamente todas las pruebas y generará un informe en el archivo "informe_analisis_modelo.md".

### Ejemplo de Salida:

El reporte generado contendrá secciones como:

# 📊 Informe de Análisis del Modelo
**Generado el:** 2025-03-02 15:30:45

---

## 🔍 Resumen del Modelo
[Información general sobre el modelo y sus características]

## ⚙️ Métricas de Rendimiento
[Detalles sobre rendimiento, memoria y CPU]

## ⏳ Análisis de Latencia
[Análisis de tiempos de respuesta]

## ✅ Validez de Predicciones
[Validación de las salidas del modelo]

## 🛡️ Pruebas de Robustez
[Resultados de pruebas de resistencia]

## 📌 Recomendación Final
**APTO**

## 🔧 Sugerencias de Mejora
[Recomendaciones para mejorar el modelo]

## Limitaciones Actuales

- El componente está optimizado para modelos Scikit-learn y puede tener limitaciones con otros frameworks de ML.
- Las pruebas de robustez son básicas y no cubren todos los escenarios posibles de entrada anómala.
- La evaluación actual se centra en la validez del modelo y consistencia de predicciones, sin métricas específicas de calidad predictiva.
- El rendimiento de los agentes puede variar dependiendo de la calidad de las respuestas del LLM utilizado.
- La traducción automática al español puede contener imprecisiones en terminología técnica.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Test_Interfaz_6_2.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Test_Reporte_6_5.png?raw=true)

================================================
File: Extra/Documentacion/docs/modules/midas_touch.md
================================================
# Midas Touch

## Descripción General

Midas Touch es el componente que automatiza el proceso completo desde la carga de datos hasta el entrenamiento de modelos. El sistema toma como entrada un dataset y una descripción en lenguaje natural de lo que se desea predecir, y genera automáticamente un modelo entrenado, documentación detallada y métricas de rendimiento.

Este componente utiliza tecnologías de IA, específicamente agentes y modelos de lenguaje grande (LLM) de Gemini (Gemini 2.0 Flash) para interpretar la intención del usuario y guiar el proceso de análisis. Implementa un enfoque basado en múltiples agentes especializados que colaboran para realizar todas las etapas del flujo de trabajo de machine learning.

Midas Touch es, a grandes rasgos, una implementación de agentes construido sobre Python "Vanilla" y bibliotecas estándar de ciencia de datos, destacando por su capacidad de autoorganización y recuperación ante fallos.

## Arquitectura Técnica

### Backend:

El backend de Midas Touch está implementado en Python y utiliza un diseño modular basado en agentes especializados:

- **Framework central**: 
  - `AICortex`: *Clase principal* que coordina el flujo de trabajo completo.
  - `OperationalContext`: *Memoria compartida y centro de coordinación* que mantiene el estado global del workflow y permite a los agentes acceder y modificar información que será utilizada por otros agentes en etapas posteriores. Contiene exactamente los valores:
 
***
    # Etapa actual del workflow
    'current_stage': WorkflowStage.DATA_LOADING,
    
    # Datos del dataset
    'dataset': None,            # Dataset cargado
    'target_column': None,      # Columna objetivo identificada
    'data_statistics': {},      # Estadísticas del dataset (ej. distribuciones, valores nulos, etc.)
    
    # Información del problema a resolver
    'problem_type': None,       # Tipo de problema: clasificación o regresión
    
    # Información del proceso y validaciones
    'validation_reports': {},   # Reportes generados durante la validación del modelo
    'pipeline_steps': [],       # Lista de pasos aplicados en el pipeline de procesamiento
    'model_versions': [],       # Versiones del modelo generadas o actualizadas durante el workflow
    
    # Gestión de errores y contingencias
    'error_log': [],            # Registro de errores ocurridos
    'retry_count': 0,           # Contador de reintentos en caso de fallos
    'fallback_activated': False, # Indicador que señala si se activó el modo fallback
    'performance_metrics': {}   # Métricas de rendimiento del modelo (ej. precisión, recall, etc.)
***

- **Agentes especializados**:
  - `IntentAgent`: *Analiza la descripción del usuario* utilizando un LLM para determinar el objetivo del análisis y el tipo de problema (clasificación/regresión).
  - `DataGuardianAgent`: *Analiza el dataset* e identifica la columna objetivo mencionada explícitamente en el prompt del usuario.
  - `DataAlchemistAgent`: *Realiza la limpieza y transformación de datos* adaptándose al tipo de problema y características de los datos.
  - `ModelShamanAgent`: *Selecciona, entrena y evalúa modelos* automáticamente, con soporte completo para problemas multiclase.
  - `OracleAgent`: *Valida la calidad* del flujo completo y los resultados.
  - `NotebookScribeAgent`: *Documenta todo el proceso* en formato Jupyter Notebook.
  - `PhoenixAgent`: *Implementa recuperación ante fallos* con estrategias adaptativas.

- **Sistema de enumeraciones y tipos de datos**:
  - `WorkflowStage`: Enumera las etapas del workflow (DATA_LOADING, DATA_VALIDATION, FEATURE_ENGINEERING, MODEL_TRAINING, MODEL_VALIDATION, DEPLOYMENT, ERROR_HANDLING).
  - `ErrorSeverity`: Define los niveles de gravedad de errores (INFO, WARNING, CRITICAL, FATAL).
  - `ErrorLogEntry`: Estructura de datos para registrar errores con timestamp, agente, mensaje, nivel, operación, intentos y traceback.
  - `ModelMetadata`: Estructura para metadatos del modelo con información sobre tipo, columna objetivo, características, métricas, estadísticas, transformaciones y mapeo de clases.

- **Tecnologías clave**:
  - Google Generative AI (Gemini): Usado para interpretación semántica y análisis de intención. Usa el LLM Gemini-2.0-Flash.
  - pandas: Para manipulación y análisis de datos.
  - scikit-learn: Para modelos de machine learning y preprocesamiento.
  - joblib: Para serialización de modelos.

- **Flujo de datos**:
  1. El usuario proporciona un dataset y una descripción del objetivo.
  2. IntentAgent extrae directamente la columna objetivo y el tipo de problema (clasificación/regresión) del prompt a través de una consulta al LLM.
  3. DataGuardianAgent identifica y selecciona la columna objetivo mencionada en el dataset.
  4. DataAlchemistAgent preprocesa los datos mediante pipelines adaptativas según el tipo de datos (tratando numéricas y categóricas de forma diferente).
  5. ModelShamanAgent selecciona entre RandomForest o GradientBoosting (según el tipo de problema y características del dataset), lo entrena y valida.
  6. El modelo se serializa junto con metadatos que incluyen el mapeo de clases para problemas de clasificación.
  7. NotebookScribeAgent genera documentación detallada y OracleAgent valida las métricas de rendimiento.

### Frontend:

- **Tecnología**: Streamlit para la interfaz web interactiva.
  
- **Componentes principales**:
  - *Panel de carga de datos*: Para subir archivos CSV, Excel, Parquet o JSON.
  - *Campo de texto*: Para describir la tarea de ML en lenguaje natural.
  - *Visor de dataset*: Muestra una vista previa de los datos cargados.
  - *Panel de métricas*: Visualiza el rendimiento del modelo entrenado.
  - *Visualización de mapeo de clases*: Muestra la correspondencia entre valores numéricos y etiquetas originales.
  - *Sistema de pestañas*: Para navegar entre descargas y logs.
  - *Interfaz de descarga*: Para obtener el modelo, documentación y reportes.

- **Personalización de interfaz**:
  - CSS personalizado para mejorar la experiencia visual.
  - Tarjetas interactivas para métricas y descargas.
  - Estilos universales compatibles con modos claro y oscuro de Streamlit.

## Funcionalidad

Midas Touch ofrece las siguientes capacidades principales:

- **Análisis automático de datasets**:
  - Carga y análisis exploratorio automático de datos.
  - Identificación directa de la columna objetivo mencionada en la descripción del usuario.
  - Detección explícita del tipo de problema (clasificación/regresión) desde el prompt.
  - Validación de calidad de datos y estrategias de mitigación.
  - Análisis de tipos de datos, valores únicos, y valores faltantes por columna.
  - Detección de columnas categóricas con alta cardinalidad (>100 valores únicos).

- **Preprocesamiento adaptativo**:
  - Manejo automático de valores faltantes según el tipo de datos (mediana para numéricas, moda para categóricas).
  - Eliminación de columnas con más del 70% de valores faltantes.
  - Detección y procesamiento de fechas, extrayendo componentes útiles (año, mes, día, día de la semana).
  - Codificación de variables categóricas (OneHotEncoder) y escalado de variables numéricas (StandardScaler).
  - Construcción de pipelines de transformación reproducibles con sklearn.
  - Manejo especial para columnas con formato de fecha detectadas automáticamente.

- **Selección y entrenamiento inteligente de modelos**:
  - Utilización del tipo de problema especificado en el prompt (clasificación/regresión).
  - Soporte robusto para problemas de clasificación multiclase con mapeo automático de etiquetas.
  - Selección entre RandomForest y GradientBoosting según las características del dataset:
    - RandomForest: Para datasets pequeños (<1000 muestras) o con muchas características (>50)
    - GradientBoosting: Para datasets más grandes con pocas características
  - Entrenamiento con validación cruzada (5-fold) para estimaciones robustas.
  - Cálculo de métricas específicas para cada tipo de problema:
    - Clasificación: accuracy, f1 (weighted), precision, recall
    - Regresión: r2, MSE, RMSE
  - Estratificación automática cuando es posible (para problemas de clasificación).
  - Manejo adecuado de clases minoritarias durante la validación.
  - Modelos fallback (DummyClassifier/DummyRegressor) en caso de problemas graves.

- **Documentación y explicabilidad**:
  - Generación de un notebook Jupyter detallando todo el proceso.
  - Documentación paso a paso de cada decisión tomada por el sistema.
  - Inclusión de código reproducible para todas las operaciones.
  - Visualización de métricas y resultados del modelo.
  - Documentación explícita del mapeo entre valores numéricos y etiquetas originales en problemas de clasificación.
  - Organización del notebook por secciones lógicas (carga, exploración, preprocesamiento, entrenamiento, evaluación).
  - Cada etapa incluye tanto explicaciones en markdown como el código Python correspondiente.

- **Recuperación ante fallos**:
  - Sistema resiliente con recuperación automática en diferentes etapas.
  - Decorador `resilient_agent` para funciones con reintentos automáticos y backoff exponencial.
  - Estrategias específicas según el tipo de error detectado:
    - Errores en DataGuardianAgent: Selección de columna alternativa (última columna del dataset)
    - Errores en DataAlchemist: Simplificación del preprocesamiento
    - Errores en ModelShaman: Utilización de modelos fallback más simples
  - Logging detallado para diagnóstico y depuración.
  - Supresión inteligente de advertencias irrelevantes (como UndefinedMetricWarning).
  - Captura y manejo de excepciones en cada etapa crítica.

- **Sistema de logging y seguimiento**:
  - Registro detallado de cada paso del proceso.
  - Estructura multinivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
  - Captura de stacktraces para errores críticos.
  - Registro temporal de cada operación con timestamps.
  - Guardado de logs tanto en archivo como en UI (interfaz Streamlit).
  - Registro enriquecido con contexto sobre el agente y la operación.

- **Serialización y persistencia**:
  - Guardado del modelo entrenado en formato .joblib.
  - Serialización de metadatos complejos a JSON, con manejo especial para tipos de datos NumPy.
  - Función `convert_to_serializable` para transformar objetos NumPy y otros tipos no serializables.
  - Creación de informes de rendimiento en formato texto.
  - Generación de archivos ZIP con todos los resultados para facilitar la descarga.
  - Versionado de modelos con timestamps.

## Guía de Uso

### Uso desde la interfaz Streamlit:

1. **Inicio de la aplicación**:
   *streamlit run Midas_Touch_Streamlit.py*

2. **Carga de datos**:
   - En el panel lateral, haz clic en "Cargar archivo de datos".
   - Selecciona un archivo en formato CSV, Excel, Parquet o JSON.
   - Se mostrará una vista previa del dataset en el panel principal.
   - También verás un resumen de información sobre las columnas (tipos, valores únicos, valores faltantes).

3. **Descripción de la tarea**:
   - En el campo "Describir tarea de ML", escribe una descripción clara de lo que deseas predecir.
   - **Importante**: Especifica explícitamente la columna objetivo y el tipo de problema.
   - Ejemplos:
     - "Predecir la columna precio de las casas, problema de regresión"
     - "Clasificar clientes según la columna abandono, problema de clasificación"
     - "Determinar si un correo es spam o no en la columna categoría, problema de clasificación"

4. **Iniciar procesamiento**:
   - Haz clic en el botón "Iniciar Procesamiento".
   - El sistema comenzará a analizar los datos y mostrará el progreso en tiempo real.
   - Este proceso puede tomar varios minutos dependiendo del tamaño del dataset.

5. **Revisar resultados**:
   - Una vez completado el proceso, se mostrarán las métricas de rendimiento del modelo.
   - Para problemas de clasificación, se mostrará el mapeo entre valores numéricos y etiquetas originales.
   - Navega por las pestañas para ver:
     - **Descargas**: Opciones para descargar el modelo, notebook y reportes.
     - **Logs**: Registro detallado de todas las operaciones realizadas.

6. **Descargar resultados**:
   - En la pestaña "Descargas", tienes varias opciones:
     - **Todo en uno**: Archivo ZIP con todos los archivos generados.
     - **Notebook**: Documentación en formato .ipynb.
     - **Modelo entrenado**: Archivo .joblib con el modelo serializado.
     - **Reporte de rendimiento**: Métricas detalladas del modelo y mapeo de clases.

### Uso desde línea de comandos:

También puedes utilizar Midas Touch directamente desde la línea de comandos:

*python Midas_Touch_V2_CLI.py*

El sistema te pedirá una descripción de la tarea de ML y procesará el archivo de datos configurado en `CONFIG['DATA_FILE']`. Al finalizar, mostrará un resumen en la consola y guardará todos los archivos generados en las ubicaciones especificadas en CONFIG.

### Configuración del sistema:

El sistema incluye un diccionario `CONFIG` con los siguientes parámetros ajustables:

- `API_KEY_ENV_VAR`: Nombre de la variable de entorno para la API key de Google.
- `MODEL_NAME`: Modelo de Gemini a utilizar (por defecto, 'gemini-2.0-flash').
- `LOG_FILE`: Ruta del archivo de log.
- `LOG_LEVEL`: Nivel de logging (INFO, DEBUG, etc.).
- `DATA_FILE`: Archivo de datos predeterminado.
- `MODEL_DIR`: Directorio para guardar modelos.
- `NOTEBOOK_FILE`: Ruta del notebook generado.
- `RETRIES`: Número de reintentos para diferentes operaciones.
- `MIN_ROWS`: Mínimo de filas recomendado para el dataset.
- `MAX_MISSING_RATIO`: Ratio máximo permitido de valores faltantes.
- `MIN_FEATURE_VARIANCE`: Varianza mínima requerida para características.
- `DEFAULT_TEST_SIZE`: Tamaño predeterminado del conjunto de prueba.
- `RANDOM_SEED`: Semilla para reproducibilidad.
- `PERFORMANCE_THRESHOLDS`: Umbrales mínimos de rendimiento para modelos.

### Ejemplos de entrada/salida:

**Entrada**:
- Dataset: archivo CSV con datos de clientes de un banco
- Descripción: "Predecir si un cliente abandonará el servicio en la columna churn, problema de clasificación"

**Salida**:
- Modelo de clasificación (RandomForest o GradientBoosting) serializado como .joblib
- Metadatos con mapeo de clases (ej: 0 → "No", 1 → "Sí")
- Notebook con documentación detallada del proceso
- Métricas como accuracy, precision, recall y F1-score (weighted para multiclase)
- Reportes en formato texto y JSON con detalles del modelo
- Archivo ZIP con todos los resultados

Durante el proceso, se ofrece información en tiempo real sobre:
- Etapa actual del workflow
- Progreso del procesamiento
- Alertas y mensajes de validación

## Referencias y Recursos

- **Código fuente**:
  - [Midas_Touch_V2_CLI.py](https://github.com/warc0s/MIDAS/blob/main/4midas_touch/Midas_Touch_V2_CLI.py) - Implementación principal
  - [Midas_Touch_Streamlit.py](https://github.com/warc0s/MIDAS/blob/main/4midas_touch/Midas_Touch_Streamlit.py) - Interfaz web

- **Tecnologías principales utilizadas**:
  - [Google Generative AI (Gemini)](https://ai.google.dev/docs) - Para las llamadas a Gemini Flash
  - [scikit-learn](https://scikit-learn.org/) - Para trabajar con los modelos de machine learning
  - [pandas](https://pandas.pydata.org/) - Para la manipulación de datos
  - [Streamlit](https://streamlit.io/) - Para la interfaz web
  - [joblib](https://joblib.readthedocs.io/) - Para serialización de modelos

- **Documentación relacionada**:
  - [Sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
  - [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
  - [Jupyter Notebook Format](https://nbformat.readthedocs.io/en/latest/)
  - [Streamlit Components](https://docs.streamlit.io/library/components)

## Limitaciones Actuales

- **Soporte de modelos ML**: Actualmente solo implementa modelos de scikit-learn, específicamente RandomForest y GradientBoosting (no usa búsqueda de hiperparámetros).
- **Soporte de modelos LLM**: Ahora mismo usa exclusivamente Gemini 2.0 Flash. En un futuro, podría usarse LiteLLM y definir el modelo + api_key en el .env.
- **Tamaño de datasets**: Está optimizado para datasets de tamaño pequeño a mediano (recomendable hasta ~25K filas). Datasets muy grandes pueden causar problemas de rendimiento.
- **Complejidad de intención**: Aunque el sistema extrae directamente la columna objetivo y el tipo de problema del prompt, descripciones ambiguas pueden llevar a interpretaciones incorrectas.
- **Preprocesamiento especializado**: Algunas transformaciones de dominio específico (como procesamiento avanzado de texto, embeddings, o series temporales) no están implementadas.
- **Explicabilidad de modelos**: No incluye herramientas avanzadas de interpretabilidad como SHAP o LIME.
- **Modo interactivo**: No implementa un modo "semi-manual" donde el sistema consulte al usuario sobre decisiones clave (ej: tratamiento de outliers, imputación de valores).
- **Visualizaciones**: En el notebook generado no se incluyen gráficas que podrían ser relevantes (importancia de características, matriz de correlación, etc.).
- **Umbrales predeterminados**: Los umbrales de rendimiento y otros parámetros están codificados en CONFIG y no son ajustables dinámicamente desde la interfaz streamlit.
- **Validación de entrada**: No hay validación avanzada del texto introducido por el usuario, lo que puede afectar la interpretación si no se sigue el formato recomendado.
- **Limitaciones de robustez**: Puede tener dificultades con estructuras de datos muy complejas o tipos de datos no estándar.

## Capturas:

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Interfaz_6_0.png?raw=true)

![Midas Imagen](https://github.com/warc0s/MIDAS/blob/main/Extra/Imagenes/Midas_Touch_Preprocesamiento_3_2.png?raw=true)
