# Componente Midas Dataset

## Descripción General

MIDAS Dataset es el componente diseñado para la generación automatizada de datasets sintéticos a través de un enfoque basado en múltiples agentes conversacionales. Su propósito principal es facilitar la creación de datos de prueba realistas para desarrollo y testing, sin necesidad de exponer información sensible o real.

El sistema utiliza tecnologías de IA, específicamente Large Language Models (LLM), para coordinar la interpretación de solicitudes de usuarios, validar parámetros y clasificar columnas de datos. Se basa en el framework AG2 (un fork mejorado de AutoGen v0.2) para la implementación de los agentes conversacionales y aprovecha la biblioteca Faker para la generación de datos sintéticos.

## Arquitectura Técnica

### Backend
- **Lenguaje y Frameworks:** 
 - Python como lenguaje principal
 - AG2 para el sistema multi-agente
 - Pandas para manipulación de datos
 - Faker para generación sintética

- **Componentes Clave:**
 - *Input Agent:* Recibe y procesa las solicitudes iniciales del usuario.
 - *Validation Agent:* Verifica que los parámetros proporcionados sean válidos.
 - *Column Classifier Agent:* Clasifica los nombres de columnas para mapearlos a atributos de Faker.
 - *User Proxy:* Coordina el flujo de trabajo entre los diferentes agentes.
 - *Generador de Datos:* Sistema que utiliza Faker para crear registros sintéticos basados en la clasificación de columnas.
 - *Sistema de Detección de Tipos:* Analiza nombres de columnas para inferir el tipo de datos a generar.

- **Flujo de Procesamiento:**
 1. El usuario proporciona parámetros (número de registros y nombres de columnas).
 2. El sistema orquesta una conversación entre los agentes para procesar la solicitud.
 3. Se detecta automáticamente el tipo de datos para cada columna.
 4. Se genera el dataset sintético utilizando Faker.
 5. El resultado se guarda en un archivo CSV.

- **Modelos Utilizados:**
 - Meta Llama 3.3 70B Instruct Turbo (a través de DeepInfra API)

## Funcionalidad

- Genera conjuntos de datos sintéticos con un número especificado de registros.
- Detecta automáticamente el tipo de datos apropiado según el nombre de la columna.
- Soporta múltiples tipos de datos (nombres, direcciones, fechas, valores numéricos, etc.).
- Localización específica para datos en español (es_ES).
- Exportación automática de resultados a formato CSV.
- Facilita la creación rápida de datasets para testing, desarrollo o demostración.
- Dirigido principalmente a desarrolladores, analistas de datos y equipos de QA.

## Guía de Uso

### Ejecución Básica

1. Asegúrate de tener configurada la variable de entorno `DEEPINFRA_KEY` con tu clave de API.
2. Ejecuta el script principal:
  *python agents_dataset.py*
3. Sigue las instrucciones en consola:
  - Ingresa el número de registros deseados
  - Proporciona los nombres de las columnas separados por comas

### Ejemplos de Uso

**Entrada:**
Número de registros: 50
Nombres de las columnas: nombre, apellido, edad, ciudad, correo

**Salida:**
- Se generará un archivo `synthetic_data.csv` con 50 registros y las columnas especificadas.
- Durante el proceso, se mostrará información sobre los tipos de columnas detectados y una vista previa del dataset generado.

## Referencias y Recursos

- [AG2 Framework](https://ag2.ai) - Framework para sistemas multi-agente basados en LLM (fork mejorado de AutoGen)
- [Pandas](https://pandas.pydata.org/) - Biblioteca para análisis y manipulación de datos
- [Faker](https://faker.readthedocs.io/) - Generador de datos falsos para múltiples idiomas
- [Meta Llama 3.3](https://ai.meta.com/llama/) - Familia de modelos de lenguaje utilizados por el sistema
- [DeepInfra API](https://deepinfra.com/) - Servicio API que proporciona acceso a los modelos LLM

## Limitaciones Actuales

- El mapeo de tipos de columnas está predefinido y podría no cubrir todos los casos de uso.
- No permite la personalización avanzada de parámetros para la generación de datos (rangos, formatos, etc.).
- No hay validación exhaustiva de las entradas del usuario ni manejo robusto de errores.
- La generación de datos está limitada a los tipos soportados por Faker.
- El sistema actual no admite relaciones entre columnas o restricciones complejas.
- La interfaz es estrictamente por línea de comandos, sin una interfaz gráfica.

Posibles mejoras futuras incluyen la adición de una interfaz web, soporte para múltiples idiomas y opciones de personalización más avanzadas.
