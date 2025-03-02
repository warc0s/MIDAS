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