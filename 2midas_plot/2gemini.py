import os
import sys
import logging
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Configuración de logging para un seguimiento más detallado
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------------------------------------------------
# Carga de variables de entorno y configuración de la API
# ---------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("google_key")

if not API_KEY:
    logging.error("❌ No se encontró la variable de entorno 'google_key'.")
    sys.exit("Por favor, configure 'google_key' en su archivo .env o variable de entorno.")

genai.configure(api_key=API_KEY)

# ---------------------------------------------------------------------
# Función para cargar el dataset de forma robusta
# ---------------------------------------------------------------------
def cargar_dataset(file_path='dataset.csv'):
    """
    Carga un archivo CSV en un DataFrame de pandas.
    Incluye manejo de errores y validaciones básicas.
    """
    if not os.path.isfile(file_path):
        logging.error(f"❌ No se encuentra el archivo: {file_path}")
        sys.exit(f"Archivo {file_path} no encontrado. Verifique la ruta o el nombre.")

    try:
        df = pd.read_csv(file_path)
        logging.info("✅ Dataset cargado correctamente.")
        return df
    except pd.errors.EmptyDataError:
        logging.error("❌ El archivo CSV está vacío.")
        sys.exit("El archivo CSV está vacío, no hay datos para procesar.")
    except pd.errors.ParserError as e:
        logging.error(f"❌ Error al parsear el archivo CSV: {str(e)}")
        sys.exit("Error al parsear el CSV, revise el formato.")
    except Exception as e:
        logging.error(f"❌ Error desconocido leyendo el dataset: {str(e)}")
        sys.exit(f"Error desconocido leyendo el dataset: {str(e)}")

# ---------------------------------------------------------------------
# Función para generar código Python de visualización con el modelo LLM
# ---------------------------------------------------------------------
def generar_codigo(df, user_input):
    """
    Envía un prompt al modelo generativo para que retorne SOLO código Python
    válido de visualización (matplotlib), usando el DataFrame proporcionado
    como contexto. Incluye manejo de errores y validaciones básicas.
    """
    # Armamos un contexto lo más completo posible sobre el dataset
    contexto = f"""
Columnas disponibles: {df.columns.tolist()}
Tipos de datos: {df.dtypes.to_dict()}
Muestra de datos (3 filas):
{df.head(3).to_string(index=False)}
Valores nulos por columna:
{df.isnull().sum().to_dict()}
"""

    prompt = f"""
Eres un experto en visualización de datos. Genera SOLO código Python válido que:
1. Lea dataset.csv usando pandas
2. Cree visualizaciones con matplotlib. SOLO matplotlib, no uses seaborn
3. Guarde como 'grafica.png'
4. Maneje tipos de datos y valores nulos
5. Incluya títulos y etiquetas descriptivas
6. Sea robusto y maneje posibles errores
7. No incluya ningún texto adicional, solo código Python válido
8. El grafico sea visual y legible, aunque se soliciten muchos datos

Solicitud del usuario: {user_input}

Contexto dataset:
{contexto}

Formato requerido:
```python
import pandas as pd
import matplotlib.pyplot as plt
# [Inicio del código]
...
# [Fin del código]
```"""

    try:
        # Configuramos el modelo con temperatura 0 para obtener respuestas deterministas
        model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config={"temperature": 0}
        )
        logging.info("🌀 Enviando prompt al modelo generativo...")

        # Generamos la respuesta
        respuesta = model.generate_content(prompt).text
        logging.info("🌀 Respuesta generada por el modelo.")

        # Intentamos extraer solo el código de Python entre los delimitadores
        if "```python" in respuesta:
            try:
                codigo = respuesta.split("```python")[-1].split("```")[0].strip()
            except IndexError:
                logging.error("❌ No se encontraron delimitadores adecuados en la respuesta del modelo.")
                sys.exit("La respuesta del modelo no contiene el formato de código esperado.")
        else:
            logging.error("❌ La respuesta no contiene bloque de código en formato ```python```.")
            sys.exit("La respuesta del modelo no contiene bloque de código en formato ```python```.")

        # Validación mínima del código (ejemplo: debe comenzar con 'import')
        if not codigo.lower().startswith("import"):
            logging.error("❌ El código generado no inicia con 'import'. Posiblemente no es válido.")
            sys.exit("El código generado no parece ser válido. Revise la respuesta del modelo.")

        logging.info("✅ Código de visualización obtenido exitosamente.")
        return codigo

    except Exception as e:
        logging.error(f"❌ Error generando código con el modelo: {str(e)}")
        sys.exit(f"Error generando código con el modelo: {str(e)}")

# ---------------------------------------------------------------------
# Función para ejecutar el código generado y guardar el gráfico
# ---------------------------------------------------------------------
def ejecutar_y_guardar(codigo):
    """
    Ejecuta el código generado de forma aislada para crear el gráfico
    y lo guarda en el archivo 'grafica.png'.
    Incluye manejo de errores para evitar la interrupción total del programa.
    """
    # Guardar el código en un archivo Python, para auditoría y depuración
    codigo_file = 'codigo.py'
    logging.info("💾 Guardando el código en archivo para auditoría y ejecución.")
    try:
        with open(codigo_file, 'w', encoding='utf-8') as f:
            f.write(codigo)
    except Exception as e:
        logging.error(f"❌ Error guardando el código en {codigo_file}: {str(e)}")
        sys.exit(f"No se pudo guardar el código en {codigo_file}: {str(e)}")

    # Creamos un espacio de nombres aislado para 'exec'
    exec_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'os': os,
        'sns': None,
        'plt': None,
        'np': None,
        'logging': logging,
        'grafica_path': 'grafica.png'
    }

    # Ejecutamos el código
    logging.info("🚀 Ejecutando el código en un espacio aislado.")
    try:
        exec(codigo, exec_globals)
    except Exception as e:
        logging.error(f"❌ Error ejecutando el código generado: {str(e)}")
        sys.exit(f"Hubo un error al ejecutar el código generado: {str(e)}")

    # Verificamos si se creó la gráfica
    if os.path.exists('grafica.png'):
        logging.info("✅ Gráfico guardado como grafica.png")
        print("✅ Gráfico guardado como grafica.png")
    else:
        logging.warning("⚠️ Código ejecutado pero no se generó 'grafica.png'.")
        print("⚠️ Código ejecutado pero no se generó 'grafica.png'.")

    # Indicamos que el código se guardó correctamente
    logging.info(f"✅ Código guardado en {codigo_file}")
    print(f"✅ Código guardado en {codigo_file}")

# ---------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------
def main():
    """
    Flujo principal del programa:
    1. Carga el dataset
    2. Toma la solicitud del usuario
    3. Genera código de visualización usando un LLM
    4. Ejecuta el código y genera la gráfica
    """
    df = cargar_dataset('dataset.csv')

    print("\n📊 Ingrese solicitud de visualización:")
    user_input = input("> ")

    # Generamos el código robusto de visualización
    codigo = generar_codigo(df, user_input)

    # Ejecutamos y guardamos el resultado
    ejecutar_y_guardar(codigo)

# ---------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
