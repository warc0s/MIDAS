#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from plot_v0.crew import PlotV0
import re

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def get_visualization_request():
    """
    Solicita al usuario la descripción completa del gráfico que desea generar.
    """
    print("\n¿Qué gráfico necesitas generar? Ejemplo:")
    print("'Gráfico de barras con los cereales que contienen una C en su nombre, mostrando sus calorías ordenadas de menor a mayor'")
    return input("\nDescribe tu gráfico: ").strip()

def clean_code(file_path='grafica.py'):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Eliminar bloques ```python y ```
    cleaned = re.sub(r'```python|```', '', content)
    
    # Eliminar líneas vacías al principio/final
    cleaned = cleaned.strip()
    
    with open(file_path, 'w') as f:
        f.write(cleaned)

def run():
    """
    Ejecuta el crew con la solicitud específica del usuario.
    """
    # Obtener la solicitud completa del usuario
    user_request = get_visualization_request()
    
    # Configurar las entradas para los agentes
    inputs = {
        'topic': user_request,  # Pasamos la descripción completa como topic
        'current_year': str(datetime.now().year)
    }
    
    try:
        # Ejecutar el crew con la solicitud del usuario
        PlotV0().crew().kickoff(inputs=inputs)
        clean_code()
        print("\n¡Gráfico generado con éxito! Ejecuta 'python grafica.py' para ver los resultados.")
    except Exception as e:
        raise Exception(f"Error al ejecutar el proceso: {e}")



if __name__ == "__main__":
    run()