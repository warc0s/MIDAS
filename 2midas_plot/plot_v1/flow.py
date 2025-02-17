#!/usr/bin/env python
import sys
import re
from datetime import datetime

from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion
from e2b_code_interpreter import Sandbox

load_dotenv()

class FlowPlotV1(Flow):
    def __init__(self, api_input=None):
        super().__init__()
        self.api_input = api_input
        # Almacena información necesaria durante el flujo
        self._custom_state = {
            'inputs': {},
            'raw_code': '',
            'clean_code': ''
        }
        # Define el modelo a usar en litellm (ajusta según tu configuración)
        self.model = "gemini/gemini-2.0-flash"

    @start()
    def inicio(self):
        """
        Paso inicial del Flow:
        - Toma el prompt y el contenido CSV si viene de la API (o de CLI en modo local).
        - Genera el código Python (raw_code) vía LLM.
        """
        if self.api_input:
            # Modo API
            user_request = self.api_input['prompt']
            csv_content = self.api_input['csv_content']
            self._custom_state['inputs'] = {
                'topic': user_request,
                'current_year': str(datetime.now().year),
                'csv_content': csv_content
            }
        else:
            # Modo CLI (interactivo)
            user_request = self._get_visualization_request()
            self._custom_state['inputs'] = {
                'topic': user_request,
                'current_year': str(datetime.now().year),
                'csv_content': None
            }
        
        # Generamos el código con LLM
        raw_code = self._generate_plot_code()
        self._custom_state['raw_code'] = raw_code
        return raw_code

    @listen(inicio)
    def limpiar_codigo(self, raw_code):
        """
        Limpia el código de los posibles backticks (```python ... ```) 
        que a veces generan los modelos, para poder ejecutarlo en la sandbox.
        """
        cleaned = re.sub(r'```python|```', '', raw_code).strip()
        self._custom_state['clean_code'] = cleaned
        return cleaned

    @listen(limpiar_codigo)
    def ejecutar_codigo(self, clean_code):
        """
        Ejecuta el código en la sandbox y extrae la imagen codificada en base64 desde stdout.
        """
        with Sandbox() as sbx:
            # Si tenemos contenido CSV, lo escribimos en un archivo dentro del sandbox
            if self._custom_state['inputs']['csv_content']:
                sbx.files.write('/data.csv', self._custom_state['inputs']['csv_content'])
            
            execution = sbx.run_code(clean_code)
            
            # Capturamos los logs de salida
            stdout = '\n'.join(execution.logs.stdout) if execution.logs.stdout else ''
            stderr = '\n'.join(execution.logs.stderr) if execution.logs.stderr else ''
            
            if stderr:
                raise RuntimeError(f"Error al ejecutar el código en sandbox: {stderr}")
            
            # Extraemos el base64 de la salida
            base64_output = self._extraer_base64(stdout)
            if not base64_output:
                raise ValueError("No se detectó base64 válido en la salida del código.")
            
            return base64_output

    def _get_visualization_request(self):
        """
        Si corres esto en modo CLI, aquí pides al usuario que describa el gráfico.
        """
        print("¿Qué gráfico necesitas generar? Ejemplo:")
        print("  'Gráfico de barras con el conteo de ventas por categoría'")
        return input("\nDescribe tu gráfico: ").strip()

    def get_generated_code(self) -> str:
        """Retorna el código Python limpio utilizado para generar la visualización"""
        return self._custom_state.get('clean_code', '')

    def _generate_plot_code(self):
        """
        Genera código Python usando un modelo LLM (litellm) para crear la visualización.
        Debe imprimir únicamente el string base64 en stdout.
        """
        user_request = self._custom_state['inputs']['topic']
        csv_instructions = ""
        # Si hay CSV, indicamos al LLM cómo cargarlo
        if self._custom_state['inputs']['csv_content']:
            csv_instructions = (
                "El dataset está disponible en '/data.csv'. "
                "Por ejemplo:\n"
                "import pandas as pd\n"
                "df = pd.read_csv('/data.csv')"
            )
        
        prompt = f"""
Genera código Python para crear un {user_request} usando matplotlib.
{csv_instructions}

Requisitos estrictos:
1. Usar matplotlib y pandas (si se necesita).
2. Codificar la imagen como base64 en memoria usando io.BytesIO.
3. Imprimir EXCLUSIVAMENTE el string base64 sin ningún texto adicional.
4. No guardes archivos en disco (excepto leer el CSV si existe).
5. Redacta en español los títulos de las graficas.
6. Formato obligatorio:

import matplotlib.pyplot as plt
import io
import base64

# Tu código para generar el gráfico...

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

print(base64.b64encode(buf.read()).decode('utf-8'))
"""

        # Llamamos a la función de litellm para obtener la respuesta (código)
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def _extraer_base64(self, stdout):
        """
        Busca una cadena que empiece con el típico encabezado base64 de PNG (iVBORw0KGgo).
        Puedes ajustar la lógica según tus necesidades.
        """
        lines = stdout.split('\n')
        for line in lines:
            candidate = line.strip()
            if candidate.startswith('iVBORw0KGgo'):
                return candidate
        return None

if __name__ == "__main__":
    """Modo CLI: se inicia el flow y pide descripción de gráfico en consola."""
    try:
        flow = FlowPlotV1()
        resultado_base64 = flow.kickoff()
        print("Base64 result:\n", resultado_base64)
    except Exception as e:
        print(f"Error en el flujo: {e}", file=sys.stderr)
        sys.exit(1)
