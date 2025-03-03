#!/usr/bin/env python
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
        # Define el modelo a usar en litellm
        self.model = "gemini/gemini-2.0-flash"

    @start()
    def inicio(self):
        """
        Paso inicial del Flow:
        - Toma el prompt y el contenido CSV desde la API.
        - Genera el código Python (raw_code) vía LLM.
        """
        if not self.api_input:
            raise ValueError("Se requiere 'api_input'. El modo CLI ha sido removido.")
        
        # Modo API
        user_request = self.api_input['prompt']
        csv_content = self.api_input['csv_content']
        self._custom_state['inputs'] = {
            'topic': user_request,
            'current_year': str(datetime.now().year),
            'csv_content': csv_content
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

    def get_generated_code(self) -> str:
        """Retorna el código Python limpio utilizado para generar la visualización."""
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

    REQUISITOS OBLIGATORIOS:
    1. Usa ÚNICAMENTE matplotlib y pandas para la visualización y procesamiento de datos.
    2. NO uses librerías como plotly, seaborn u otras que puedan causar problemas de serialización.
    3. Codifica la imagen EXCLUSIVAMENTE como base64 en memoria usando io.BytesIO.
    4. El código debe imprimir SOLAMENTE el string base64 resultante - nada más, nada menos.
    5. NO guardes archivos en disco (todo debe procesarse en memoria).
    6. Usa ÚNICAMENTE CARACTERES ASCII en todos los textos (títulos, etiquetas, leyendas).
    7. NO uses tildes, ñ, ni caracteres especiales en ningún texto.

    TRATAMIENTO DE DATOS:
    8. Limpia el dataset antes de visualizarlo:
       - Maneja explícitamente valores nulos o faltantes (reemplaza numéricos con 0, texto con "sin datos")
       - Elimina o filtra filas/columnas completamente vacías si es necesario
       - Verifica y convierte tipos de datos según sea necesario

    VISUALIZACIÓN:
    9. Usa una paleta de colores contrastante y accesible.
    10. Asegura que todos los elementos (títulos, etiquetas, leyendas) sean claros y legibles.
    11. Ajusta automáticamente tamaños y escalas para evitar superposiciones o texto cortado.
    12. Usa un tamaño de figura adecuado (mínimo 10x6 pulgadas) para buena resolución.

    ESTRUCTURA DE CÓDIGO OBLIGATORIA:

    import matplotlib.pyplot as plt
    import pandas as pd
    import io
    import base64
    import numpy as np  # Si es necesario

    # Configuración para evitar problemas
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6), dpi=100)

    try:
        # Tu código para procesar datos y generar el gráfico
        # ASEGÚRATE de manejar excepciones y valores faltantes
        
        # Guarda la figura en memoria
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        # IMPRIME SOLO el base64, nada más
        print(base64.b64encode(buf.read()).decode('utf-8'))
    except Exception as e:
        # En caso de error, genera una imagen simple con mensaje de error
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f"Error en visualización: {{str(e)}}", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
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
        """
        lines = stdout.split('\n')
        for line in lines:
            candidate = line.strip()
            if candidate.startswith('iVBORw0KGgo'):
                return candidate
        return None
