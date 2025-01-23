import asyncio
import os
import uuid
from typing import Any, Dict
from datetime import datetime
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Cargar variables de entorno
load_dotenv()

# Configurar modelo
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Nueva herramienta para transformaciones de datos
async def herramienta_transformacion(
    filtro: str = None, 
    sort_by: str = None, 
    top_n: int = None,
    dataset: str = "cereal.csv"
) -> Dict[str, Any]:
    """Realiza transformaciones en el dataset y guarda un archivo temporal."""
    import pandas as pd
    try:
        df = pd.read_csv(dataset)
        
        # Aplicar filtros
        if filtro:
            df = df.query(filtro)
        
        # Ordenar datos
        if sort_by:
            ascending = not sort_by.startswith('-')
            sort_col = sort_by[1:] if sort_by.startswith('-') else sort_by
            df = df.sort_values(by=sort_col, ascending=ascending)
        
        # Seleccionar top N
        if top_n:
            df = df.head(top_n)
        
        # Generar nombre único para el archivo temporal
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}.csv"
        df.to_csv(temp_filename, index=False)
        
        return {
            "success": True,
            "temp_file": temp_filename,
            "registros": len(df),
            "columnas": df.columns.tolist(),
            "preview": df.head(3).to_dict(orient='records')
        }
    except Exception as e:
        return {"error": str(e)}

# Herramienta para leer el CSV
async def herramienta_csv() -> Dict[str, Any]:
    """Lee el archivo cereal.csv y devuelve información detallada de columnas."""
    import pandas as pd
    try:
        df = pd.read_csv('cereal.csv')
        
        # Mejor muestra de datos con variedad
        sample = pd.concat([df.head(3), df.sample(3), df.tail(3)]).drop_duplicates()
        
        return {
            "columns": {
                col: {
                    "tipo": str(dtype),
                    "valores_unicos": df[col].unique()[:10].tolist(),
                    "ejemplo_valores": sample[col].tolist()
                } for col, dtype in df.dtypes.items()
            },
            "total_registros": len(df),
            "message": "Datos cargados exitosamente"
        }
    except Exception as e:
        return {"error": f"Error al leer el archivo: {str(e)}"}

# Mejora en la herramienta de gráficos
async def herramienta_grafico(
    tipo: str,
    eje_x: str,
    eje_y: str = None,
    dataset: str = "cereal.csv",
    kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Genera gráficos desde cualquier dataset temporal."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    try:
        # Cargar dataset
        df = pd.read_csv(dataset)
        
        # Obtener parámetros adicionales con valores por defecto
        kwargs = kwargs or {}
        
        # Limpieza básica
        df = df.dropna().reset_index(drop=True)
        
        # Aplicar filtro si existe
        if 'filtro' in kwargs:
            df = df.query(kwargs['filtro'])
            if df.empty:
                return {"error": "El filtro eliminó todos los registros"}
        
        # Configuración inicial
        sns.set_style(kwargs.get('estilo', 'whitegrid'))
        plt.figure(figsize=kwargs.get('size', (12, 7)))
        
        # Lógica de gráficos (actualizada)
        if tipo == 'scatter':
            plot = sns.scatterplot(data=df, x=eje_x, y=eje_y, **kwargs)
        elif tipo == 'bar':
            if not eje_y:
                df[eje_x].value_counts().plot(kind='bar', **kwargs)
            else:
                df.groupby(eje_x)[eje_y].mean().plot(kind='bar', **kwargs)
        elif tipo == 'line':
            df.plot.line(x=eje_x, y=eje_y, **kwargs)
        elif tipo == 'histogram':
            df[eje_x].plot.hist(bins=kwargs.get('bins', 20), **kwargs)
        else:
            return {"error": f"Tipo de gráfico '{tipo}' no soportado"}
        
        # Personalización
        plt.title(kwargs.get('titulo', "") or f"{tipo.capitalize()} de {eje_x}" + (f" vs {eje_y}" if eje_y else ""))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar y cerrar
        plt.savefig('grafico.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "success": True,
            "file_path": "grafico.png",
            "dataset_used": dataset,
            "registros_usados": len(df)
        }
    
    except Exception as e:
        return {"error": f"Error en generación: {str(e)}"}

# Definición de agentes
coordinador = AssistantAgent(
    "coordinador",
    model_client=model_client,
    handoffs=["analista_datos", "transformador_datos", "visualizador", "user"],
    system_message="""Flujo DEBE seguir estos pasos:
1. Analizar requerimiento del usuario para detectar necesidades de transformación
2. Si se necesita filtrar/ordenar:
   a. Consultar a analista_datos por metadatos
   b. Enviar solicitud a transformador_datos
   c. Validar dataset temporal resultante
3. Pasar dataset temporal al visualizador
4. Verificar que el gráfico usa el dataset correcto
5. Gestionar archivos temporales post-proceso

NUNCA permitas handoff sin confirmar éxito en cada paso intermedio."""
)

# Nuevo agente transformador de datos
transformador_datos = AssistantAgent(
    "transformador_datos",
    model_client=model_client,
    tools=[herramienta_transformacion],
    handoffs=["coordinador"],
    system_message="""Eres el experto en transformación de datos. Tu tarea es:
1. Recibir requerimientos de filtrado/ordenación del coordinador
2. Usar herramienta_transformacion para:
   - Aplicar filtros complejos con sintaxis Pandas
   - Ordenar datos por columnas específicas
   - Seleccionar top N registros
3. Validar que el dataset resultante es válido
4. Reportar el archivo temporal generado
5. Alertar si se pierden demasiados registros"""
)

analista_datos = AssistantAgent(
    "analista_datos",
    model_client=model_client,
    tools=[herramienta_csv],
    handoffs=["coordinador"],
    system_message="""Provee inteligencia sobre los datos:
1. Al recibir consulta, usar herramienta_csv
2. Identificar columnas relevantes para transformaciones
3. Sugerir posibles estrategias de filtrado/ordenación
4. Alertar sobre posibles problemas en transformaciones
5. Proveer ejemplos de sintaxis para filtros complejos"""
)

visualizador = AssistantAgent(
    "visualizador",
    model_client=model_client,
    tools=[herramienta_grafico],
    handoffs=["coordinador"],
    system_message="""Genera gráficos desde cualquier dataset:
1. Extraer parámetros esenciales (tipo, eje_x, eje_y, dataset)
2. Pasar parámetros opcionales como keyword arguments:
   - titulo: str
   - filtro: str (sintaxis Pandas)
   - size: tuple (ancho, alto)
   - estilo: str (whitegrid/dark/ticks)
3. Validar presencia de columnas en el dataset
4. Manejar automáticamente parámetros no reconocidos"""
)

# Configuración del equipo
termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE")
equipo_graficos = Swarm(
    participants=[coordinador, analista_datos, visualizador, transformador_datos],
    termination_condition=termination
)

# Función para ejecutar el flujo
async def run_team_stream() -> None:
    """Maneja la ejecución del flujo de trabajo"""
    print("Bienvenido al sistema de generación de gráficos")
    print("Ejemplo de solicitud: 'Genera un gráfico de dispersión de rating vs azúcar'")
    
    tarea = input("\nIngresa tu solicitud: ")
    
    resultado = await Console(equipo_graficos.run_stream(task=tarea))
    ultimo_mensaje = resultado.messages[-1]
    
    while isinstance(ultimo_mensaje, HandoffMessage) and ultimo_mensaje.target == "user":
        respuesta = input("Usuario: ")
        
        mensaje_retorno = HandoffMessage(
            source="user",
            target=ultimo_mensaje.source,
            content=respuesta
        )
        
        resultado = await Console(equipo_graficos.run_stream(task=mensaje_retorno))
        ultimo_mensaje = resultado.messages[-1]

if __name__ == "__main__":
    asyncio.run(run_team_stream())