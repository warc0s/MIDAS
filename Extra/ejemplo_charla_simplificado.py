import asyncio
import os
from typing import Any, Dict, List
from datetime import datetime

from dotenv import load_dotenv
import aiohttp

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Cargar variables de entorno desde .env
load_dotenv()

# Aquí se definen claves de API como placeholders
API_KEY_1 = os.getenv("API_KEY_1")  # Ejemplo: usado para bolsa
API_KEY_2 = os.getenv("API_KEY_2")  # Ejemplo: usado para noticias

# Comprobar que las variables de entorno estén definidas
if not API_KEY_1:
    raise ValueError("Falta API_KEY_1 en el entorno (.env).")
if not API_KEY_2:
    raise ValueError("Falta API_KEY_2 en el entorno (.env).")


async def herramienta1(simbolo: str) -> Dict[str, Any]:
    """
    Ejemplo de herramienta para obtener datos financieros de 'simbolo'.
    Devuelve un diccionario con precio, volumen, etc.
    """
    base_url = "https://www.ejemplo-finanzas.com/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": simbolo,
        "apikey": API_KEY_1
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

    global_quote = data.get("Global Quote", {})
    price = float(global_quote.get("05. price", 0.0))
    volume = int(global_quote.get("06. volume", 0))
    pe_ratio = "N/A"
    market_cap = "N/A"

    return {
        "symbol": simbolo.upper(),
        "price": price,
        "volume": volume,
        "pe_ratio": pe_ratio,
        "market_cap": market_cap
    }


async def herramienta2(consulta: str) -> List[Dict[str, str]]:
    """
    Ejemplo de herramienta para obtener artículos/noticias recientes
    relacionados con la palabra clave 'consulta'.
    Devuelve una lista de diccionarios con título, fecha y resumen.
    """
    base_url = "https://www.ejemplo-noticias.org/v2/everything"
    params = {
        "q": consulta,
        "sortBy": "publishedAt",
        "apiKey": API_KEY_2,
        "language": "en",
        "pageSize": 5
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

    articles = data.get("articles", [])
    lista_noticias = []
    for article in articles:
        lista_noticias.append({
            "title": article.get("title", "No Title"),
            "date": article.get("publishedAt", "Unknown Date"),
            "summary": article.get("description", "No Description Provided")
        })

    return lista_noticias


async def herramienta3(contenido: str, nombre_agente: str) -> Dict[str, str]:
    """
    Ejemplo de herramienta para escribir 'contenido' en un archivo .md.
    Se crea una carpeta reports/<nombre_agente> y allí se guarda el informe.
    Devuelve un diccionario con la ruta del archivo creado.
    """
    folder_path = os.path.join("reports", nombre_agente)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(contenido)

    return {"file_path": file_path}


# Cliente de modelo (PLACEHOLDER) para interacciones de agente
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key="sk-placeholder"
)

# Definición de agentes (se mantiene la lógica, pero con nombres genéricos)
agente1 = AssistantAgent(
    "agente1",
    model_client=model_client,
    handoffs=["agente2", "agente3", "agente4", "user"],
    system_message="""Eres el coordinador principal.
- Delegas la investigación a otros agentes especializados.
- Haces 'handoff' a un solo agente a la vez.
- Cuando el trabajo esté completo, haces 'handoff' al usuario."""
)

agente2 = AssistantAgent(
    "agente2",
    model_client=model_client,
    tools=[herramienta1, herramienta3],
    handoffs=["agente1"],
    system_message="""Eres el analista financiero.
- Usa herramienta1 para obtener datos financieros.
- Documenta con herramienta3.
- Devuelve control con 'handoff' a agente1 o usuario."""
)

agente3 = AssistantAgent(
    "agente3",
    model_client=model_client,
    tools=[herramienta2, herramienta3],
    handoffs=["agente1"],
    system_message="""Eres el analista de noticias.
- Usa herramienta2 para obtener noticias.
- Documenta con herramienta3.
- Devuelve control con 'handoff' a agente1 o usuario."""
)

agente4 = AssistantAgent(
    "agente4",
    model_client=model_client,
    tools=[herramienta3],
    handoffs=["agente1"],
    system_message="""Eres el redactor final.
- Compila y redacta el informe con herramienta3.
- Devuelve control con 'handoff' a agente1 o usuario."""
)

# Condiciones de terminación del flujo de agentes
handoff_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")
termination = handoff_termination | text_termination

# Equipo de agentes
equipo_investigacion = Swarm(
    participants=[agente1, agente2, agente3, agente4],
    termination_condition=termination
)


async def run_team_stream() -> None:
    """
    Permite iniciar la conversación multi-agente en consola.
    El usuario ingresa la tarea inicial, los agentes se pasan el testigo (handoff),
    y el usuario puede responder cuando reciba el testigo.
    """
    tarea = input("Por favor, introduce la tarea inicial: ")

    # Iniciar conversación con la tarea
    resultado_tarea = await Console(equipo_investigacion.run_stream(task=tarea))
    ultimo_mensaje = resultado_tarea.messages[-1]

    # Mientras el agente finalice con un handoff al usuario,
    # seguimos pidiendo input al usuario y reenviándolo al agente correspondiente.
    while isinstance(ultimo_mensaje, HandoffMessage) and ultimo_mensaje.target == "user":
        mensaje_usuario = input("Usuario: ")

        # Creación de un nuevo HandoffMessage que devuelve al agente que hizo el handoff
        mensaje_retorno = HandoffMessage(
            source="user",
            target=ultimo_mensaje.source,
            content=mensaje_usuario
        )

        resultado_tarea = await Console(equipo_investigacion.run_stream(task=mensaje_retorno))
        ultimo_mensaje = resultado_tarea.messages[-1]


if __name__ == "__main__":
    asyncio.run(run_team_stream())
