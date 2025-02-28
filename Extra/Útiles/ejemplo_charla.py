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

# Load environment variables from .env
load_dotenv()

# Get API keys from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# If you don't have them set, raise an error or warning.
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Missing ALPHA_VANTAGE_API_KEY. Please set it in your .env file.")
if not NEWS_API_KEY:
    raise ValueError("Missing NEWS_API_KEY. Please set it in your .env file.")


async def get_stock_data(symbol: str) -> Dict[str, Any]:
    """
    Get real stock market data for a given symbol using the Alpha Vantage API.
    Returns a dictionary with price, volume, and other metrics.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

    # Alpha Vantage returns a dictionary with a "Global Quote" field
    global_quote = data.get("Global Quote", {})
    price = float(global_quote.get("05. price", 0.0))
    volume = int(global_quote.get("06. volume", 0))
    pe_ratio = "N/A"  # Global Quote doesn't always include P/E ratio
    market_cap = "N/A"  # Not provided in GLOBAL_QUOTE

    return {
        "symbol": symbol.upper(),
        "price": price,
        "volume": volume,
        "pe_ratio": pe_ratio,
        "market_cap": market_cap
    }


async def get_news(query: str) -> List[Dict[str, str]]:
    """
    Get recent news articles about a company from the News API.
    Returns a list of dictionaries containing title, date, and summary.
    """
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 5  # limit results
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()

    articles = data.get("articles", [])
    news_data = []
    for article in articles:
        news_data.append({
            "title": article.get("title", "No Title"),
            "date": article.get("publishedAt", "Unknown Date"),
            "summary": article.get("description", "No Description Provided")
        })

    return news_data


async def write_report(content: str, agent_name: str) -> Dict[str, str]:
    """
    Write the given content to a .md file. Each agent writes its own file
    into a subfolder (named after the agent) inside the 'reports' folder.

    Returns a dict with the path to the written file.
    """
    # Ensure the reports/<agent_name> directory exists
    folder_path = os.path.join("reports", agent_name)
    os.makedirs(folder_path, exist_ok=True)

    # Create a filename with a timestamp (to keep multiple reports distinct)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    file_path = os.path.join(folder_path, filename)

    # Write the content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {"file_path": file_path}


# Create an OpenAI ChatCompletion client (be sure to configure your OpenAI key)
model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
)

# Define your specialized agents
planner = AssistantAgent(
    "planner",
    model_client=model_client,
    handoffs=["financial_analyst", "news_analyst", "writer", "user"],
    system_message="""You are a research planning coordinator.
    Coordinate market research by delegating to specialized agents:
    - Financial Analyst: For stock data analysis
    - News Analyst: For news gathering and analysis
    - Writer: For compiling final report
    Always send your plan first, then hand off to the appropriate agent.
    Always hand off to a single agent at a time.
    When research is complete, hand off to the user""",
)

financial_analyst = AssistantAgent(
    "financial_analyst",
    model_client=model_client,
    tools=[get_stock_data, write_report],
    handoffs=["planner"],
    system_message="""You are a financial analyst.
    - Use the get_stock_data(tool) to analyze financial metrics.
    - Summarize or document findings by calling write_report(content, 'financial_analyst').
    Always handoff back to planner or user when analysis is complete."""
)

news_analyst = AssistantAgent(
    "news_analyst",
    model_client=model_client,
    tools=[get_news, write_report],
    handoffs=["planner"],
    system_message="""You are a news analyst.
    - Use the get_news(tool) to gather relevant articles.
    - Summarize or document findings by calling write_report(content, 'news_analyst').
    Always handoff back to planner or user when analysis is complete."""
)

writer = AssistantAgent(
    "writer",
    model_client=model_client,
    tools=[write_report],
    handoffs=["planner"],
    system_message="""You are a financial report writer.
    - Compile all research into a final Markdown report using write_report(content, 'writer').
    Always handoff back to planner or user when writing is complete."""
)


# Define termination conditions
handoff_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")
termination = handoff_termination | text_termination

research_team = Swarm(
    participants=[planner, financial_analyst, news_analyst, writer],
    termination_condition=termination
)


async def run_team_stream() -> None:
    """
    Runs the research flow in an interactive console environment.
    The script will ask for an initial task, coordinate agent handoffs,
    and then allow user input whenever a handoff is directed to 'user'.
    """
    task = input("Please enter the initial task: ")

    # Start the multi-agent conversation with the provided task
    task_result = await Console(research_team.run_stream(task=task))
    last_message = task_result.messages[-1]

    # Continue looping if the last message is a Handoff to the user
    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        # Create a new HandoffMessage that sends the user's reply
        # back to whoever delegated (last_message.source)
        handoff_message = HandoffMessage(
            source="user",
            target=last_message.source,
            content=user_message
        )

        task_result = await Console(research_team.run_stream(task=handoff_message))
        last_message = task_result.messages[-1]


if __name__ == "__main__":
    asyncio.run(run_team_stream())
