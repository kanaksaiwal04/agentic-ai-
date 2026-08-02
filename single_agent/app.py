import os
import certifi
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

# ==========================
# LOAD ENV VARIABLES
# ==========================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ==========================
# LLM
# ==========================

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0
)

# ==========================
# WEATHER TOOL
# ==========================

@tool
def weather_agent(city: str) -> str:
    """Returns the current weather of the given city."""

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": city
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "success" in data and data["success"] == False:
        return data["error"]["info"]

    return f"""
Location: {data['location']['name']}, {data['location']['country']}
Temperature: {data['current']['temperature']}°C
Feels Like: {data['current']['feelslike']}°C
Condition: {data['current']['weather_descriptions'][0]}
Humidity: {data['current']['humidity']}%
Wind Speed: {data['current']['wind_speed']} km/h
"""

# ==========================
# SEARCH TOOL
# ==========================

search = TavilySearchResults(api_key=TAVILY_API_KEY)

# ==========================
# TOOLS
# ==========================

tools = [
    search,
    weather_agent
]

# ==========================
# PROMPT
# ==========================

prompt = hub.pull("hwchase17/react")

# ==========================
# CREATE AGENT
# ==========================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================
# EXECUTOR
# ==========================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ==========================
# RUN
# ==========================

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    response = agent_executor.invoke(
        {
            "input": query
        }
    )

    print("\nAnswer:")
    print(response["output"])