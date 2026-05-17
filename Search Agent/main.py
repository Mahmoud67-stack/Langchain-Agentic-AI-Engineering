from typing import List

from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

class Source(BaseModel):

    """
    Schema for a source used by the agent
    """
    
    url: str = Field(description="The URL of the source")

    name: str = Field(description="The name of the source")

class AgentResponse(BaseModel):
    """
    Schema for the response of the agent
    """
    answer: str = Field(description="The answer to the question or query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used by the agent")
    



# The Tavily client can automatically search the .env file for the proper keys if the load_dotenv() function is called and the name of the key should be TAVILY_API_KEY
# tavily = TavilyClient()

# Custom tool this syntax is nessecary to make sure the agent can use the tool properly
# @tool
# def search(query: str) -> str:
#     """
#     Tool that seacrhes over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The results of the search
#     """
#
#     print(f"Searching the web for {query}...")
#     return tavily.search(query=query)

key = os.getenv("OPENROUTER_API_KEY")

if not key:
    raise ValueError("OPENROUTER_API_KEY is missing")

key = key.strip()

llm = ChatOpenAI(
    model="openai/gpt-5-nano",
    temperature=0,
    api_key=key,
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=1,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "LangChain Course",
    },
)

# Keep Tavily lightweight and fast
tools = [
    TavilySearch(
        max_results=3,
        search_depth="basic",
    )
]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)


def main():
    print("Hello from search-agent!")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search for exactly 3 AI jobs in Dubai on Linkedin "
                        "that mention LangChain. "
                        "Return only:\n"
                        "- Job title\n"
                        "- Company\n"
                        "- Location\n"
                        "- Link\n"
                        "- 1 line summary\n\n"
                        "Do not search more than once."
                    )
                )
            ]
        },
        config={
            "recursion_limit": 4
        }
    )

    print("\nFINAL RESULT:\n")
    print(result)


if __name__ == "__main__":
    main()