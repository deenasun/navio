from typing import Any
import json
import httpx
from mcp.server.fastmcp import FastMCP
from backend import SupabaseClient, SupabaseInsert
from webscraper import WebScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("navio_server")


@mcp.tool()
def hello_world(name: str) -> str:
    """Say hello!

    Args:
        name: The name of the person to say hello to.
    """
    return f"Hello, {name}!"


@mcp.tool()
async def navio_search(query: str) -> str:
    """Search the Navio database for the most similar vectors to the query.
    The Navio database is a vector database that maps user queries to actionable workflows to complete the user's request.
    For example, if the user asks "order a bagel", the Navio database will return matching workflows.
    An example of a matching workflow:
    {
        "url": "https://www.example.com/bagel_shop",
        "steps": [
            {
                "step": 1,
                "description": "navigate to the url",
            },
            {
                "step": 2,
                "description": "click on the bagel order button",
            },
            {
                "step": 3,
                "description": "select the bagel type and quantity",
            },
            {
                "step": 4,
                "description": "click on the order button",
            },
            {
                "step": 5,
                "description": "confirm the order and pay for the bagel",
            }
        ]
    }
    Args:
        query: The query to search for.
    """
    logger.info(f"Starting navio_search with query: {query}")
    supabase = await SupabaseClient.create()
    response = await supabase.query_vector(query)
    logger.info(f"navio_search response: {response}")
    if response.success:
        if not response.matches:
            return "Search completed but no matches were returned."
        return json.dumps([match.model_dump() for match in response.matches], indent=2)
    else:
        return f"Failed to search the Navio database: {response.error}"


@mcp.tool()
async def navio_insert(description: str, workflow: dict) -> str:
    """Insert a new document into the Navio database.
    The Navio database is a vector database that maps user queries to actionable workflows to complete the user's request.
    For example, if the user asks "order a bagel", the Navio database will return matching workflows.
    An example of a matching workflow:
    {
        "url": "https://www.example.com/bagel_shop",
        "steps": [
            {
                "step": 1,
                "description": "navigate to the url",
            },
            {
                "step": 2,
                "description": "click on the bagel order button",
            },
            {
                "step": 3,
                "description": "select the bagel type and quantity",
            },
            {
                "step": 4,
                "description": "click on the order button",
            },
            {
                "step": 5,
                "description": "confirm the order and pay for the bagel",
            }
        ]
    }

    Args:
        description: The description of the document.
        workflow: The workflow of the document. The document should contain the following fields:
            - name: str (name of the workflow)
            - url: str (url of the website where the workflow applies to)
            - steps: List[dict] (list of steps in the workflow)
                - step: int (step number)
                - description: str (description of the step)
    """
    logger.info(f"Starting navio_insert with description: {description}")
    supabase = await SupabaseClient.create()
    embedding = supabase.embedder.embed_text(description, return_numpy=True).tolist()
    document = SupabaseInsert(embedding=embedding, description=description, workflow=workflow)
    response = await supabase.insert_vector(document)
    logger.info(f"navio_insert response: {response}")
    if response.success:
        return f"Document inserted successfully with ID: {response.inserted_id}"
    else:
        return f"Failed to insert the document into the Navio database: {response.error}"


@mcp.tool()
async def navio_scrape(url: str) -> str:
    """Scrape a website and return a summary of the content.
    The summary will include the following:
        - a high-level overview of the website's structure and key subpages in a file diagram format
        - a detailed summary of the website's content
        - brief descriptions about the types of actions and workflows that can be performed on the website

    Args:
        url: The URL of the website to scrape.
    """
    scraper = WebScraper()
    scraper.scrape_website(url)
    summary = scraper.summarize_content()
    return summary.model_dump_json(indent=2)

@mcp.tool()
async def navio_query_agent_txt(url: str) -> str:
    """Query the agents_txt table for the agent.txt for the given website URL.
    The agent.txt is a detailed summary of a website containing the following:
    - a high-level overview of the website's structure and key subpages in a file diagram format
    - a detailed summary of the website's content
    - brief descriptions about the types of actions and workflows that can be performed on the website
    
    The agent.txt can be used to guide another agent to efficiently navigate the website and perform actions.

    Args:
        url: The URL of the website to query the agent.txt for.
    Returns:
        The agent.txt for the given website URL.
    """
    supabase = await SupabaseClient.create()
    response = await supabase.query_agent_txt(url)
    logger.info(f"navio_query_agent_txt response: {response}")
    return response.results

def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
