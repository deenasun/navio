import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel, Field

load_dotenv()

SYSTEM_PROMPT = """
Your goal is to summarize the content of a website.
Your summary should include the following:
- a high-level overview of the website's structure and key subpages in a file diagram format
- a detailed summary of the website's content
- brief descriptions about the types of actions and workflows that can be performed on the website

The summary should be informative enough to guide another agent to efficiently navigate the website and perform actions.
"""

GEMINI_MODEL = "google/gemini-2.5-flash"


class Summary(BaseModel):
    file_diagram: str = Field(description="A high-level overview of the website's structure and key subpages in a file diagram format")
    content_summary: str = Field(description="A detailed summary of the website's content")
    workflows: list[str] = Field(description="Brief descriptions about the types of actions and workflows that can be performed on the website")


class WebScraper:
    def __init__(self, max_depth: int = 3):
        self.headers = {"User-Agent": "agent/1.0"}
        self.domain = None
        self.max_depth = max_depth
        self.scraped_content = []
        self.client = instructor.from_provider(GEMINI_MODEL, api_key=os.getenv("GOOGLE_AI_API_KEY"))

    def scrape_website(self, url: str, depth: int = 0) -> str:
        if depth == 0:
            self.domain = urlparse(url).netloc
        elif depth > self.max_depth or self.domain not in url:
            return
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.text, "html.parser")

            # Append content from this page to the scraped_content list
            html_text = soup.get_text("\n", strip=True)
            self.scraped_content.append({"url": url, "content": html_text})

            # Find all links on the page
            links = soup.find_all("a")
            for link in links:
                # Recursively scrape links on the same domain
                if self.domain in link.get("href", ""):
                    self.scrape_website(link.get("href"), depth + 1)
        except requests.exceptions.RequestException as e:
            pass

    def summarize_content(self) -> str:
        content = "\n".join([item["content"] for item in self.scraped_content])
        response = self.client.create(
            response_model=Summary,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        return response


def main():
    scraper = WebScraper(max_depth=1)
    scraper.scrape_website("https://en.wikipedia.org/wiki/La_La_Land")
    summary = scraper.summarize_content()
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
