from serpapi import GoogleSearch
from typing import Literal, Optional
from .base import ToolResult, BaseAnthropicTool
import os
from icecream import ic
from rich import print as rr
from dotenv import load_dotenv

load_dotenv()

class GoogleSearchTool(BaseAnthropicTool):
    """
    A tool that performs Google searches using the SerpAPI service and returns structured results.
    """

    name: Literal["google_search"] = "google_search"
    api_type: Literal["custom"] = "custom"
    description: str = "A tool that performs Google searches and returns structured results including organic listings, ads, and related queries."

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location to geo-target results (e.g. 'Baltimore, Maryland, United States')"
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language code (e.g. 'en')"
                    }
                },
                "required": ["query"]
            }
        }

    async def __call__(
        self,
        *,
        query: str,
        location: Optional[str] = "Baltimore, Maryland, United States",
        language: Optional[str] = "en",
        **kwargs,
    ) -> ToolResult:
        """
        Executes a Google search using the provided parameters.
        """
        try:
            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                return ToolResult(error="SERPAPI_KEY environment variable not set")

            # Configure search parameters
            params = {
                "api_key": api_key,
                "engine": "google",
                "q": query,
                "google_domain": "google.com",
                "gl": "us",
                "hl": language,
                "device": "desktop"
            }

            # Add optional location if provided
            if location:
                params["location"] = location

            # Execute search
            ic("Executing Google search with params:", params)
            search = GoogleSearch(params)
            results = search.get_dict()
            return ToolResult(output=results)

        except Exception as e:
            ic(e)
            error_msg = f"Failed to execute Google search: {str(e)}"
            rr(f"Google search error: {error_msg}")
            return ToolResult(error=error_msg)
