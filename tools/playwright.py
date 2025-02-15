#playwright.py
import os
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Literal, Optional, Dict, Any, List
import requests
from bs4 import BeautifulSoup, Comment  # Add Comment to the import
import re
from .base import ToolResult
# Configure logging for user feedback and debugging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

class WebNavigatorTool:
    """
    A versatile tool that uses Playwright to interact with the web, including reading information,
    navigating websites, filling forms, extracting data, interacting with dynamic elements, and downloading files.
    It also integrates with external APIs and includes enhanced error handling and #logging.
    """

    name: Literal["web_navigator"] = "web_navigator"
    api_type: Literal["custom"] = "custom"
    description: str = (
        """A comprehensive tool that uses Playwright to perform various web interactions such as reading information, 
        navigating websites, filling out forms, extracting data, interacting with dynamic elements, and downloading files. 
        It also integrates with external APIs and includes enhanced error handling and logging for improved user experience."""
    )

    def __init__(self, download_dir: Optional[str] = None, api_credentials: Optional[Dict[str, Any]] = None):
        """
        Initializes the WebNavigatorTool with optional download directory and API credentials.
        """
        self.download_dir = download_dir or os.path.join(os.getcwd(), "downloads")
        os.makedirs(self.download_dir, exist_ok=True)
        self.api_credentials = api_credentials or {}
        self.session_history = []  # For contextual awareness
        self.browser = None
        self.context = None
        self.page = None
        logging.info("WebNavigatorTool initialized with download directory at '%s'.", self.download_dir)

    async def init_browser(self):
        """Initialize browser if not already running"""
        if not self.browser:
            p = await async_playwright().start()
            
            # Use existing Chrome installation and profile
            chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            self.browser = await p.chromium.launch_persistent_context(
                user_data_dir=os.path.expanduser("~/AppData/Local/Google/Chrome/User Data/Default"),
                executable_path=chrome_path,
                headless=False,
                handle_sigint=False,
                handle_sigterm=False,
                handle_sighup=False
            )
            self.page = await self.browser.new_page()
            # No need for separate context since we're using launch_persistent_context

    async def cleanup(self):
        """Close browser and cleanup resources"""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None

    def to_params(self) -> dict:
        """
        Defines the parameters for the tool, specifying the input schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to perform the action on."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["read", "navigate", "download", "fill_form", "extract_data", "click_element"],
                        "description": "The action to perform."
                    },
                    "params": {
                        "type": "object",
                        "description": "Additional parameters required for the action.",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to save the downloaded file (required for 'download' action)."
                            },
                            "form_selector": {
                                "type": "string",
                                "description": "CSS selector for the form to fill (required for 'fill_form' action)."
                            },
                            "form_data": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "string"
                                },
                                "description": "Data to fill into the form fields."
                            },
                            "data_selector": {
                                "type": "string",
                                "description": "CSS selector for data extraction (required for 'extract_data' action)."
                            },
                            "element_selector": {
                                "type": "string",
                                "description": "CSS selector of the element to click (required for 'click_element' action)."
                            }
                        }
                    }
                },
                "required": ["url", "action"]
            }
        }
    async def __call__(
        self,
        url: str,
        action: Literal["read", "navigate", "download", "fill_form", "extract_data", "click_element"],
        params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:  
        """        Args:
            url (str): The URL to perform the action on.
            action (Literal["read", "navigate", "download", "fill_form", "extract_data", "click_element"]): 
                The action to perform. Must be one of "read", "navigate", "download", "fill_form", "extract_data", "click_element".
            params (Optional[Dict[str, Any]]): Optional parameters for the action. 
                - For "download", requires "file_path".
                - For "fill_form", requires "form_selector" and "form_data".
                - For "extract_data", requires "data_selector".
                - For "click_element", requires "element_selector".
        Returns:
            ToolResult: A ToolResult object containing the output or error message.
        Raises:
            ValueError: If required parameters for the specified action are missing.
            PlaywrightTimeoutError: If a timeout occurs while performing the action.
            Exception: For any other errors that occur during the action.
        Executes the specified action on the given URL with optional parameters.
        Returns a ToolResult object containing the output or error message.
        """
        params = params or {}
        
        try:
            await self.init_browser()
            result = None
            
            if action == "read":
                result = await self.read_info(self.page, url)
            elif action == "navigate":
                result = await self.navigate_website(self.page, url)
            elif action == "download":
                file_path = params.get("file_path")
                if not file_path:
                    raise ValueError("Parameter `file_path` is required for action: download")
                result = await self.download_file(self.page, url, file_path)
            elif action == "fill_form":
                form_selector = params.get("form_selector")
                form_data = params.get("form_data")
                if not form_selector or not form_data:
                    raise ValueError("Parameters `form_selector` and `form_data` are required for action: fill_form")
                result = await self.fill_form(self.page, url, form_selector, form_data)
            elif action == "extract_data":
                data_selector = params.get("data_selector")
                if not data_selector:
                    raise ValueError("Parameter `data_selector` is required for action: extract_data")
                result = await self.extract_data(self.page, url, data_selector)
            elif action == "click_element":
                element_selector = params.get("element_selector")
                if not element_selector:
                    raise ValueError("Parameter `element_selector` is required for action: click_element")
                result = await self.click_element(self.page, url, element_selector)
            else:
                raise ValueError(f"Unrecognized action '{action}'")
            
            return ToolResult(output=result)

        except PlaywrightTimeoutError:
            error_msg = f"Timeout occurred while performing action '{action}' on {url}."
            logging.error(error_msg)
            return ToolResult(error=error_msg)
        except Exception as e:
            error_msg = f"An error occurred while performing action '{action}' on {url}: {str(e)}"
            return ToolResult(error=error_msg)

    async def read_info(
            self, 
            page, 
            url: str,
            content_type: str = "structured",  # Options: "raw", "cleaned", "text", "structured"
            selectors: Optional[List[str]] = None,  # CSS selectors to specifically target
            exclude_selectors: Optional[List[str]] = None,  # CSS selectors to exclude
            remove_scripts: bool = True,
            remove_styles: bool = True,
            remove_comments: bool = True,
            preserve_links: bool = True,
            max_length: Optional[int] = None
        ) -> str:
            """
            Reads and processes the content of the specified URL.
            
            Args:
                page: Playwright page object
                url: Target URL
                content_type: Type of content processing to apply
                selectors: List of CSS selectors to specifically target
                exclude_selectors: List of CSS selectors to exclude
                remove_scripts: Whether to remove script tags
                remove_styles: Whether to remove style tags
                remove_comments: Whether to remove HTML comments
                preserve_links: Whether to preserve href attributes in the output
                max_length: Maximum length of returned content
            """
            await page.goto(url)
            raw_html = await page.content()
            self.session_history.append(f"read_info: {url}")
            
            # Create BeautifulSoup object for parsing
            soup = BeautifulSoup(raw_html, 'html.parser')

            if content_type == "raw":
                return raw_html

            # Remove unwanted elements
            if remove_scripts:
                for script in soup.find_all('script'):
                    script.decompose()

            if remove_styles:
                for style in soup.find_all('style'):
                    style.decompose()
                # Remove inline styles
                for tag in soup.find_all(style=True):
                    del tag['style']

            if remove_comments:
                for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                    comment.extract()

            # Remove tracking and analytics elements
            tracking_classes = {'analytics', 'tracking', 'advertisement', 'ad-', 'cookie-banner'}
            for element in soup.find_all(class_=lambda x: x and any(track in x.lower() for track in tracking_classes)):
                element.decompose()

            # Process based on content type
            if content_type == "cleaned":
                # Keep only specified selectors if provided
                if selectors:
                    new_soup = BeautifulSoup('', 'html.parser')
                    for selector in selectors:
                        for element in soup.select(selector):
                            new_soup.append(element)
                    soup = new_soup

                # Remove excluded selectors
                if exclude_selectors:
                    for selector in exclude_selectors:
                        for element in soup.select(selector):
                            element.decompose()

                # Clean up remaining HTML
                for tag in soup.find_all(True):
                    # Remove empty tags
                    if len(tag.get_text(strip=True)) == 0 and tag.name not in ['img', 'br', 'hr']:
                        tag.decompose()
                        continue
                    
                    # Remove all attributes except href if preserve_links is True
                    if preserve_links and tag.name == 'a':
                        href = tag.get('href', '')
                        tag.attrs = {'href': href} if href else {}
                    else:
                        tag.attrs = {}

                content = str(soup)

            elif content_type == "text":
                # Extract only text content
                content = soup.get_text(separator=' ', strip=True)
                # Clean up whitespace
                content = re.sub(r'\s+', ' ', content)

            elif content_type == "structured":
                # Create a structured representation of the content
                content = self._create_structured_content(soup)

            else:
                raise ValueError(f"Unsupported content_type: {content_type}")

            # Truncate if max_length is specified
            if max_length and len(content) > max_length:
                content = content[:max_length] + "..."

            logging.info("Read and processed content from '%s' using mode '%s'.", url, content_type)
            return content
    def _create_structured_content(self, soup: BeautifulSoup) -> str:
        """
        Creates a structured representation of the content.
        """
        structure = []
        
        # Extract title
        title = soup.title.string if soup.title else ""
        if title:
            structure.append(f"Title: {title.strip()}")

        # Extract headings
        headings = []
        for tag in ['h1', 'h2', 'h3']:
            for heading in soup.find_all(tag):
                text = heading.get_text(strip=True)
                if text:
                    headings.append(f"{tag.upper()}: {text}")
        if headings:
            structure.append("\nHeadings:\n" + "\n".join(headings))

        # Extract main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content_text = main_content.get_text(separator=' ', strip=True)
            content_text = re.sub(r'\s+', ' ', content_text)
            structure.append("\nMain Content:\n" + content_text)

        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            text = link.get_text(strip=True)
            href = link['href']
            if text and href:
                links.append(f"- {text}: {href}")
        if links:
            structure.append("\nLinks:\n" + "\n".join(links))

        return "\n\n".join(structure)
    async def navigate_website(self, page, url: str) -> str:
        """
        Navigates to the specified URL and returns the page title.
        """
        await page.goto(url)
        title = await page.title()
        self.session_history.append(f"navigate: {url}")
        logging.info("Navigated to '%s' with title '%s'.", url, title)
        return f"Navigated to {url}. Page title: {title}"

    async def download_file(self, page, url: str, file_path: str) -> str:
        """
        Downloads a file from the specified URL and saves it to the given file path.
        """
        os.makedirs(self.download_dir, exist_ok=True)
        async with page.expect_download() as download_info:
            await page.goto(url)
        download = await download_info.value
        save_path = os.path.join(self.download_dir, file_path)
        await download.save_as(save_path)
        self.session_history.append(f"download_file: {save_path}")
        logging.info("Downloaded file from '%s' to '%s'.", url, save_path)
        return f"File downloaded and saved to {save_path}"

    async def fill_form(self, page, url: str, form_selector: str, form_data: Dict[str, str]) -> str:
        """
        Fills out a form identified by form_selector with the provided form_data.
        """
        await page.goto(url)
        for field, value in form_data.items():
            await page.fill(f"{form_selector} {field}", value)
        await page.click(f"{form_selector} button[type='submit']")
        self.session_history.append(f"fill_form: {url} with {form_data}")
        logging.info("Filled form on '%s' with data '%s'.", url, form_data)
        return f"Form on {url} filled with provided data and submitted."

    async def extract_data(self, page, url: str, data_selector: str) -> str:
        """
        Extracts and returns data from the specified selector on the webpage.
        """
        await page.goto(url)
        data = await page.inner_text(data_selector)
        self.session_history.append(f"extract_data: {url} - {data_selector}")
        logging.info("Extracted data from '%s' using selector '%s'.", url, data_selector)
        return data

    async def click_element(self, page, url: str, element_selector: str) -> str:
        """
        Clicks an element identified by the selector on the webpage.
        """
        await page.goto(url)
        await page.click(element_selector)
        self.session_history.append(f"click_element: {url} - {element_selector}")
        logging.info("Clicked element '%s' on '%s'.", element_selector, url)
        return f"Clicked element '{element_selector}' on {url}."

    def integrate_external_api(self, api_url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Integrates with an external API and returns the response.
        """
        params = params or {}
        headers = {
            "Authorization": f"Bearer {self.api_credentials.get('api_key', '')}",
            "Content-Type": "application/json"
        }
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            logging.info("Successfully integrated with external API '%s'.", api_url)
            return response.json()
        else:
            logging.error("Failed to integrate with external API '%s'. Status code: %s", api_url, response.status_code)
            raise ConnectionError(f"Failed to connect to external API '{api_url}'. Status code: {response.status_code}")

    def get_session_history(self) -> str:
        """
        Returns the session history for contextual awareness.
        """
        history = "\n".join(self.session_history)
        logging.info("Session history retrieved.")
        return history
