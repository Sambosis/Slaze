import difflib
import pandas as pd
from typing import Optional, Literal, TypedDict
# from tools import ToolResult, , BaseAnthropicTool  # Adjust the import path as necessary
from enum import StrEnum
from .base import ToolError, ToolResult, BaseAnthropicTool
from icecream import ic
from playwright.async_api import async_playwright
import os
class Resolution(TypedDict):
    width: int
    height: int
from rich import print as rr

class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"

class Options(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: Optional[int]

class GoToURLReportsTool(BaseAnthropicTool):
    """
    A tool that allows the agent to list available reports and run a selected report
    from the Auto Chlor System Web Portal.
    """

    name: Literal["reports"] = "reports"
    api_type: Literal["custom"] = "custom"
    description: str = "A tool that allows the agent to list available reports and run a selected report from the Auto Chlor System Web Portal."
    height: int
    display_num: Optional[int]

    _screenshot_delay = 1.0
    _scaling_enabled = True

    @property
    def options(self) -> Options:
        return {
            "display_width_px": self.width,
            "display_height_px": self.height,
            "display_number": self.display_num,
        }

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {  # Use parameters instead of custom.input_schema
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["list_reports", "run_report"],
                        "description": "The command to execute. Either 'list_reports' to list all reports or 'run_report' to execute a specific report."
                    },
                    "report_name": {
                        "type": "string",
                        "description": "The name of the report to run. Required if command is 'run_report'."
                    }
                },
                "required": ["command"],
            }
        }
        
    async def __call__(     
        self,
        *,
        command: Literal["list_reports", "run_report"],
        report_name: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Executes the specified command.
        """
        if command == "list_reports":
            return await self.list_reports()
        elif command == "run_report":
            if not report_name:
                raise ToolError("Parameter `report_name` is required for command: run_report")
            return await self.run_report(report_name)
        else:
            raise ToolError(
                f"Unrecognized command '{command}'. Allowed commands: 'list_reports', 'run_report'."
            )

    async def list_reports(self) -> ToolResult:
        """
        Lists all available reports.
        """
        try:
            # Example: Replace with your actual data source
            df = pd.read_csv(r"C:\Users\Machine81\code\anthropic-quickstarts\computer-use-demo\computer_use_demo\filtered_links.csv")  # Ensure this path is correct
            report_list = "\n".join(f"- {name}" for name in df['display_name'])
            # ic(report_list)
            return ToolResult(output=f"Available Reports:\n{report_list}")
        except Exception as e:
            raise ToolError(f"Failed to list reports: {str(e)}")

    async def run_report(self, report_name: str) -> ToolResult:
        """
        Runs the specified report by finding the closest match and opening the URL.
        """
        try:
            # Example: Replace with your actual data source
            df = pd.read_csv(r"C:\Users\Machine81\code\anthropic-quickstarts\computer-use-demo\computer_use_demo\filtered_links.csv")  # Ensure this path is correct
            ic(df.head())
            # replace _ with space in report_name
            report_name = report_name.replace("_", " ")
            rr(report_name)
            matches = difflib.get_close_matches(report_name, df['display_name'], n=1, cutoff=0.1,)
            ic(matches)
            if not matches:
                raise ToolError(f"No matching report found for '{report_name}'.")

            best_match = matches[0]

            url_suffix = df.loc[df['display_name'] == best_match, 'url'].iloc[0]
            full_url = f"https://www.autochlor.net/wps/{url_suffix}"
            ic(f"Full URL: {full_url}")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                ic
                context = await browser.new_context(storage_state=r"C:\mygit\compuse\computer_use_demo\state.json")
                ic(context)
                page = await context.new_page()
                ic(page)
                await page.goto(full_url)
                # press the PRINT button
                page.click('input[type="submit"][value="PRINT"]')

            return ToolResult(output="That report is available ")
        except Exception as e:
            raise ToolError(f"Failed to run report '{report_name}': {str(e)}")
        


    async def download_file(self, url: str):

        # Initialize Playwright
        async with async_playwright() as p:
            # Launch Chromium with specified user data directory
            browser = await p.chromium.launch(channel="chrome", headless=False)
            context = browser.new_context(storage_state=r"C:\mygit\compuse\computer_use_demo\state.json")
            # Navigate to page (state will be preserved)
            page.goto('https://www.autochlor.net/wps')
            
            # Verify state is loaded
            cookies = context.cookies()
            local_storage = page.evaluate("""() => {
                return Object.entries(localStorage);
            }""")
            
 
            # Open a new page
            page = await browser.new_page()

            await page.goto(url)

            # Directory to save downloads
            download_path = os.path.join(os.getcwd(), "downloads")
            os.makedirs(download_path, exist_ok=True)

            for index, row in df.iterrows():
                if row['url_type'] == 'File':
                    file_url = base_url + row['url']
                    rr(f"Downloading from URL: {file_url}")

                    try:
                        # Listen for the download event
                        async with page.expect_download() as download_info:
                            await page.goto(file_url)
                        download = await download_info.value
                        
                        # Save the downloaded file to the specified path
                        await download.save_as(os.path.join(download_path, download.suggested_filename))
                        rr(f"Downloaded: {download.suggested_filename}")

                    except Exception as e:
                        rr(f"Error downloading from URL: {file_url}")
                        rr(e)

            # Close the browser after all downloads
            await browser.close()   