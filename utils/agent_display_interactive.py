import asyncio
import json
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from pathlib import Path
from .agent_display_console import AgentDisplayConsole


class AgentDisplayInteractive(AgentDisplayConsole):
    """
    Interactive display that shows tool calls to the user for review and editing
    before execution.
    """

    def __init__(self):
        super().__init__()
        self.tool_review_mode = True
        self.console.print("[bold green]Interactive Tool Review Mode Enabled[/bold green]")
        self.console.print("You will be able to review and edit all tool calls before execution.\n")

    def show_tool_call_details(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Display the tool call details in a formatted way."""
        
        # Create a table for the tool call details
        table = Table(title=f"Tool Call: {tool_name}", show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="yellow", width=20)
        table.add_column("Value", style="white", width=60)
        
        for key, value in tool_input.items():
            # Format the value based on its type
            if isinstance(value, (dict, list)):
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)
            
            # Truncate very long values
            if len(formatted_value) > 500:
                formatted_value = formatted_value[:500] + "... (truncated)"
            
            table.add_row(key, formatted_value)
        
        self.console.print(table)

    def show_tool_call_json(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Display the tool call as JSON for easy editing."""
        tool_call_data = {
            "tool_name": tool_name,
            "parameters": tool_input
        }
        
        json_content = json.dumps(tool_call_data, indent=2)
        syntax = Syntax(json_content, "json", theme="monokai", line_numbers=True)
        
        self.console.print(Panel(
            syntax,
            title="Tool Call JSON",
            border_style="blue"
        ))

    def get_user_tool_decision(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Show the tool call to the user and get their decision.
        
        Returns:
            Dict with keys:
            - 'action': 'execute', 'edit', 'skip', or 'exit'
            - 'modified_input': The potentially modified tool input
        """
        
        self.console.print(f"\n[bold yellow]═══ Tool Call Review ═══[/bold yellow]")
        
        # Show tool call details
        self.show_tool_call_details(tool_name, tool_input)
        
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("1. [green]Execute[/green] - Run the tool call as-is")
        self.console.print("2. [yellow]Edit[/yellow] - Modify the parameters")
        self.console.print("3. [blue]View JSON[/blue] - See the full JSON representation")
        self.console.print("4. [red]Skip[/red] - Skip this tool call")
        self.console.print("5. [red]Exit[/red] - Stop the agent")
        
        while True:
            choice = Prompt.ask(
                "\nWhat would you like to do?",
                choices=["1", "2", "3", "4", "5", "execute", "edit", "json", "skip", "exit"],
                default="1"
            )
            
            if choice in ["1", "execute"]:
                return {"action": "execute", "modified_input": tool_input}
            
            elif choice in ["2", "edit"]:
                modified_input = self.edit_tool_parameters(tool_name, tool_input)
                if modified_input is not None:
                    return {"action": "execute", "modified_input": modified_input}
                # If editing was cancelled, continue the loop
                continue
            
            elif choice in ["3", "json"]:
                self.show_tool_call_json(tool_name, tool_input)
                continue
            
            elif choice in ["4", "skip"]:
                return {"action": "skip", "modified_input": tool_input}
            
            elif choice in ["5", "exit"]:
                return {"action": "exit", "modified_input": tool_input}

    def edit_tool_parameters(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Allow the user to edit tool parameters.
        
        Returns:
            Modified tool input dict, or None if cancelled
        """
        
        self.console.print(f"\n[bold cyan]Editing parameters for {tool_name}[/bold cyan]")
        self.console.print("Enter new values for parameters (press Enter to keep current value):")
        
        modified_input = tool_input.copy()
        
        for key, current_value in tool_input.items():
            current_display = str(current_value)
            if len(current_display) > 100:
                current_display = current_display[:100] + "..."
            
            self.console.print(f"\n[yellow]{key}[/yellow] (current: {current_display})")
            
            # Handle different types of values
            if isinstance(current_value, bool):
                new_value = Confirm.ask(f"New value for {key}", default=current_value)
                modified_input[key] = new_value
            
            elif isinstance(current_value, (int, float)):
                try:
                    new_value_str = Prompt.ask(f"New value for {key}", default=str(current_value))
                    if new_value_str != str(current_value):
                        if isinstance(current_value, int):
                            modified_input[key] = int(new_value_str)
                        else:
                            modified_input[key] = float(new_value_str)
                except ValueError:
                    self.console.print(f"[red]Invalid number format, keeping original value[/red]")
            
            elif isinstance(current_value, (dict, list)):
                self.console.print(f"Current JSON structure:")
                self.console.print(Syntax(json.dumps(current_value, indent=2), "json"))
                
                if Confirm.ask(f"Edit JSON structure for {key}?", default=False):
                    new_json_str = Prompt.ask(
                        f"Enter new JSON for {key}",
                        default=json.dumps(current_value, indent=2)
                    )
                    try:
                        modified_input[key] = json.loads(new_json_str)
                    except json.JSONDecodeError:
                        self.console.print(f"[red]Invalid JSON format, keeping original value[/red]")
            
            else:
                # String or other types
                new_value = Prompt.ask(f"New value for {key}", default=str(current_value))
                if new_value != str(current_value):
                    modified_input[key] = new_value
        
        # Show the modified parameters
        self.console.print(f"\n[bold green]Modified parameters:[/bold green]")
        self.show_tool_call_details(tool_name, modified_input)
        
        if Confirm.ask("Confirm these changes?", default=True):
            return modified_input
        else:
            return None

    async def wait_for_user_input(self, prompt_message="➡️ Your input: "):
        """Override to maintain compatibility with parent class."""
        return await super().wait_for_user_input(prompt_message)