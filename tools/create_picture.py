from typing import Literal
from pathlib import Path
from .base import ToolResult, BaseAnthropicTool
import replicate
import base64
from icecream import ic
from enum import Enum
from dotenv import load_dotenv
load_dotenv()
import os
from PIL import Image
from utils.file_logger import log_file_operation
from config import get_constant, get_project_dir, to_docker_path
from utils.docker_service import DockerService
from tools.file_manager import FileManagerTool

class PictureCommand(str, Enum):
    CREATE = "create"

class PictureGenerationTool(BaseAnthropicTool):
    """Tool for generating pictures using the Flux Schnell model"""

    name: Literal["picture_generation"] = "picture_generation"
    api_type: Literal["custom"] = "custom"
    description: str = "Creates pictures based on text prompts. This is how you will create pictures that you need for projects."

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display  # Explicitly set self.display
        self.file_manager = FileManagerTool(display=display)  # Initialize FileManagerTool

    def to_params(self) -> dict:
        ic(f"PictureGenerationTool.to_params called with api_type: {self.api_type}")
        # Use the format that has worked in the past
        params = {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [cmd.value for cmd in PictureCommand],
                        "description": "Command to execute: create"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where the generated image will be saved"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Optional width to resize the image"
                    },
                    "height": {
                        "type": "integer",
                        "description": "Optional height to resize the image"
                    }
                },
                "required": ["command", "prompt"]
            }
        }
        ic(f"PictureGenerationTool params: {params}")
        return params

    async def generate_picture(self, prompt: str, output_path: str, width: int = None, height: int = None) -> dict:
        """
        Generates an image based on the prompt and saves it to the output path.
        
        Args:
            prompt: Text description of the image to generate
            output_path: Path where the image should be saved
            width: Optional width to resize the image to
            height: Optional height to resize the image to
            
        Returns:
            A dictionary containing the result
        """
        try:
            # Import necessary libraries
            import replicate
            import base64
            from PIL import Image
            from pathlib import Path
            import os
            from config import get_constant, REPO_DIR
            docker_service = DockerService()
            output_path_obj = docker_service.from_docker_path(output_path)
            # Create input data for the model
            input_data = {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "safety_filter_level": "block_only_high"
            }
            # Get the image data from replicate
            client = replicate.Client()
            output = client.run(
                "google/imagen-3-fast",
                input=input_data
            )

            # Read all bytes from the FileOutput object
            image_data = output.read()

            if not image_data:
                raise Exception("No image data received from the model")

            # Save the raw bytes to file
            output_path_obj.write_bytes(image_data)

            # Log the file creation with metadata
            metadata = {
                "prompt": prompt,
                "dimensions": f"{width}x{height}" if width and height else "original",
                "model": "google/imagen-3-fast",
                "generation_params": input_data
            }
            
            try:
                self.file_manager.create_file(output_path_obj, image_data)  # Use FileManagerTool to log creation
            except Exception as log_error:
                print(f"Warning: Failed to log image creation: {log_error}")
                # Continue anyway - don't let logging failure prevent success
            
            # Create base64 for display
            base64_data = base64.b64encode(image_data).decode("utf-8")

            # Handle resizing if needed
            if width or height:
                img = Image.open(output_path_obj)
                if width and height:
                    new_size = (width, height)
                elif width:
                    ratio = width / img.width
                    new_size = (width, int(img.height * ratio))
                else:
                    ratio = height / img.height
                    new_size = (int(img.width * ratio), height)
                
                img = img.resize(new_size, Image.LANCZOS)
                img.save(output_path_obj)
                
                # Update metadata with new dimensions
                metadata["dimensions"] = f"{new_size[0]}x{new_size[1]}"
                try:
                    self.file_manager.edit_file(output_path_obj, image_data)  # Use FileManagerTool to log edit
                except Exception as log_error:
                    print(f"Warning: Failed to log image resize: {log_error}")
                    # Continue anyway - don't let logging failure prevent success
                
                # Update base64 data
                image_data = output_path_obj.read_bytes()
                base64_data = base64.b64encode(image_data).decode("utf-8")
            
            # Convert the local output path to Docker path for display
            
            # Create a nice HTML message for the output
            html_output = f'<div><p>Image generated from prompt: "{prompt}"</p>'
            html_output += f'<p>Saved to: {output_path_obj}</p>'
            html_output += f'<img src="data:image/png;base64,{base64_data}" style="max-width:100%; max-height:500px;"></div>'
            self.display.add_message("tool", html_output)
            return {
                "status": "success",
                "output_path": output_path_obj,
                "html": html_output
            }
            
        except Exception as e:
            import traceback
            error_stack = traceback.format_exc()
            error_message = f"Error generating image: {str(e)}"
            print(f"Error generating image: {str(e)}\n{error_stack}")
            return {
                "status": "error",
                "message": error_message
            }

    def format_output(self, data: dict) -> str:
        """
        Format the output of the tool for display.
        
        Args:
            data: The data returned by the tool
            
        Returns:
            A formatted string for display
        """
        if "error" in data and data["error"]:
            return f"Error: {data['error']}"
            
        if "output" in data:
            return data["output"]
            
        # For backward compatibility
        if "message" in data:
            return data["message"]
            
        # Default case
        return str(data)

    async def __call__(
        self,
        *,
        command: PictureCommand,
        prompt: str,
        output_path: str = "output",
        width: int = None,
        height: int = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute the tool with the given command and parameters.
        
        Args:
            command: The command to execute
            prompt: The text prompt to generate the image from
            output_path: The path to save the image to
            width: Optional width to resize the image to
            height: Optional height to resize the image to
            **kwargs: Additional parameters
            
        Returns:
            A ToolResult object with the result of the operation
        """
        try:
            if command == PictureCommand.CREATE:
                result = await self.generate_picture(prompt, output_path, width, height)
                
                if "error" in result and result["error"]:
                    return ToolResult(
                        error=result["error"],
                        tool_name=self.name,
                        command=command
                    )
                    
                return ToolResult(
                    output=result.get("output", "Image generated successfully"),
                    base64_image=result.get("base64_image"),
                    message=result.get("message"),
                    tool_name=self.name,
                    command=command
                )
            else:
                return ToolResult(
                    error=f"Unknown command: {command}",
                    tool_name=self.name,
                    command=command
                )
        except Exception as e:
            import traceback
            error_message = f"Error in PictureGenerationTool: {str(e)}\n{traceback.format_exc()}"
            return ToolResult(
                error=error_message,
                tool_name=self.name,
                command=command
            )
