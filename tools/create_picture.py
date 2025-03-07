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
        Generate a picture using the Flux Schnell model.
        
        Args:
            prompt: The text prompt to generate the image from
            output_path: The path to save the image to
            width: Optional width to resize the image to
            height: Optional height to resize the image to
            
        Returns:
            A dictionary with the result of the operation
        """
        try:
            import replicate
            import base64
            from PIL import Image
            from utils.file_logger import log_file_operation
            from pathlib import Path
            from config import get_constant
            
            # Handle path conversion for Docker environment
            project_dir = get_constant('PROJECT_DIR')
            # Ensure project_dir is a string
            if isinstance(project_dir, Path):
                project_dir = str(project_dir)
                
            # Try to get docker project dir, but if it doesn't exist, use a default
            try:
                docker_project_dir = get_constant('DOCKER_PROJECT_DIR')
            except:
                docker_project_dir = '/app'
                
            # Ensure docker_project_dir is a string
            if isinstance(docker_project_dir, Path):
                docker_project_dir = str(docker_project_dir)
            
            # Ensure output_path is a string
            if isinstance(output_path, Path):
                output_path = str(output_path)
                
            # Ensure output path is absolute
            if not os.path.isabs(output_path):
                output_path = os.path.join(project_dir, output_path)
                
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
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
            with open(output_path, 'wb') as f:
                f.write(image_data)

            # Log the file creation with metadata
            metadata = {
                "prompt": prompt,
                "dimensions": f"{width}x{height}" if width and height else "original",
                "model": "google/imagen-3-fast",
                "generation_params": input_data
            }
            # Convert output_path to Path for logging
            output_path_obj = Path(output_path)
            log_file_operation(output_path_obj, "create", metadata=metadata)

            # Create base64 for display
            base64_data = base64.b64encode(image_data).decode("utf-8")

            # Handle resizing if needed
            if width or height:
                img = Image.open(output_path)
                if width and height:
                    new_size = (width, height)
                elif width:
                    ratio = width / img.width
                    new_size = (width, int(img.height * ratio))
                else:
                    ratio = height / img.height
                    new_size = (int(img.width * ratio), height)
                
                img = img.resize(new_size, Image.LANCZOS)
                img.save(output_path)
                
                # Update metadata with new dimensions
                metadata["dimensions"] = f"{new_size[0]}x{new_size[1]}"
                log_file_operation(output_path_obj, "update", metadata=metadata)
                
                # Update base64 data
                with open(output_path, 'rb') as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode("utf-8")
            
            # Convert the local output path to Docker path for display
            docker_output_path = output_path
            if output_path.startswith(project_dir):
                docker_output_path = output_path.replace(project_dir, docker_project_dir)
            
            # Create HTML message for display
            html_message = f"""
            <div style="text-align: center;">
                <p>Image generated and saved to: <code>{docker_output_path}</code></p>
                <img src="data:image/png;base64,{base64_data}" style="max-width: 100%; max-height: 500px;">
            </div>
            """
            
            return {
                "output": f"Image generated successfully and saved to {docker_output_path}",
                "base64_image": base64_data,
                "message": html_message
            }
        except Exception as e:
            import traceback
            error_message = f"Error generating image: {str(e)}\n{traceback.format_exc()}"
            return {"error": error_message}

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
