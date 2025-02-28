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
        super().__init__(display)

    def to_params(self) -> dict:
        return {
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

    async def generate_picture(self, prompt: str, output_path: str, width: int = None, height: int = None) -> dict:
        """Generates a picture using the Flux Schnell model"""
        try:
            # Convert the output_path to absolute path if it's relative to project path
            project_dir = get_constant("PROJECT_DIR")
            docker_project_dir = get_constant("DOCKER_PROJECT_DIR")

            # Handle path conversion for Docker environment
            # if docker_project_dir and str(docker_project_dir) in output_path:
            #     # This is a Docker path, convert to local path for file operations
            rel_path = Path(output_path).relative_to(docker_project_dir)
            output_path = project_dir / rel_path

            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            input_data = {
                "prompt": prompt,
                "aspect_ratio": "1:1",
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

            # Log the file creation
            log_file_operation(Path(output_path), 'create')

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

                resized_img = img.resize(new_size, Image.LANCZOS)
                resized_img.save(output_path)

            # For display purposes, convert back to Docker path if needed
            display_path = output_path
            if docker_project_dir and project_dir:
                # Convert local path to Docker path for display
                try:
                    rel_path = Path(output_path).relative_to(project_dir)
                    display_path = docker_project_dir / rel_path
                except ValueError:
                    # Not relative to project_dir, keep as is
                    pass

            # Display message
            html_message = (
                "<div>"
                "<p>Generated image saved to: {}</p>"
                f'<img src="data:image/png;base64,{base64_data}" alt="Generated Image" style="max-width:100%;">'
                "</div>"
            ).format(display_path)

            if self.display:
                self.display.add_message("tool", html_message)

            return {
                "status": "success",
                "output_files": [str(display_path)],  # Use display path in output
                "prompt": prompt
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string"""
        output_lines = []
        output_lines.append(f"Status: {data['status']}")

        if data['status'] == 'success':
            output_lines.append(f"Prompt: {data['prompt']}")
            output_lines.append("Generated files:")
            for file in data['output_files']:
                output_lines.append(f"  - {file}")
        else:
            output_lines.append(f"Error: {data['error']}")

        return "\n".join(output_lines)

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
        """Executes the picture generation command"""
        try:
            if self.display:
                self.display.add_message("user", f"Creating picture with description:\n{prompt}")

            # No need to create directories here as generate_picture handles it

            if command == PictureCommand.CREATE:
                result_data = await self.generate_picture(prompt, output_path, width, height)
            else:
                return ToolResult(error=f"Unknown command: {command}")

            formatted_output = self.format_output(result_data)

            return ToolResult(output=formatted_output)

        except Exception as e:
            if self.display:
                self.display.add_message("user", f"PictureGenerationTool error: {str(e)}")
            return ToolResult(error=f"Failed to execute {command}: {str(e)}")
