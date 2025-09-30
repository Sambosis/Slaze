from typing import Literal
from .base import ToolResult, BaseTool
import logging
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
# DockerService import removed

logger = logging.getLogger(__name__)


class PictureCommand(str, Enum):
    CREATE = "create"


class PictureGenerationTool(BaseTool):
    """Tool for generating pictures using the Flux Schnell model"""

    name: Literal["picture_generation"] = "picture_generation"
    api_type: Literal["custom"] = "custom"
    description: str = "Creates pictures based on text prompts. This is how you will create pictures that you need for projects."

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display  # Explicitly set self.display

    def to_params(self) -> dict:
        logger.debug(
            f"PictureGenerationTool.to_params called with api_type: {self.api_type}"
        )
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type":
                    "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": [cmd.value for cmd in PictureCommand],
                            "description": "Command to execute: create",
                        },
                        "prompt": {
                            "type":
                            "string",
                            "description":
                            "Text description of the image to generate",
                        },
                        "output_path": {
                            "type":
                            "string",
                            "description":
                            "Path where the generated image will be saved, relative to REPO_DIR (e.g., 'images/my_pic.png').",
                        },
                        "width": {
                            "type": "integer",
                            "description":
                            "Width to resize the image (required)",
                        },
                        "height": {
                            "type": "integer",
                            "description":
                            "Height to resize the image (required)",
                        },
                    },
                    "required":
                    ["command", "prompt", "output_path", "width", "height"],
                },
            },
        }
        logger.debug(f"PictureGenerationTool params: {params}")
        return params

    async def generate_picture(self, prompt: str, output_path: str, width: int,
                               height: int) -> dict:
        """
        Generates an image based on the prompt using the specified width and height,
        and saves it to the output path relative to REPO_DIR.

        Args:
            prompt: Text description of the image to generate
            output_path: Path where the image should be saved, relative to REPO_DIR.
            width: Width to resize the image to (required)
            height: Height to resize the image to (required)

        Returns:
            A dictionary containing the result
        """
        try:
            # Import necessary libraries
            import replicate
            import base64
            from PIL import Image
            from pathlib import Path
            from config import get_constant  # Added import

            # --- Path Construction ---
            host_repo_dir = get_constant("REPO_DIR")
            if not host_repo_dir:
                raise ValueError("REPO_DIR is not configured in config.py.")

            base_save_path = Path(host_repo_dir)
            output_path_obj = (base_save_path / output_path).resolve()

            # Ensure the parent directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Resolved image output path: {output_path_obj}")

            # Create input data for the model
            input_data = {
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "safety_filter_level": "block_only_high",
            }
            # Get the image data from replicate
            client = replicate.Client()
            output = client.run("google/imagen-4-fast", input=input_data)

            # Read all bytes from the FileOutput object
            image_data = output.read()

            if not image_data:
                raise Exception("No image data received from the model")

            # Save the raw bytes to file
            output_path_obj.write_bytes(image_data)

            # Log the file creation with metadata
            metadata = {
                "prompt": prompt,
                "dimensions":
                f"{width}x{height}" if width and height else "original",
                "model": "google/imagen-3-fast",
                "generation_params": input_data,
            }

            try:
                from utils.file_logger import log_file_operation

                log_file_operation(file_path=output_path_obj,
                                   operation="create",
                                   metadata=metadata)
            except Exception as log_error:
                logger.warning(f"Failed to log image creation: {log_error}",
                               exc_info=True)
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
                    log_file_operation(output_path_obj,
                                       "update",
                                       metadata=metadata)
                except Exception as log_error:
                    logger.warning(f"Failed to log image resize: {log_error}",
                                   exc_info=True)
                    # Continue anyway - don't let logging failure prevent success

                # Update base64 data
                image_data = output_path_obj.read_bytes()
                base64_data = base64.b64encode(image_data).decode("utf-8")

            # Convert the local output path to Docker path for display

            # Create a nice HTML message for the output
            html_output = f'<div><p>Image generated from prompt: "{prompt}"</p>'
            html_output += f"<p>Saved to: {output_path_obj}</p>"
            html_output += f'<img src="data:image/png;base64,{base64_data}" style="max-width:100%; max-height:500px;"></div>'
            self.display.add_message("tool", html_output)
            return {
                "status": "success",
                "output_path": output_path_obj,
                "html": html_output,
            }

        except Exception as e:
            import traceback

            error_stack = traceback.format_exc()
            error_message = f"Error generating image: {str(e)}"
            logger.error(f"Error generating image: {str(e)}\n{error_stack}",
                         exc_info=True)
            return {"status": "error", "message": error_message}

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
        output_path: str,  # Removed default, now required
        width: int,
        height: int,
        **kwargs,
    ) -> ToolResult:
        """
        Execute the tool with the given command and parameters.

        Args:
            command: The command to execute
            prompt: The text prompt to generate the image from
            output_path: The path to save the image to, relative to REPO_DIR.
            width: Width to resize the image (required)
            height: Height to resize the image (required)
            **kwargs: Additional parameters

        Returns:
            A ToolResult object with the result of the operation
        """
        try:
            if command == PictureCommand.CREATE:
                result = await self.generate_picture(prompt, output_path,
                                                     width, height)

                if "error" in result and result["error"]:
                    return ToolResult(error=result["error"],
                                      tool_name=self.name,
                                      command=command)

                return ToolResult(
                    output=result.get("output",
                                      "Image generated successfully"),
                    base64_image=result.get("base64_image"),
                    message=result.get("message"),
                    tool_name=self.name,
                    command=command,
                )
            else:
                return ToolResult(
                    error=f"Unknown command: {command}",
                    tool_name=self.name,
                    command=command,
                )
        except Exception as e:
            import traceback

            error_message = (
                f"Error in PictureGenerationTool: {str(e)}\n{traceback.format_exc()}"
            )
            return ToolResult(error=error_message,
                              tool_name=self.name,
                              command=command)
