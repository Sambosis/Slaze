# server.py

from fastmcp import FastMCP


# Create an MCP server
mcp = FastMCP("Demo")
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m ** 2)
