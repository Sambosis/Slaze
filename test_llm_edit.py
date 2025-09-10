#!/usr/bin/env python3
"""
Test script for the LLM-enhanced edit tool functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the workspace to the path
sys.path.insert(0, '/workspace')

from tools.edit import EditTool

async def test_llm_edit():
    """Test the LLM-enhanced edit functionality."""
    
    # Create an instance of the edit tool
    editor = EditTool()
    
    # Create a test file
    test_file = Path("/workspace/repo/test_example.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initial content
    initial_content = """def calculate_sum(a, b):
    # This function calculates sum
    result = a + b
    return result

def main():
    x = 5
    y = 10
    total = calculate_sum(x, y)
    print(f"The sum is: {total}")

if __name__ == "__main__":
    main()
"""
    
    print("Creating test file...")
    result = await editor(
        command="create",
        path=str(test_file),
        file_text=initial_content
    )
    print(f"Result: {result.output}\n")
    
    # Test 1: LLM-based replacement
    print("Test 1: Using LLM to add error handling to the calculate_sum function")
    result = await editor(
        command="str_replace",
        path=str(test_file),
        old_str="The calculate_sum function without error handling",
        new_str="Add type checking and error handling to ensure inputs are numbers",
        match_mode="llm"
    )
    print(f"Result: {result.output}\n")
    
    # View the file to see changes
    print("Viewing modified file:")
    result = await editor(
        command="view",
        path=str(test_file)
    )
    print(result.output)
    print("\n" + "="*50 + "\n")
    
    # Test 2: LLM-based insertion
    print("Test 2: Using LLM to insert a new function")
    result = await editor(
        command="insert",
        path=str(test_file),
        insert_line=5,
        new_str="Add a function called calculate_product that multiplies two numbers with proper error handling"
    )
    print(f"Result: {result.output}\n")
    
    # Test 3: Traditional exact replacement
    print("Test 3: Traditional exact replacement")
    result = await editor(
        command="str_replace",
        path=str(test_file),
        old_str='print(f"The sum is: {total}")',
        new_str='print(f"The result is: {total}")',
        match_mode="exact"
    )
    print(f"Result: {result.output}\n")
    
    # Final view
    print("Final file content:")
    result = await editor(
        command="view",
        path=str(test_file)
    )
    print(result.output)

if __name__ == "__main__":
    print("Testing LLM-Enhanced Edit Tool")
    print("="*50)
    asyncio.run(test_llm_edit())
    print("\n" + "="*50)
    print("Test completed!")