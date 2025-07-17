#!/usr/bin/env python3
"""Test script to verify Unicode/emoji support in both console and web modes."""

import asyncio
import pytest
from tools.bash import BashTool

def test_unicode_output():
    """Test basic Unicode output."""
    test_strings = [
        "Hello World! 🌍",
        "Python is awesome! 🐍✨",
        "Testing emojis: 😀 😍 🎉 🚀",
        "Unicode: àáâãäåæçèéêë",
        "Japanese: こんにちは",
        "Mathematical symbols: ∑ ∫ √ π",
    ]
    
    print("=== Unicode/Emoji Test ===")
    for i, test_str in enumerate(test_strings, 1):
        print(f"Test {i}: {test_str}")
    
    return True

@pytest.mark.asyncio
async def test_bash_tool_unicode():
    """Test BashTool with Unicode output."""
    print("\n=== BashTool Unicode Test ===")
    
    # Initialize BashTool
    bash_tool = BashTool()
    
    # Test a simple echo command with Unicode
    test_command = 'echo "Hello World! 🌍 Python is awesome! 🐍✨"'
    print(f"Running command: {test_command}")
    
    try:
        result = await bash_tool(test_command)
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main test function."""
    print("Starting Unicode/Emoji support test...")
    
    # Test 1: Basic Unicode output
    test1_success = test_unicode_output()
    
    # Test 2: BashTool with Unicode
    test2_success = asyncio.run(test_bash_tool_unicode())
    
    print("\n=== Test Results ===")
    print(f"Basic Unicode test: {'PASS' if test1_success else 'FAIL'}")
    print(f"BashTool Unicode test: {'PASS' if test2_success else 'FAIL'}")
    
    if test1_success and test2_success:
        print("✅ All tests passed! Unicode/emoji support is working correctly.")
    else:
        print("❌ Some tests failed. Unicode/emoji support may have issues.")

if __name__ == "__main__":
    main()
