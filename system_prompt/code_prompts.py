from config import get_constant 
from typing import List, Dict, Optional


PROMPT_FOR_CODE="""I need you to break down a python file in a structured way that concisely describes how to interact with the code.  
    It should serve as a basic reference for someone who is unfamiliar with the codebase and needs to understand how to use the functions and classes defined in the file.
    It should largely be natural language based.
    The breakdown should include the following elements(this will vary based on the codebase, it should cover every class and function in the file including the main function and gloabl variables and imports):

    Imports: <list of imports and their purpose>
    Global Variables: <list of global variables and their purpose>
    Classes: <list of classes and their purpose>
    Functions: <list of functions and their purpose>

    Class: <class_name>
    Purpose: <description of what the class does>
    Methods: <list of methods and their purpose>
    Attributes: <list of attributes and their purpose>
    Summary: <a concise summary of the class's purpose and behavior>
    Usage: <How to use, When to use, and and Why you should use this function and any other important information>


    Function: <function_name>
    Purpose:  <description of what the function does>
    Parameters: <list of parameters and their types> 
    Returns: <the type of the value returned by the function>
    Summary: <a concise summary of the function's purpose and behavior>
    Usage: <How to use, When to use, and and Why you should use this function and any other important information>


    It should be concise and easy to understand. 
    It should abstract away the implementation details and focus on the high-level functionality of the code.
    It should give someone everything they need to know to use the function without needing to read the implementation details.
    Ensure your response is neatly organized in markdown format.
    """

def code_prompt_research(current_code_base, code_description):
    messages=[
            {
            "role": "user",
            "content": [
                {
                    
                "type": "text",
                "text": f"""At the bottom is a detailed description of code of the code the programmer needs to write.
                you will review it and help them by giving them insight into ways to approaches their task.
                You try to anticipate common bugs, inefficiencies and suggest improvements to the origninal specs to add advanced performance and functionality.
                You need to give 2 different approaches on how to accomplish this task and detail the benefits and limitations of each approach.  If useful, you can provide small code snippets to illustrate your points, however you are not to write the code for them.
                Make observations about how each approach will interact with the existing code base and how it will affect the overall performance of the program. Make certain notes about the file structure and how the new code will fit in.  Try to guide them in using proper import statements and how to structure their code in a way that is easy to read and maintain.
                You take the whole scope of the program into consideration when reviewing their task description.
                Do not tell them which of the approaches they need to take, just provide them with the information they need to make an informed decision and insights to common pitfalls and   best practices of each approach.
                Here is all of the code that has been created for the project so far:
                {current_code_base}
                
                Here is the requeste:
                {code_description}"""
                },
            ]
            }
        ]
    return messages


def code_skeleton_prompt(code_description, existing_structure=None):
    """
    Creates a prompt that asks the LLM to generate code skeleton/structure
    based on a description.

    Args:
        code_description (str): Detailed description of what code to generate
        existing_structure (str, optional): Existing code structure to build upon

    Returns:
        list: Formatted messages for the LLM prompt
    """
    messages = [
        {
            "role": "user",
            "content": f"""You are an expert software architect who creates clean, well-structured code skeletons.
            Your task is to create a comprehensive code structure with proper imports, class definitions, method signatures, 
            and detailed docstrings - but WITHOUT implementation details.
            
            Provide a complete scaffold that a developer can fill in with implementation code later.
            Follow these principles:
            1. Include ALL necessary imports at the top
            2. Define ALL classes with proper inheritance and attributes
            3. Include ALL methods with proper signatures (parameters with type hints)
            4. Write DETAILED docstrings for everything (classes, methods, functions)
            5. Use proper typing annotations throughout
            6. Include constructor methods and initialization where needed
            7. Include abstract methods where appropriate
            8. Add placeholder comments for complex logic sections
            9. Follow PEP standards for Python code
            
            The output should be properly structured, syntactically valid code that could be directly
            placed into a file and run (though it would only contain structure, not implementation).
            Please create a complete code skeleton based on the following description.
            Include all necessary imports, class/method signatures, and detailed docstrings.
            
            {'' if existing_structure is None else f'Build upon or integrate with this existing structure:\n\n{existing_structure}\n\n'}
            
            Code description:
            {code_description}
            
            Please provide ONLY the code skeleton with detailed docstrings and proper type hints, 
            but with minimal implementation details (use `pass` or simple placeholder comments for method bodies).
            """,
        },
    ]

    return messages


def code_prompt_generate(
    current_code_base: str,
    code_description: str,
    research_string: str,
    agent_task: str,
    skeletons: Optional[str] = None,
    external_imports: Optional[List[str]] = None,
    internal_imports: Optional[List[str]] = None,
    target_file: Optional[str] = None
) -> list:
    """
    Generates the prompt messages for code generation, incorporating skeletons and file-specific import lists.
    """
    system_prompt = f"""You are an expert software developer tasked with writing code for a specific file within a larger project.
Your goal is to generate clean, efficient, and correct code based on the provided description, context, and overall project goal.

Overall Project Goal: {agent_task}

You will be given:
1.  A detailed description of the code required for the *target file* ({target_file or 'unknown'}).
2.  Code skeletons for *all* files in the project (if available). These provide the basic structure (classes, functions, imports).
3.  A list of required *external* libraries/packages *specifically for the target file*.
4.  A list of required *internal* modules/files within the project *imported specifically by the target file*.
5.  (Optional) Existing code from the project for context.
6.  (Optional) Research notes related to the task.

Instructions:
- Focus *only* on generating the complete code for the specified *target file*: **{target_file or 'unknown'}**.
- Use the provided skeletons as a starting point and fill in the implementation details.
- Ensure all necessary imports (both external and internal, as provided in the lists *for this file*) are included in the generated code for the target file.
- Adhere strictly to the requirements outlined in the code description for the target file.
- Write production-quality code: include comments, docstrings, error handling, and follow best practices for the language.
- If the language is not specified, infer it from the filename or description, defaulting to Python if unsure.
- Output *only* the raw code for the target file, enclosed in a single markdown code block (e.g., ```python ... ```). Do not include explanations or introductory text outside the code block.
"""

    user_prompt_parts = [f"## Target File: {target_file or 'unknown'}\n"]
    user_prompt_parts.append(f"## Code Description for Target File:\n{code_description}\n")

    if skeletons:
        user_prompt_parts.append(f"## Code Skeletons for Project Files (Context):\n{skeletons}\n") # Clarified context purpose

    if external_imports:
        user_prompt_parts.append(f"## Required External Imports (for {target_file or 'Target File'}):\n- " + "\n- ".join(external_imports) + "\n") # Clarified scope

    if internal_imports:
        user_prompt_parts.append(f"## Required Internal Imports (for {target_file or 'Target File'}):\n- " + "\n- ".join(internal_imports) + "\n") # Clarified scope

    if current_code_base:
        user_prompt_parts.append(f"## Existing Codebase Context:\n```\n{current_code_base}\n```\n")

    if research_string:
        user_prompt_parts.append(f"## Research Notes:\n{research_string}\n")

    user_prompt_parts.append(f"Please generate the complete code for the target file '{target_file or 'unknown'}' based on its description and the specific imports listed above, using the provided skeletons and context.") # Updated final instruction

    user_prompt = "\n".join(user_prompt_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
