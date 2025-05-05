from config import get_constant 
from typing import List, Dict, Optional, Any


def code_skeleton_prompt(
    code_description: str,
    target_file: str,
    agent_task: str,
    external_imports: Optional[List[str]] = None,
    internal_imports: Optional[List[str]] = None,
    all_file_details: Optional[
        List[Dict[str, Any]]
    ] = None,  # Add context of other files
    ) -> list:
    """
    Creates a prompt that asks the LLM to generate code skeleton/structure
    based on a description, considering the broader project context.

    Args:
        code_description (str): Detailed description of what code to generate for the target file.
        target_file (str): The name/path of the file for which the skeleton is being generated.
        agent_task (str): The overall goal or task of the agent/project.
        external_imports (Optional[List[str]]): List of external libraries specifically needed for this file.
        internal_imports (Optional[List[str]]): List of internal modules/files specifically needed by this file.
        all_file_details (Optional[List[Dict[str, Any]]]): List of dictionaries, each containing details
                                                            (filename, description) about all files in the project.

    Returns:
        list: Formatted messages for the LLM prompt
    """
    system_prompt = f"""You are an expert software architect specializing in creating clean, well-structured code skeletons.
    Your task is to create a comprehensive code structure (skeleton) for a specific file within a larger project.
    This skeleton should include proper imports, class definitions, method/function signatures, and detailed docstrings, but WITHOUT implementation details (use 'pass' or placeholder comments).

    Overall Project Goal: {agent_task}

    Target File for this Skeleton: {target_file}

    You will be given:
    1. A detailed description of the code required for the *target file*.
    2. A list of required *external* libraries/packages *specifically for the target file*.
    3. A list of required *internal* modules/files within the project *imported specifically by the target file*.
    4. Context about *all* files planned for the project (names and descriptions).

    Instructions:
    - Focus *only* on generating the skeleton for the specified *target file*: **{target_file}**.
    - Include ALL necessary imports at the top, based on the provided lists and the code description. Prioritize the provided lists.
    - Define ALL necessary classes with proper inheritance (if applicable).
    - Include ALL necessary functions and methods with proper signatures (parameters with type hints, return types).
    - Write DETAILED docstrings for all classes, methods, and functions explaining their purpose, args, and returns.
    - Use proper typing annotations throughout (`typing` module).
    - Include constructor methods (`__init__`) where appropriate, initializing attributes mentioned or implied in the description.
    - Use `pass` for the body of functions and methods. Add brief `# TODO: Implement logic` comments if complexity warrants it.
    - Consider the descriptions of other project files (`all_file_details`) to anticipate necessary interactions or structures, but *only* generate the skeleton for the *target file*.
    - Follow PEP 8 standards for Python code (or relevant style guides for other languages).
    - Output *only* the raw code skeleton for the target file, enclosed in a single markdown code block (e.g., ```python ... ```). Do not include explanations or introductory text outside the code block.
    """

    user_prompt_parts = [f"## Target File: {target_file}\n"]
    user_prompt_parts.append(
        f"## Code Description for Target File Skeleton:\n{code_description}\n"
    )

    if external_imports:
        user_prompt_parts.append(
            f"## Required External Imports (for {target_file}):\n- "
            + "\n- ".join(external_imports)
            + "\n"
        )
    else:
        user_prompt_parts.append(
            f"## No Specific External Imports Provided for {target_file}.\n   (Infer necessary external imports from the description)\n"
        )

    if internal_imports:
        user_prompt_parts.append(
            f"## Required Internal Imports (for {target_file}):\n- "
            + "\n- ".join(internal_imports)
            + "\n"
        )
    else:
        user_prompt_parts.append(
            f"## No Specific Internal Imports Provided for {target_file}.\n   (Infer necessary internal imports from the description and other file details)\n"
        )

    if all_file_details:
        file_context = "\n".join(
            [
                f"- {f.get('filename', 'N/A')}: {f.get('code_description', 'No description')}"
                for f in all_file_details
            ]
        )
        user_prompt_parts.append(
            f"## Overall Project File Structure (Context):\n{file_context}\n"
        )
    else:
        user_prompt_parts.append(
            "## No overall project file structure context available.\n"
        )

    user_prompt_parts.append(
        f"Please generate ONLY the complete code skeleton for the target file '{target_file}' based on its description, considering the provided imports and overall project context."
    )

    user_prompt = "\n".join(user_prompt_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def code_prompt_generate(
    current_code_base: str,
    code_description: str,
    research_string: str,
    agent_task: str,
    skeletons: Optional[str] = None,
    external_imports: Optional[List[str]] = None,
    internal_imports: Optional[List[str]] = None,
    target_file: Optional[str] = None
    )-> list:
    """
    Generates the prompt messages for code generation, incorporating skeletons and file-specific import lists.
    """
    ## ________________________________________________ ##

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
    else:
        user_prompt_parts.append("## No Code Skeletons Currently Available\n")  # Added handling for no skeletons

    if external_imports:
        user_prompt_parts.append(f"## Required External Imports (for {target_file or 'Target File'}):\n- " + "\n- ".join(external_imports) + "\n") # Clarified scope
    else:
        user_prompt_parts.append("## No External Imports Currently Available\n")  # Added handling for no external imports
    if internal_imports:
        user_prompt_parts.append(f"## Required Internal Imports (for {target_file or 'Target File'}):\n- " + "\n- ".join(internal_imports) + "\n") # Clarified scope
    else:
        user_prompt_parts.append("## No Internal Imports Currently Available\n")  # Added handling for no internal imports

    if current_code_base:
        user_prompt_parts.append(f"## Existing Codebase Context:\n```\n{current_code_base}\n```\n")
    else:
        user_prompt_parts.append("## No Existing Codebase Context Available\n")  # Added handling for no existing codebase

    if research_string:
        user_prompt_parts.append(f"## Research Notes:\n{research_string}\n")
    else:
        user_prompt_parts.append("## No Research Notes Currently Available\n")  # Added handling for no research notes

    user_prompt_parts.append(f"Please generate the complete code for the target file '{target_file or 'unknown'}' based on its description and the specific imports listed above, using the provided skeletons and context.") # Updated final instruction

    user_prompt = "\n".join(user_prompt_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


PROMPT_FOR_CODE = """I need you to break down a python file in a structured way that concisely describes how to interact with the code.  
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
    messages = [
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
                {code_description}""",
                },
            ],
        }
    ]
    return messages
