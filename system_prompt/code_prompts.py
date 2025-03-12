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


def code_prompt_generate(current_code_base, code_description, code_skeleton):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""At the bottom is a detailed description of code that you need to write, followed by a code skeleton that provides the structure.
            
            Your task is to implement the full code based on the description and skeleton. Make sure you provide your response in the requested programming language. Your response should include the code for the entire file including proper imports lines, but do not actually include the code from other files.

            You should print verbose output to the console to show the progress of the code execution. This will help the user understand what is happening at each step of the process.
            
            If you are requested to make changes to an existing file, please provide the full file with the correction made as your response. Please keep your changes limited to what was requested and keep the rest of the code unchanged.
            
            If you see possible errors that are outside of the scope of the request, please make a note of them using inline comments in the code. Provide additional explanation as needed in comments.
            
            If you see additional improvements that could be made to the code, please make a note of them at the bottom of the code in the comments, but do not make the changes.
            
            You may be occasionally asked to provide non-typical file responses, such as Markdown, Readme, ipynb, txt, etc. Please provide the file in the requested format instead of providing code that would generate the file.

            All of the code that you provide needs to be enclosed in a single markdown style code block with the language specified.
            Here is an example of what your response should look like:
            ```python
            # Your code goes here
            ```
            If it was javascript it would look like this:
            ```javascript
            // Your code goes here
            ```
            
            Here is all of the code that has been created for the project so far:
            Code:
            {current_code_base}
            
            Here is the description of the code:
            {code_description}
            
            Here is the code skeleton to implement:
            {code_skeleton}
            """,
                },
            ],
        }
    ]
    return messages
