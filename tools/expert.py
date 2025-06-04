# expert.py
from openai import OpenAI

# load the API key from the environment
from dotenv import load_dotenv
from icecream import ic
from typing import Optional, Literal
from .base import ToolError, ToolResult, BaseAnthropicTool, ToolFailure

load_dotenv()
from rich import print as rr


class GetExpertOpinionTool(BaseAnthropicTool):
    """
    A tool takes a detailed description of the problem and everything that has been tried so far, and returns an expert opinion on the problem.
    """

    name: Literal["Opinion"] = "opinion"
    api_type: Literal["custom"] = "custom"
    description: str = "A tool takes a detailed description of the problem and everything that has been tried so far, and returns an expert opinion on the problem."

    def to_params(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["get_opinion"],
                            "description": "The command to get an expert opinion.",
                        },
                        "problem_description": {
                            "type": "string",
                            "description": "A detailed description of the problem and everything that has been tried so far. If for programming, include the code that has been tried.",
                        },
                    },
                    "required": ["command", "problem_description"],
                },
            },
        }

    async def __call__(
        self,
        *,
        command: Literal["get_opinion"],
        problem_description: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Executes the specified command.
        """

        if command == "get_opinion":
            return await self.get_opinion(problem_description=problem_description)
        if command == "get_plan":
            return await self.get_plan(problem_description=problem_description)
        else:
            raise ToolError(
                f"Unrecognized command '{command}'. Allowed commands: 'list_reports', 'run_report'."
            )

    async def get_plan(self, problem_description) -> ToolResult:
        prompt = f"""
Objective:
You are a Task Decomposition Specialist. Your goal is to meticulously break down any given computer-based task into the smallest reasonable steps. Each step should be clear, actionable, and independently verifiable for completion by someone without prior knowledge of the task or access to the execution environment.
Instructions:
For each step (and sub-step, if necessary), provide the following:
Step [Number]:
Action: A detailed and specific description of the action to be performed.
Expected Result: A description of what should occur or be produced after the action is completed.
Verification Method: A precise method to confirm that the step has been completed, which should:
Be executable by someone who has not seen the previous steps.
Not require access to the environment where the task was performed.
Focus on confirming the completion of the action, not the accuracy of the result (unless completion inherently requires accuracy).
Guidelines:
Clarity and Specificity:
Use clear, unambiguous language.
Include specific details such as file names, URLs, commands, or search queries where applicable.
Step Structure:
Main Steps: Numbered sequentially (e.g., Step 1, Step 2, Step 3).
Sub-Steps: If a step requires multiple actions, break it down into sub-steps (e.g., Sub-Step 1.1, Sub-Step 1.2).
Formatting Requirements:
Use bold headings for Action, Expected Result, and Verification Method for clarity.
Present information in a structured and organized manner.
Independent Verification:
Ensure each verification method can be performed independently of other steps.
Verification should rely only on the outputs or artifacts produced by that specific step.
Verification Methods:
Should be objective, measurable, and specific.
May include:
Checking the existence or properties of a file or document.
Viewing metadata, timestamps, or file contents.
Confirming the presence of specific data, entries, or outputs.
Reviewing screenshots or exported logs.
No Assumptions:
Do not assume the verifier has any prior knowledge of the task or access to previous results.
Do not require the verifier to access the execution environment or external systems beyond what is produced in the step.
Example Format:
Step 1:
Action: Create a new folder named Elevator_Project on your desktop.
Expected Result: A folder named Elevator_Project exists on the desktop.
Verification Method: Check the desktop for the presence of the Elevator_Project folder.
Sub-Step 1.1:
Action: Inside the Elevator_Project folder, create an Excel file named elevator_data.xlsx with two sheets labeled Company_Info and Models.
Expected Result: An Excel file elevator_data.xlsx with two sheets named Company_Info and Models exists in the Elevator_Project folder.
Verification Method: Open elevator_data.xlsx and confirm the sheets Company_Info and Models are present.
Sub-Step 1.2:
Action: In the Company_Info sheet, add column headers in the first row: Manufacturer, Public/Private, Number of Employees, Years in Business.
Expected Result: The Company_Info sheet has the specified headers correctly labeled in the first row.
Verification Method: Open the Company_Info sheet and verify the headers are present and correctly labeled.
Your Task:
{problem_description}
Apply the above guidelines and format to break down the provided task:
Instructions for Completion:
Identify Main Steps:
Start by outlining the major components required to complete the task.
Decompose into Sub-Steps:
Break down each main step into smaller, actionable sub-steps as necessary.
Detail Each Step:
For every step and sub-step, provide the Action, Expected Result, and Verification Method as per the guidelines.
Ensure Independent Verification:
Make sure that each verification method allows someone to confirm completion without prior knowledge or access to the execution environment.
Maintain Clarity and Organization:
Use the specified formatting for consistency and ease of understanding.
Begin your detailed task breakdown below:
        """
        return ToolResult(output=prompt)

    async def get_opinion(self, problem_description) -> ToolResult:
        """
        Lists all available reports.
        """
        try:
            client = OpenAI()
            ic(client)
            prompt = f"""
            You are an expert in the field of computer science.
            You will be asked to provide an expert opinion on a computer science problem.
            Typically, it will about 1 of 3 things you are asked about:
            1. The user is having a coding problem that they can't solve.
                a. In this case, give your best advice on how to approach the problem and provide revevent code suggestions
                b. The user is a VERY competent programmer, so you can assume they know all the basics. They usually just have a hard time piecing it all together or have a conflict in their environment.
                c. You can just be their coach and give deep advice on how to approach the problem.
            2. The user has a problem with the coding environment.
                a. The user is very good at navigating a Linux environment, however they are currently coding in Windows. 
                b You need to remind them that they are in Windows, and give detailed instructions of what commands to use and remind them of what commands will not work. 
                c. Most of what they need to do is with Powershell, but sometimes keyboard shortcuts can help them. 
                d. you can also suggest they use pyperclip and pyautogui to automate some of the tasks. 
                e. You will have to be very specific in your Windows commands.
            3. The last thing you will be asked is to suggest improvements to the code.
                a. While talented, the user can be lazy so your job will be to push them to their limits of capabilities.
                b. You need to help make them their best so really push them to make something better (whether peformance, ui or adding additional features).
                c. They have the capability to be really great with your help so don't hold back giving stretch suggestions and push yourself to help create something better than anyone has thought of before. 
        
           The user would like your expert opinion on the following problem:
           {problem_description}
            """
            rr(problem_description)
            response = client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            ic(response)
            ex_opinion = response.choices[0].message.content
            return ToolResult(output=ex_opinion)
        except Exception as e:
            ic(e)
            rr(f"{str(e).encode('ascii', errors='replace').decode('ascii')}")
            return ToolFailure(error="Failed to generate opinion.")
