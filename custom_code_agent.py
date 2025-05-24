import os
import sys
import subprocess
from dotenv import load_dotenv
from typing import Type, List

from pydantic import BaseModel, Field 

from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq

load_dotenv()

#1. Define the Custom Tool 
class PythonCodeExecutorToolInput(BaseModel):
    code: str = Field(description="The Python code to execute. It should be a complete, runnable script without any markdown formatting.")

class PythonCodeExecutorTool(BaseTool):
    name: str = "python_code_executor"
    description: str = (
        "Executes a given snippet of Python code and returns its standard output and standard error. "
        "Use this tool ONLY for running Python code. "
        "The input 'code' MUST be raw Python code only, without any surrounding text, explanations, or markdown "
        "fences (like ```python or ```). "
        "Ensure the Python code is self-contained and prints any results to standard output (e.g., using `print(result)`)."
    )
    args_schema: Type[BaseModel] = PythonCodeExecutorToolInput

    def _clean_code(self, code_input: str) -> str:
        """Removes common markdown fences and leading/trailing whitespace."""
        code = code_input.strip()
    
        if code.startswith("```python"):
            code = code[len("```python"):].strip()

        elif code.startswith("```"):
            code = code[len("```"):].strip()
        
        if code.endswith("```"):
            code = code[:-len("```")].strip()
        return code

    def _run(self, code: str) -> str: 
        """Executes the python code after cleaning and returns stdout/stderr."""
        
        cleaned_code = self._clean_code(code) 

        print("\n--- PROPOSED CLEANED CODE ---")
        print(cleaned_code)
        print("---------------------------\n")

        if not cleaned_code:
            return "Error: No valid Python code provided after cleaning. The input might have been empty or only markdown."

     
        confirm = input(f"Do you want to execute this cleaned code? [y/N]: ")
        if confirm.lower() != 'y':
            return "Code execution CANCELED by user."

        try:
            process = subprocess.run(
                [sys.executable, '-c', cleaned_code],
                capture_output=True,
                text=True,
                timeout=60, 
                check=False 
            )

            output_parts = []
            if process.stdout:
                output_parts.append(f"Standard Output:\n{process.stdout.strip()}")
            if process.stderr:
                output_parts.append(f"Standard Error:\n{process.stderr.strip()}")
            
            result_message = "\n".join(output_parts)

            if not result_message and process.returncode == 0:
                result_message = "Code executed successfully with no output to stdout or stderr."
            elif not result_message and process.returncode != 0:
                result_message = f"Code execution failed with return code {process.returncode} and no specific error message."
            elif process.returncode != 0 and "Standard Error" not in result_message: 
                 result_message += f"\nCode execution finished with return code: {process.returncode}"

            return result_message.strip()

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 60 seconds."
        except Exception as e:
            return f"An unexpected error occurred during Python code execution: {str(e)}"

    async def _arun(self, code: str) -> str:
      
        return self._run(code)

#2. Initialize LLM and Tools
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in environment variables.")
    print("Please create a .env file with GROQ_API_KEY='your_key_here'")
    sys.exit(1)

try:
    llm = ChatGroq(
        temperature=0.05, 
        model_name="llama3-8b-8192",
        groq_api_key=GROQ_API_KEY
    )
   
    llm.invoke("Respond with only 'OK'.") 
    print("Groq LLM initialized and tested successfully.")
except Exception as e:
    print(f"Error initializing or testing Groq LLM: {e}")
    print("Please ensure your GROQ_API_KEY is correct and the model name is valid.")
    print("Check available models at https://console.groq.com/docs/models")
    sys.exit(1)

tools: List[BaseTool] = [PythonCodeExecutorTool()]

#3. Define the Agent Prompt for ReAct 
REACT_PROMPT_TEMPLATE = """
You are a specialized AI assistant that generates and executes Python code to answer user requests.
You MUST use the 'python_code_executor' tool to run any Python code.

TOOLS:
------
You have access to the following tools:
{tools}

RESPONSE FORMAT:
--------------------
Follow this strict format for your responses:

Question: The user's request.
Thought: Briefly explain your plan to address the question. If the task is complex, break it into smaller, runnable Python code steps. Each step's code should print its result or status.
Action: The tool to use. This MUST be one of [{tool_names}].
Action Input: The input for the tool.
  - For 'python_code_executor', this MUST be ONLY the raw Python code.
  - Do NOT include any explanations, comments outside the code, or markdown like ```python or ```.
  - The Python code MUST be self-contained and print its output using `print()`.
Observation: The result from the tool (this will be provided by the system).
... (Repeat Thought/Action/Action Input/Observation as needed)
Thought: I have all the information needed to answer the user.
Final Answer: Provide the final answer to the user's question, based on the observations. If code was executed, summarize what it did and its output.

IMPORTANT RULES:
1.  Always use the `python_code_executor` for any Python execution.
2.  The `Action Input` for `python_code_executor` must be ONLY raw Python code. No markdown, no fluff.
3.  Ensure generated Python code prints its results.
4.  If a task is complex (e.g., building a game), generate code in smaller, verifiable steps.

HANDLING USER CANCELLATION:
---------------------------
If an Observation indicates "Code execution CANCELED by user":
Thought: The user canceled the last proposed code. I should ask if they want to modify the code, try a different approach, or if I should abandon this line of inquiry for the current question.
Action: (Decide if another tool use is appropriate, or if it's time for a Final Answer stating the cancellation)
Action Input: (If applicable)
Final Answer: (If abandoning, e.g., "User canceled code execution. How can I help further?")

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

#4. Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

#5. Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,        

)

#6. Main Interaction Loop
def main():
    print("\nVerified Code Agent (Groq - Llama3-8B) Initialized.")
    print("Type 'exit' to quit.")
    print("Agent will attempt to generate and execute Python code based on your prompts.")
    print("For complex tasks, the agent may need several steps or might not complete it perfectly in one go.")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            print("Exiting agent.")
            break
        if not user_input.strip():
            continue

        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"\nAn error occurred during agent execution: {e}")
            # import traceback 
            # traceback.print_exc()

if __name__ == "__main__":
    main()