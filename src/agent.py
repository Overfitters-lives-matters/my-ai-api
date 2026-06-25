import json
import re
from src.tools import TOOLS
from src.llm_client import LLMClient


SYSTEM_PROMPT = """You are a research agent. Your job is to:
1. Search the web to gather information on the user's topic
2. Synthesize the findings into a well-structured report
3. Save the report to a file using file_write
4. Always save a file — this is mandatory, not optional

You have access to these tools:

- web_search(query): Search the web for information
- file_write(filename, content): Save content to documents/ folder

To use a tool, respond EXACTLY in this format (no extra text before it):

TOOL: tool_name
PARAMS: {"param1": "value1", "param2": "value2"}

After seeing the tool result, continue reasoning. When you have searched enough (at least 2-3 searches) and written the file, respond with:

FINAL: <your summary to the user of what you found and saved>

Rules:
- Always do at least 2 web searches before writing the file
- The file content must be detailed and well-structured (use headings, bullet points)
- The filename should reflect the topic (e.g., "ai_trends_2025.txt")
- Never stop before saving the file
"""


def parse_tool_call(text: str):
    """Extract tool name and params from LLM output."""
    tool_match = re.search(r"TOOL:\s*(\w+)", text)
    params_match = re.search(r"PARAMS:\s*(\{.*?\})", text, re.DOTALL)

    if not tool_match:
        return None, None

    tool_name = tool_match.group(1).strip()
    params = {}

    if params_match:
        try:
            params = json.loads(params_match.group(1))
        except json.JSONDecodeError:
            pass

    return tool_name, params


def run_tool(tool_name: str, params: dict) -> str:
    """Execute a tool and return its result."""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"

    tool = TOOLS[tool_name]
    fn = tool["fn"]

    try:
        result = fn(**params)
        return result
    except TypeError as e:
        return f"Tool call error (wrong params): {str(e)}"
    except Exception as e:
        return f"Tool execution error: {str(e)}"


class Agent:
    def __init__(self, llm_client: LLMClient, max_steps: int = 10):
        self.llm = llm_client
        self.max_steps = max_steps

    def run(self, user_query: str) -> dict:
        """
        Run the ReAct agent loop.
        Returns a dict with the final answer and a log of all steps.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        steps = []
        step_count = 0

        while step_count < self.max_steps:
            step_count += 1

            # Get LLM decision
            llm_response = self.llm.stream_response(messages)

            # Check for final answer
            if "FINAL:" in llm_response:
                final_match = re.search(r"FINAL:\s*(.*)", llm_response, re.DOTALL)
                final_answer = final_match.group(1).strip() if final_match else llm_response
                steps.append({"step": step_count, "type": "final", "content": final_answer})
                return {"answer": final_answer, "steps": steps}

            # Parse tool call
            tool_name, params = parse_tool_call(llm_response)

            if not tool_name:
                # LLM didn't call a tool or give a final — nudge it
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": "Continue. Use a tool or provide your FINAL answer."
                })
                continue

            # Log the step
            steps.append({
                "step": step_count,
                "type": "tool_call",
                "tool": tool_name,
                "params": params
            })

            # Execute the tool
            tool_result = run_tool(tool_name, params)

            steps.append({
                "step": step_count,
                "type": "tool_result",
                "tool": tool_name,
                "result": tool_result
            })

            # Feed result back into conversation
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{tool_result}\n\nContinue."
            })

        return {
            "answer": "Agent reached maximum steps without completing.",
            "steps": steps
        }