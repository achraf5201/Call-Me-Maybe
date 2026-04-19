import json
import sys
from typing import List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDef, Prompt, FunctionCall
from src.parser import parse_args


def build_prompt(user_prompt: Prompt, functions: List[FunctionDef]) -> str:
    text = "You are a function calling system.\n\n"

    text += "Available functions:\n\n"

    for func in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in func.parameters.items()
        )

        text += f"- {func.name}({params}) -> {func.returns.type}\n"
        text += f"  Description: {func.description}\n\n"

    text += "User request:\n"
    text += f"{user_prompt.prompt}\n\n"

    text += "Return ONLY a valid JSON object.\n"
    text += "Do not add explanations or markdown.\n\n"

    text += "Output format:\n"
    text += "{\n"
    text += '  "function": "function_name",\n'
    text += '  "arguments": {\n'
    text += '    "param1": value\n'
    text += "  }\n"
    text += "}\n"

    return text

def main():
    args = parse_args()

    try:
        with open(args.functions_definition) as f:
            functions_data = json.load(f)

        with open(args.input) as f:
            prompts_data = json.load(f)

    except Exception as e:
        print("Error loading files:", e)
        sys.exit(1)

    try:
        definitions: List[FunctionDef] = [
            FunctionDef(**elem) for elem in functions_data
        ]

        prompts: List[Prompt] = [
            Prompt(**elem) for elem in prompts_data
        ]

    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)

    print("PROMPTS:")
    print(prompts)

    print("\nFUNCTIONS:")
    print(definitions)


if __name__ == "__main__":
    main()