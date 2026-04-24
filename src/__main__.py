import json
import sys
from typing import List
from src.models import FunctionDef, Prompt
from src.parser import parse_args
from src.decoder import ConstrainedDecoder


def build_prompt(user_prompt: Prompt, functions: List[FunctionDef]) -> str:
    text = "You are a function calling system.\n\n"

    text += "Available functions:\n\n"

    for func in functions:
        params = ", ".join(
            f"{name}: {param.type}" for name, param in func.parameters.items()
        )

        text += f"- {func.name}({params}) -> {func.returns.type}\n"
        text += f"  Description: {func.description}\n\n"

    text += "Output format:"
    text += "{"
    text += 'function": "function_name",'
    text += '"parameters": {'
    text += '"a b": value'
    text += "}"
    text += "}"
    text += f"prompt: {user_prompt}\n"

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

        prompts: List[Prompt] = [Prompt(**elem) for elem in prompts_data]

    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)
    texts: List[str] = []
    for prompt in prompts:
        texts.append(build_prompt(prompt, definitions))
        print(prompt)
    i = 0
    v = ConstrainedDecoder()
    # print(v.state)
    for text in texts:
        print(v.decoder(text))
        print("done")
        i += 1


if __name__ == "__main__":
    main()
