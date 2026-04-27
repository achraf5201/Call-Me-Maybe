import json
import sys
from typing import List
from src.models import FunctionDef, Prompt
from src.parser import parse_args
from src.decoder import ConstrainedDecoder


def build_prompt(user_prompt: Prompt, functions: List[str]) -> str:
    prompt = """<|im_start|>system
        You are a function calling system.\nYou must return a valid json.
        No explanation.
        "Output format:"
        {"name": "function_name","parameters": {"param": value}}"
    <|im_end|>"""
    prompt += f"""alowed function {functions}"""
    prompt += f""" <|im_start|>{user_prompt}
        <|im_end|>
        <|im_start|>assistant
        """
    return prompt


def get_func_name(functions_data) -> List[str]:
    func = []
    for d in functions_data:
        func.append(d["name"])
    print(func)
    return func


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
        _: List[FunctionDef] = [
            FunctionDef(**elem) for elem in functions_data
        ]

        prompts: List[Prompt] = [Prompt(**elem) for elem in prompts_data]

    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)
    texts: List[str] = []
    parsed_prompt = []
    for prompt in prompts:
        texts.append(build_prompt(prompt, functions_data))
        parsed_prompt.append(prompt.model_dump())

    v = ConstrainedDecoder()
    # print(v.state)
    i = 0
    for text in texts:
        print(v.decoder(text, parsed_prompt[i]["prompt"], functions_data))
    #     print("done")
        i += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(e)
