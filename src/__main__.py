import json
import sys
from typing import List
from src.models import FunctionDef, Prompt
from src.parser import parse_args
from src.decoder import ConstrainedDecoder
import os
import time


PINK = "\033[95m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def type_writer(text, speed=0.002):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()


def show_header():
    header = f"""
{PINK}{BOLD}
   ██████╗ █████╗ ██╗     ██╗     ███╗   ███╗███████╗███╗   ███╗ █████╗ ██╗   ██╗██████╗ ███████╗
  ██╔════╝██╔══██╗██║     ██║     ████╗ ████║██╔════╝████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝
  ██║     ███████║██║     ██║     ██╔████╔██║█████╗  ██╔████╔██║███████║ ╚████╔╝ ██████╔╝█████╗  
  ██║     ██╔══██║██║     ██║     ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██╗██╔══╝  
  ╚██████╗██║  ██║███████╗███████╗██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║██║  ██║   ██║   ██████╔╝███████╗
   ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚══════╝


{CYAN}         ✨ CALL ME MAYBE PROJECT ✨{RESET}

{YELLOW}         Initializing... Please wait 🚀{RESET}
"""
    type_writer(header, speed=0.0005)


def build_prompt(user_prompt: Prompt, functions: List[str]) -> str:
    prompt = """<|im_start|>system
        You are a function calling system.\nYou must return a valid json.
        "Output format:
        {"name": "fn_func", "parameters": {"param": value}}"
    <|im_end|>"""
    prompt += f"""alowed function {functions} format of all functions"""
    prompt += """exemple:
    {"name": "fn_format_template","parameters": {"template": "Say \"hello\" to {name}"}}"""
    prompt += f""" <|im_start|>{user_prompt}
        <|im_end|>
        <|im_start|>assistant
        """
    return prompt


def remove_folder(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)


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
        _: List[FunctionDef] = [FunctionDef(**elem) for elem in functions_data]

        prompts: List[Prompt] = [Prompt(**elem) for elem in prompts_data]

    except Exception as e:
        print("Validation error:", e)
        sys.exit(1)
    texts: List[str] = []
    parsed_prompt = []
    for prompt in prompts:
        texts.append(build_prompt(prompt, functions_data))
        ready_prompt = prompt.model_dump()["prompt"].replace("\\", "\\\\")
        parsed_prompt.append(ready_prompt)

    v = ConstrainedDecoder()
    i = 0
    arr = []
    for text in texts:
        arr.append(v.decoder(text, parsed_prompt[i], functions_data))
        i += 1
    with open("data/output/function_calling_results.json", "w") as f:
        try:
            data = []
            for s in arr:
                data.append(json.loads(s))
            for d in data:
                for key in d["parameters"]:
                    if key == "a" or key == "b":
                        try:
                            d["parameters"][key] = float(d["parameters"][key])
                        except ValueError:
                            pass
                    if key == "replacement":
                        try:
                            d["parameters"][key] = d["parameters"][
                                key
                            ].replace(" ", "")
                        except ValueError:
                            pass
            json.dump(data, f, indent=4)
        except Exception:
            pass


if __name__ == "__main__":
    clear_terminal()
    show_header()

    time.sleep(1)
    print(f"{CYAN}Welcome to your project!{RESET}")
    try:
        main()
    except KeyboardInterrupt as e:
        print(e)