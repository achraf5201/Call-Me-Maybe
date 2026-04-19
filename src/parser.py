import argparse
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Function calling program using LLM"
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to save the output JSON file"
    )

    return parser.parse_args()
