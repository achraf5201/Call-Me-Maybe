import numpy as np
import json
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)
            self.v = {
                value: key.replace("Ġ", " ").replace("Ċ", "\n")
                for key, value in self.vocab.items()
            }
        self.closings = {
            vid: ss.count("{") - ss.count("}")
            for ss, vid in self.vocab.items()
        }
        self.brace_weight = 2
        self.output = []

    def check_function(self, functions_name, generated_text: str):
        for func in functions_name:
            if func in generated_text:
                return True
        return False

    def update_state_and_get_allowed_token(
        self, state, generated_text, functions_name
    ):
        if state == "START" and ' "name":' in generated_text:
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join(
                    '"' + str(item) for item in functions_name
                ),
            }

        if state == "FUNCTION_NAME" and self.check_function(
            functions_name, generated_text
        ):
            return {"flag": "forced", "state": "GET_QUOTE", "allowed": ""}
        elif state == "FUNCTION_NAME" and not self.check_function(
            functions_name, generated_text
        ):
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join(
                    '"' + str(item) for item in functions_name
                ),
            }

        if state == "GET_QUOTE" and '", "parameters": {' in generated_text:
            return {"flag": "not_forced", "state": "GET_PARAM", "allowed": ""}

        if state == "GET_PARAM" and (
            "}}" in generated_text or "} }" in generated_text
        ):
            return {"flag": "forced", "state": "DONE", "allowed": ""}

        return {"flag": "not_forced", "state": state, "allowed": ""}

    def get_function_name(self, functions_data):
        functions_name = []
        for func in functions_data:
            functions_name.append(func["name"])
        return functions_name

    def decoder(self, text: str, prompt, functions_data) -> str:
        i = 0
        state = "START"
        generated_text = (
            "{" + '"prompt": "' + prompt.replace('"', '\\"') + '", '
        )
        state_and_token = {}
        functions_name = self.get_function_name(functions_data)
        i = 1
        while True:
            if i == 1:
                generated_text += '"name":'
                print(generated_text, end="")
                i = 2
            state_and_token = self.update_state_and_get_allowed_token(
                state, generated_text, functions_name
            )
            state = state_and_token["state"]
            if state == "DONE":
                break
            if state == "GET_QUOTE":
                generated_text += '", "parameters": {'
                print('", "parameters": {', end="")
                continue
            # ------------------ param
            for func in functions_data:
                if func["name"] in generated_text and generated_text.endswith(
                    '", "parameters": {'
                ):
                    key = list(func["parameters"].items())
                    print(f' "{key[0][0]}": ', end="", flush=True)
                    generated_text += f' "{key[0][0]}": '
            # ------------------ end param
            input_ids = self.model.encode(text + generated_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_ids = []

            ids = []
            enc = self.model.encode(state_and_token["allowed"])
            ids = enc[0].tolist() if enc is not None else []

            allowed_ids.extend(ids)

            logits = np.array(logits)
            if state_and_token["flag"] == "forced":
                state = state_and_token["state"]
                masked = np.full(len(logits), float("-inf"))
                for idx in allowed_ids:
                    masked[idx] = logits[idx]
                next_token_id = int(np.argmax(masked))
            elif state_and_token["flag"] == "not_forced":
                next_token_id = int(np.argmax(logits))

            token = self.v.get(next_token_id)
            print(token, end="", flush=True)
            generated_text += token

            if self.brace_weight <= 0:
                break
            if "}}" in token:
                break
        return generated_text
