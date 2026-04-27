import numpy as np
import json
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)

    def check_function(self, generated_text: str):
        if '"fn_greet"' in generated_text:
            return True
        elif '"fn_add_numbers"' in generated_text:
            return True
        elif '"fn_reverse_string"' in generated_text:
            return True
        elif '"fn_get_square_root"' in generated_text:
            return True
        elif '"fn_substitute_string_with_regex"' in generated_text:
            return True
        return False

    def update_state_and_get_allowed_token(self, state, generated_text, functions_name):
        if state == "START" and '{"name":' in generated_text:
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join('"' + str(item) + '",' for item in functions_name)
            }

        if state == "FUNCTION_NAME" and self.check_function(generated_text):
            return {
                "flag": "forced",
                "state": "GET_QUOTE",
                "allowed": '"'
            }
        elif state == "FUNCTION_NAME" and not self.check_function(generated_text):
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join('"' + str(item) + '",' for item in functions_name)
            }
        if state == "GET_QUOTE" and '"parameters":{' in generated_text:
            return {
                "flag": "forced",
                "state": "GET_PARAM",
                "allowed": 'parameters":{'
            }
        if state == "GET_PARAM" and '"parameters":{' in generated_text:
            return {
                "flag": "not_forced",
                "state": "GET_PARAM",
                "allowed": ""
            }
        return {
            "flag": "not_forced",
            "state": state,
            "allowed": ""
        }


    def get_function_name(self, functions_data):
        functions_name = []
        for func in functions_data:
            functions_name.append(func["name"])
        return functions_name

    def decoder(self, text: str, prompt, functions_data) -> str:
        # ------------------------------------------------------------------------------
        state = "START"
        generated_text = ""
        generated_text += "prompt: " + prompt + "\n"
        state_and_token = {}
        functions_name = self.get_function_name(functions_data)
        i = 1
        while True:
            if i == 1:
                generated_text += '{"name":'
                i = 2
            state_and_token = self.update_state_and_get_allowed_token(state, generated_text, functions_name)
            # state_and_token["flag"]
            # print(state_and_token["flag"])
            input_ids = self.model.encode(text + generated_text)[0].tolist()
            logits = self.model.get_logits_from_input_ids(input_ids)
            allowed_ids = []
            # ------------------------------------------------
            ids = []
            if state_and_token["allowed"]:
                enc = self.model.encode(state_and_token["allowed"])
                ids = enc[0].tolist() if enc is not None else []

            allowed_ids.extend(ids)
            # ------------------------------------------------
            logits = np.array(logits)
            if state_and_token["flag"] == "forced":
                state = state_and_token["state"]
                masked = np.full(len(logits), float("-inf"))
                for idx in allowed_ids:
                    masked[idx] = logits[idx]
                next_token_id = int(np.argmax(masked))
            else:
                next_token_id = int(np.argmax(logits))

            token = self.model.decode([next_token_id])
            input_ids.append(next_token_id)
            generated_text += token
            if "}}" in token:
                break
            print(generated_text)

        return generated_text
