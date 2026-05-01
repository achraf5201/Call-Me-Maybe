import numpy as np
import json
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)

    def check_function(self, functions_name, generated_text: str):
        for func in functions_name:
            if func in generated_text:
                return True
        return False

    def update_state_and_get_allowed_token(self, state, generated_text, functions_name):
        if state == "START" and ' "name":' in generated_text:
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join('"' + str(item) for item in functions_name)
            }

        if state == "FUNCTION_NAME" and self.check_function(functions_name, generated_text):
            return {
                "flag": "forced",
                "state": "GET_QUOTE",
                "allowed": ""
            }
        elif state == "FUNCTION_NAME" and not self.check_function(functions_name, generated_text):
            return {
                "flag": "forced",
                "state": "FUNCTION_NAME",
                "allowed": " ".join('"' + str(item) for item in functions_name)
            }

        if state == "GET_QUOTE" and '", "parameters": {' in generated_text:
            return {
                "flag": "not_forced",
                "state": "GET_PARAM",
                "allowed": ""
            }
        if state == "GET_PARAM" and ("}}" in generated_text or "} }" in generated_text):
            return {
                "flag": "forced",
                "state": 'DONE',
                "allowed": ''
            }
        # if state == "GET_}" and ("}}" in generated_text or "} }" in generated_text):
        #     return {
        #         "flag": "forced",
        #         "state": "DONE",
        #         "allowed": ""
        #     }
        # if state == "GET_PARAM":
        #     return {
        #         "flag": "not_forced",
        #         "state": "ARG",
        #         "allowed": ""
        #     }
        # elif state == "ARG" and '":{' not in generated_text:
        #     return {
        #         "flag": "forced",
        #         "state": "GET_PARAM",
        #         "allowed": '":{'
        #     }

        return {
            "flag": "not_forced",
            "state": state,
            "allowed": ""
        }

    def get_function_name(self, functions_data):
        functions_name = []
        for func in functions_data:
            functions_name.append(func["name"])
        # print(functions_name)
        return functions_name

    def decoder(self, text: str, prompt, functions_data) -> str:
        # ------------------------------------------------------------------------------
        state = "START"
        generated_text = ""
        generated_text += "{" + '"prompt": "' + prompt.replace('"', "\\\"") + '", '
        state_and_token = {}
        functions_name = self.get_function_name(functions_data)
        i = 1
        while True:
            # print("============", state)
            if i == 1:
                generated_text += '"name":'
                print(generated_text, end="")
                i = 2
            state_and_token = self.update_state_and_get_allowed_token(state, generated_text, functions_name)
            state = state_and_token["state"]
            if state == "DONE":
                break
            if state == "GET_QUOTE":
                # print(state)
                # print(state_and_token["allowed"])
                generated_text += '", "parameters": {'
                print('", "parameters": {', end="")
                continue
                # state_and_token = self.update_state_and_get_allowed_token(state, generated_text, functions_name)
                # state = state_and_token["state"]
                # print(state)
                # print(state_and_token["allowed"])
            # print(state_and_token["state"])
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
            elif state_and_token["flag"] == "not_forced":
                next_token_id = int(np.argmax(logits))
                # if '"' in self.model.decode([next_token_id]):
                #     count += 1
                # if 5 > count >= 2:
                #     ids = self.model.encode(' "')[0].tolist()
                #     ids.append(self.model.encode('"')[0].tolist())
                #     # print(ids)
                #     for id in ids:
                #         logits[id] = float("-inf")
                #     next_token_id = int(np.argmax(logits))
                # else:
                #     next_token_id = int(np.argmax(logits))

            token = self.model.decode([next_token_id])
            # print(token)
            print(token, end="", flush=True)
            input_ids.append(next_token_id)
            generated_text += token
            # print(state)
            if "}}" in token:
                break

        return generated_text
