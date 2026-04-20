import numpy as np
import json
from llm_sdk import Small_LLM_Model
from src.stage import Stage


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)

    def update_state(self, state, token):
        if state == "START" and token == "{":
            return "ARGUMENTS"
        if state == "ARGUMENTS" and token == "a":
            return "VALUE_A"
        if state == "VALUE_A" and token.isdigit():
            return "AFTER_A"
        if state == "AFTER_A" and token == "b":
            return "VALUE_B"
        if state == "VALUE_B" and token.isdigit():
            return "DONE1"
        if state == "DONE1" and token == "}":
            return "DONE2"
        if state == "DONE2" and token == "}":
            return "DONE"
        return state

    def get_allowed(self, state):
        if state == "START":
            return [self.vocab["{"]]

        # if state == "ARGUMENTS":
        #     return [self.vocab['"a"']]

        if state == "VALUE":
            return list(range(10))

        if state == "DONE":
            return [self.vocab["}"]]

        return list(range(len(self.vocab)))

    def decoder(self, text: str) -> str:
        state = "START"
        
        # Fix: strip batch dimension if encode returns [[...]]
        input_ids = self.model.encode(text).tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        
        result = ""

        while state != "DONE":
            # 1. Get model predictions
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits)
            if logits.ndim == 2:
                logits = logits[-1]

            # 2. Mask logits to allowed tokens only
            allowed = self.get_allowed(state)
            masked = np.full(len(logits), -np.inf)
            masked[allowed] = logits[allowed]

            # 3. Pick next token (greedy)
            next_token_id = int(np.argmax(masked))

            # 4. Decode and accumulate
            token = self.model.decode([next_token_id])
            input_ids.append(next_token_id)
            result += token
            print(result)

            # 5. Advance FSM state
            state = self.update_state(state, token)
            print(state)
            if state == "DONE":
                break

        return result


# input_ids = tokenizer("", return_tensors="pt").input_ids

# generated = ""

# for step in range(5):
#     outputs = model(input_ids)
#     logits = outputs.logits[:, -1, :]

#     # Get allowed tokens for this step
#     allowed = allowed_tokens(step)
#     allowed_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in allowed]

#     # Remove all tokens NOT allowed
#     for token_id in range(logits.shape[-1]):
#         if token_id not in allowed_ids:
#             logits[0, token_id] = -float("inf")

#     # Pick next token
#     next_token_id = logits.argmax(dim=-1).unsqueeze(0)
#     input_ids = next_token_id
#     token = tokenizer.decode(next_token_id[0])

#     generated += token

# print("Generated:", generated)
