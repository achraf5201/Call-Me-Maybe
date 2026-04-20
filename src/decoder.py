import numpy as np
import json
from llm_sdk import Small_LLM_Model
import time


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)

    def decoder(self, text: str) -> str:
        result = ""
        for key, val in self.vocab.items():
            if key == '{' or key == '"':
                print(key, val)
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