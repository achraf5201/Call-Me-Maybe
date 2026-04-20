import numpy as np
from llm_sdk import Small_LLM_Model


def decoder(text: str) -> str:
    result = ""
    model = Small_LLM_Model()
    input_ids = model.encode(text)
    output_ids = input_ids[0].tolist()
    for _ in range(150):
        logits = model.get_logits_from_input_ids(output_ids)
        next_token_id = int(np.argmax(logits))
        output_ids.append(next_token_id)
        result += model.decode([next_token_id])
        print(result)
    return result
