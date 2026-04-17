from llm_sdk import Small_LLM_Model
import numpy as np


try:
    # SYSTEM_PROMPT = "Answer in one short sentence. No explanations."
    # i = 0
    model = Small_LLM_Model()
    user_question = "what is the subtraction of 10 and 1?"

    # # Qwen3 chat template format built manually
    # text = (
    #     f"{SYSTEM_PROMPT}\nUser: {user_question}\nAssistant:"
    # )

    # print(text, end="", flush=True)

    SYSTEM_PROMPT = "You are a helpful math assistant. Answer briefly."

    text = f"{SYSTEM_PROMPT}\nUser: {user_question}\nAssistant:"

    for i in range(500):
        input_ids = model.encode(text)
        logits = model.get_logits_from_input_ids(input_ids[0].tolist())

        next_token_id = int(np.argmax(logits))
        next_word = model.decode([next_token_id])

        if next_word in ["<|im_end|>", "</s>"]:
            break

        text += next_word
except KeyboardInterrupt:
    print("program stopped")
