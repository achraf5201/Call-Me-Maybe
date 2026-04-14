from llm_sdk import Small_LLM_Model
import numpy as np

SYSTEM_PROMPT = "You are a helpful math assistant. Answer questions clearly and concisely. And dont think."

if __name__ == "__main__":
    i = 0
    model = Small_LLM_Model()
    user_question = "what is the subtraction of 10 and 1?"

    # Qwen3 chat template format built manually
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_question}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n</think>\n\n"
    )

    print(text, end="", flush=True)
# soft max
# top k
# top q
# greedy
    while i < 150:
        input_ids = model.encode(text)
        logits = model.get_logits_from_input_ids(input_ids[0].tolist())
        next_token_id = int(np.argmax(logits))
        next_word = model.decode([next_token_id])
        text += next_word
        print(next_word, end="", flush=True)
        i += 1

    print()