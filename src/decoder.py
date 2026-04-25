import numpy as np
import json
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self):
        self.model = Small_LLM_Model()
        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.load(f)
        self.state = {
            "START":
            [
                {
                    "alowed_tokens": "{",
                    "next_state": "EXPECT_START_BRACES"
                }
            ],
            "EXPECT_START_BRACES":
            [
                {
                    "alowed_tokens": "\"",
                    "next_state": "EXPECT_START_QOUTS"
                }
            ],
            "EXPECT_START_QOUTS":
            [
                {
                    "alowed_tokens": "name",
                    "next_state": "EXPECT_NAME"
                }
            ],
            "EXPECT_NAME":
            [
                {
                    "alowed_tokens": "\"",
                    "next_state": "EXPECT_END_QOUTS"
                }
            ],
            "EXPECT_END_QOUTS": [
                {
                    "alowed_tokens": ":",
                    "next_state": "EXPECT_COLON_AFTER_NAME"
                }
            ],
            # "EXPECT_COLON_AFTER_NAME": [
            #     {
            #         "alowed_tokens": "\"fn_add_numbers\"",
            #         "next_state": "fn_add_numbers_FUN_NAME"
            #     },
            #     {
            #         "alowed_tokens": "\"fn_greet\"",
            #         "next_state": "fn_greet_FUN_NAME"
            #     },
            #     {
            #         "alowed_tokens": "\"fn_reverse_string\"",
            #         "next_state": "fn_reverse_string_FUN_NAME"
            #     },
            #     {
            #         "alowed_tokens": "\"fn_get_square_root\"",
            #         "next_state": "fn_get_square_root_FUN_NAME"
            #     },
            #     {
            #         "alowed_tokens": "\"fn_substitute_string_with_regex\"",
            #         "next_state": "fn_substitute_string_with_regex_FUN_NAME"
            #     }
            # ],
            # for fn_add_numbers
            "fn_add_numbers_FUN_NAME": [
                {
                    "alowed_tokens": ",",
                    "next_state": "fn_add_numbers_COMMA_BEFORE_PARAMS"
                },
            ],
            "fn_add_numbers_COMMA_BEFORE_PARAMS": [
                {
                    "alowed_tokens": "\"parameters\"",
                    "next_state": "fn_add_numbers_PARAMS_KEY"
                }
            ],
            "fn_add_numbers_PARAMS_KEY": [
                {
                    "alowed_tokens": ":",
                    "next_state": "fn_add_numbers_PARAM_KEY_COLON"
                }
            ],
            "fn_add_numbers_PARAM_KEY_COLON": [
                {
                    "alowed_tokens": "{",
                    "next_state":
                    "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS"
                }
            ],
            "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS": [
                {
                    "alowed_tokens": "\"a\"",
                    "next_state":
                    "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS_KEY_ARG"
                }
            ],
            "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS_KEY_ARG": [
                {
                    "alowed_tokens": ":",
                    "next_state":
                    "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS_VALUE"
                }
            ],
            "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS_VALUE": [
                {
                    "alowed_tokens": "<ANY_NUMBER>",
                    "next_state":
                    "fn_add_numbers_EXPECT_OPENING_BRACE_FOR_PARAMS_VALUE"
                },
                {
                    "alowed_tokens": ",",
                    "next_state": "fn_add_numbers_AFTER_a"
                }
            ],
            "fn_add_numbers_AFTER_a": [
                {
                    "alowed_tokens": "\"b\"",
                    "next_state": "fn_add_numbers_AFTER_a_KEY_ARG"
                }
            ],
            "fn_add_numbers_AFTER_a_KEY_ARG": [
                {
                    "alowed_tokens": ":",
                    "next_state": "fn_add_numbers_AFTER_a_VALUE"
                }
            ],
            "fn_add_numbers_AFTER_a_VALUE": [
                {
                    "alowed_tokens": "<ANY_NUMBER>",
                    "next_state": "fn_add_numbers_AFTER_a_VALUE"
                },
                {
                    "alowed_tokens": "}",
                    "next_state": "fn_add_numbers_PARAM_END_BRACE"
                }
            ],
            "fn_add_numbers_PARAM_END_BRACE": [
                {
                    "alowed_tokens": "}",
                    "next_state": "DONE"
                }
            ],
            # for fn_greet
            "fn_greet_FUN_NAME": [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_END_QOUTES_FOR_FUN_NUM"
                }
            ],
            "fn_greet_END_QOUTES_FOR_FUN_NUM": [
                {
                    "alowed_tokens": ",",
                    "next_state": "fn_greet_COMMA_BEFORE_PARAMS"
                }
            ],
            "fn_greet_COMMA_BEFORE_PARAMS": [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_QOUTES_BEFORE_PARAMS"
                }
            ],
            "fn_greet_QOUTES_BEFORE_PARAMS": [
                {
                    "alowed_tokens": "parameters",
                    "next_state": "fn_greet_PARAMS_KEY",
                }
            ],
            "fn_greet_PARAMS_KEY": [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_QUOTES_AFTER_PARAMS"
                }
            ],
            "fn_greet_QUOTES_AFTER_PARAMS": [
                {
                    "alowed_tokens": ":",
                    "next_state": "fn_greet_PARAM_KEY_COLON"
                }
            ],
            "fn_greet_PARAM_KEY_COLON": [
                {
                    "alowed_tokens": "{",
                    "next_state": "fn_greet_EXPECT_OPENING_BRACE_FOR_PARAMS"
                }
            ],
            "fn_greet_EXPECT_OPENING_BRACE_FOR_PARAMS":
            [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_QOUTES_BEFORE_NAME"
                }
            ],
            "fn_greet_QOUTES_BEFORE_NAME": [
                {
                    "alowed_tokens": "name",
                    "next_state": "fn_greet_NAME",
                }
            ],
            "fn_greet_NAME": [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_QOUTES_AFTER_NAME"
                }
            ],
            "fn_greet_QOUTES_AFTER_NAME": [
                {
                    "alowed_tokens":  ":",
                    "next_state":
                    "fn_greet_EXPECT_OPENING_BRACE_FOR_PARAMS_VALUE"
                }
            ],
            "fn_greet_EXPECT_OPENING_BRACE_FOR_PARAMS_VALUE": [
                {
                    "alowed_tokens": "<ANY_STRING>",
                    "next_state": "flag"
                },
            ],
            "flag": [
                {
                    "alowed_tokens": "\"",
                    "next_state": "fn_greet_QOUTES_AFTER_GENERATE"
                }
            ],
            "fn_greet_QOUTES_AFTER_GENERATE": [
                {
                    "alowed_tokens": "}",
                    "next_state": "fn_greet_PARAM_END_BRACE"
                }
            ],
            "fn_greet_PARAM_END_BRACE": [
                {
                    "alowed_tokens": "}",
                    "next_state": "DONE"
                }
            ]
        }

    def get_alowed(self, state):
        # arr = []
        for key, val in self.state.items():
            if key == state:
                if len(val) == 1:
                    for _, v in val[0].items():
                        return v
                else:
                    for i in range(len(val)):
                        for _, v in val[i].items():
                            return v
        return None

    def update_state(self, state, token):
        for key, val in self.state.items():
            if key == state:
                return val[0]["next_state"]
                # else:
                #     for i in range(len(val)):
                #         for _, v in val[i].items():
                #             print("ba9i mahandlithach")

    def decoder(self, text: str) -> str:
        state = "START"
        # Fix: strip batch dimension if encode returns [[...]]
        input_ids = self.model.encode(text)[0].tolist()
        result = ""
        while state != "DONE":
            # 1. Get model predictions
            logits = self.model.get_logits_from_input_ids(input_ids)
            logits = np.array(logits)

            if state == "EXPECT_COLON_AFTER_NAME":
                next_token_id = int(np.argmax(logits))

                token = self.model.decode([next_token_id])
                input_ids.append(next_token_id)
                result += token

                print(result)
                if "fn_add_numbers" in result:
                    state = "fn_add_numbers_FUN_NAME"
                elif "fn_greet" in result:
                    state = "fn_greet_FUN_NAME"
                else:
                    state = "EXPECT_COLON_AFTER_NAME"
                if state == "DONE":
                    break
            else:
                # 2. Mask logits to allowed tokens only
                # lmochkil hnaya
                alowed = self.get_alowed(state)
                if "<ANY_" in alowed:
                    print(alowed)
                    next_token_id = int(np.argmax(logits))

                    # 4. Decode and accumulate
                    token = self.model.decode([next_token_id])
                    if token == "\",":
                        state = "flag"
                        continue
                    input_ids.append(next_token_id)
                    result += token
                    print(state)
                    print(result)
                    continue
                allowed_as_id = self.vocab[alowed]
                masked = np.full(len(logits), -np.inf)
                masked[allowed_as_id] = logits[allowed_as_id]

                # 3. Pick next token (greedy)
                next_token_id = int(np.argmax(masked))

                # 4. Decode and accumulate
                token = self.model.decode([next_token_id])
                input_ids.append(next_token_id)
                result += token

                # 5. Advance FSM state
                state = self.update_state(state, token)
                print(result)
                if state == "DONE":
                    break

        return result
