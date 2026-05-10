import json

func = {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}},
    "returns": {"type": "number"},
}

print(list(func["parameters"].items())[0][0])
key = list(func["parameters"].items())
print(key[0][0])

# # print(arr)

# key, _ = func["parameters"].items()
# print(f' "{key[0]}": ', end="", flush=True)
