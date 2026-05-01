import json

data = [
    {
        'prompt': "Format template: Say \"hello\" to",
        'name': 'fn_format_template',
        'parameters': {'template': "Say 'hello' to'"}
    }
]

with open("result.json", "a") as f:
    print(data)
    json.dump(data, f, indent=4)
