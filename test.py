import json


arr = {"Ga": 15}
print(arr)
arr = {value: key.replace("G", " ") for key, value in arr.items()}


print(arr)
