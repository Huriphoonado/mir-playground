# import matplotlib.pyplot as plt
# import matplotlib.style as ms
import json

with open('predict.json', 'r') as file:
    predictions = json.load(file)

print(predictions[0]['t_id'])
print(type(predictions[0]['labels']), len(predictions[0]['labels']))
print(type(predictions[0]['guesses']), len(predictions[0]['guesses']))
print(type(predictions[0]['times']), len(predictions[0]['times']))
