import pandas as pd

data_dict = {'input_text':[],'target_text':[]}
data = pd.read_json('intents.json').to_dict(orient='records')

for d in data:
    input = d['intents']['patterns']
    target = d['intents']['responses']
    for i in input:
        for t in target:
            data_dict['input_text'].append(i)
            data_dict['target_text'].append(t)

df = pd.DataFrame(data_dict)
# df = df.sample(frac=1, random_state=42)
df.to_csv('chat.csv')
print(df)