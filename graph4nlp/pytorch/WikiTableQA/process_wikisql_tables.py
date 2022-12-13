import json
import pandas as pd
from tqdm import tqdm

with open(f'WikiSQL/data/train.tables.jsonl', 'r') as json_file:
    json_list = list(json_file)

for item in tqdm(json_list, total=len(json_list)):
    d = json.loads(item)
    pd.DataFrame(d['rows'], columns=d['header']).to_csv(f"WikiSQL/csv/train/{d['id']}.csv", index=False)
    
with open(f'WikiSQL/data/dev.tables.jsonl', 'r') as json_file:
    json_list = list(json_file)

for item in tqdm(json_list, total=len(json_list)):
    d = json.loads(item)
    pd.DataFrame(d['rows'], columns=d['header']).to_csv(f"WikiSQL/csv/dev/{d['id']}.csv", index=False)
