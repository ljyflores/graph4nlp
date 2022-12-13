from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import random
import numpy as np

KEY = 'WikiSQL'
DATASET = 'dev'
MODEL = 'wtq' if KEY=='WikiTableQuestions' else 'wikisql'

device = torch.device("cuda:7")
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/tapex-large-finetuned-{MODEL}")
model = AutoModelForSeq2SeqLM.from_pretrained(f"microsoft/tapex-large-finetuned-{MODEL}").to(device)
df_main = pd.read_csv(f'{KEY}/seq2seq_data/{DATASET}_perturb.csv')

def to_device(item):
    for key in item:
        item[key] = item[key].to(device)
    return item

result_dict = {'ans_orig':[], 'ans_pert':[], 'pred_orig':[], 'pred_pert':[]}
for i in random.sample(list(range(df_main.shape[0])), 1000):
    row = df_main.iloc[i]
    try:
        if KEY=='WikiTableQuestions':
            pd_orig = pd.read_csv(f"WikiTableQuestions/{row['context']}")
            pd_pert = pd.read_csv(f"WikiTableQuestions/csv_perturbed/{DATASET}/{row['id']}.csv")
        elif KEY=='WikiSQL':
            pd_orig = pd.read_csv(f"WikiSQL/csv/{DATASET}/{row['table_id']}.csv")
            pd_pert = pd.read_csv(f"WikiSQL/csv_perturbed/{DATASET}/{row['table_id']}.csv")
        else:
            assert False
    except:
        tab_col = 'context' if KEY=='WikiTableQuestions' else 'table_id'
        id_col = 'id' if KEY=='WikiTableQuestions' else 'table_id'
        print(f"Could not find {row[tab_col]}, {row[id_col]}")    
        continue
    
    pd_orig = pd_orig.fillna("").applymap(str)
    pd_pert = pd_pert.fillna("").applymap(str)
    
    ans_col = 'targetValue' if KEY=='WikiTableQuestions' else 'answer'
    ans_new_col = 'targetValue_new'
    qst_col = 'utterance' if KEY=='WikiTableQuestions' else 'question'
    
    ans_orig = row[ans_col] 
    ans_pert = row[ans_new_col]
    question = row[qst_col]
    
    encoding_orig = tokenizer(pd_orig, question, return_tensors="pt", max_length=1024)
    encoding_pert = tokenizer(pd_pert, question, return_tensors="pt", max_length=1024)

    encoding_orig = to_device(encoding_orig)
    encoding_pert = to_device(encoding_pert)
    
    # let the model generate an answer autoregressively
    outputs_orig = model.generate(**encoding_orig, max_new_tokens=1024)
    outputs_pert = model.generate(**encoding_pert, max_new_tokens=1024)

    # decode back to text
    predicted_orig = tokenizer.batch_decode(outputs_orig, skip_special_tokens=True)[0]
    predicted_pert = tokenizer.batch_decode(outputs_pert, skip_special_tokens=True)[0]
    
    result_dict['ans_orig'].append(ans_orig)
    result_dict['ans_pert'].append(ans_pert)
    result_dict['pred_orig'].append(predicted_orig)
    result_dict['pred_pert'].append(predicted_pert)

result_df = pd.DataFrame(result_dict)
result_df['ans_orig'] = result_df['ans_orig'].apply(lambda x: x.strip('[]').strip("'"))
acc_orig = np.mean(np.array([a.lower().strip()==p.lower().strip() for (a,p) in zip(result_df['ans_orig'], result_df['pred_orig'])]))
acc_pert = np.mean(np.array([a.lower().strip()==p.lower().strip() for (a,p) in zip(result_df['ans_pert'], result_df['pred_pert'])]))
per_chng = np.mean(np.array([o.lower().strip()==p.lower().strip() for (o,p) in zip(result_df['pred_orig'], result_df['pred_pert'])]))
print(f"Acc (Orig): {acc_orig}, Acc (Pert): {acc_pert}, % Change: {per_chng}")

### WTQ Tapex (Dev)
# Acc (Orig): 0.851, Acc (Pert) 0.465, % No Change: 0.455

# WikiSQL Tapex (Dev)
# Acc (Orig): 0.768, Acc (Pert): 0.225, % No Change: 0.7
