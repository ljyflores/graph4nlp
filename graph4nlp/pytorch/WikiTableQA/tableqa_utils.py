import pandas as pd
from pandas.errors import ParserError
import json
WTQ_DIR = '../../../graph4nlp/pytorch/WikiTableQA/WikiTableQuestions'
WSQ_DIR = '../../../graph4nlp/pytorch/WikiTableQA/WikiSQL'

# def get_item_table(row):
    # try:
    #     pd_table = pd.read_csv(f"{WTQ_DIR}/{row['context'][:-4]}.tsv", 
    #                        delimiter='\t')
    # except:
    # pd_table = pd.read_csv(f"{WTQ_DIR}/{row['context']}")
    # # Get question and answer
    # question = row['utterance']
    # answer   = row['targetValue']
    # return pd_table, question, answer

def get_sql_table(row, DATASET):
    DATASET_ = 'train' if 'train' in DATASET else \
               'test' if 'test' in DATASET else \
               'dev' 
    FOLDER   = 'csv_perturbed' if (('perturb' in DATASET) and ('changed' in DATASET)) else 'csv'
    ans_col  = 'targetValue_new' if (('perturb' in DATASET) and ('changed' in DATASET)) else 'answer'
    pd_table = pd.read_csv(f"{WSQ_DIR}/{FOLDER}/{DATASET_}/{row['table_id']}.csv")
    question = row['question']
    answer   = row[ans_col]
    return pd_table, question, answer

def get_item_sql(item):
    # Load table, question, and answer
    folder, table_idx = item['tbl'].split('_')
    try:
        pd_table = pd.read_csv(f"{WTQ_DIR}/csv/{folder}-csv/{table_idx}.tsv", delimiter='\t') 
    except:
        pd_table = pd.read_csv(f"{WTQ_DIR}/csv/{folder}-csv/{table_idx}.csv")
    question          = ' '.join(map(lambda x: x[1], item['sql']))
    answer            = item['tgt']
    return pd_table, question, answer

def load_data(KEY, DATASET):
    # Declare output lists
    question_lst, answer_lst, table_lst = [], [], []        
    
    # Load data source
    if 'both' in KEY:
        if 'WTQ' in KEY:
            if 'perturb' in DATASET:
                if 'test' in DATASET:
                    print(f"Main File: {WTQ_DIR}/seq2seq_data/train_perturb.csv")
                    iter_item = pd.read_csv(f"{WTQ_DIR}/seq2seq_data/train_perturb.csv")
                elif 'dev' in DATASET:
                    print(f"Main File: {WTQ_DIR}/seq2seq_data/dev_perturb.csv")
                    iter_item = pd.read_csv(f"{WTQ_DIR}/seq2seq_data/dev_perturb.csv")                    
                else:
                    assert False
            elif DATASET=='train':
                iter_item = pd.read_csv(f'{WTQ_DIR}/data/training.tsv', delimiter='\t')
            elif DATASET=='test':
                iter_item = pd.read_csv(f'{WTQ_DIR}/data/pristine-unseen-tables.tsv', delimiter='\t')
            elif 'dev' in DATASET:
                iter_item = pd.read_csv(f'{WTQ_DIR}/data/random-split-{DATASET[-1]}-dev.tsv', delimiter='\t')
            else:
                print("Dataset should be in ['train','test','dev']")
                assert False
            print(f"Loaded WikitableQA, {DATASET}")
        elif 'WikiSQL' in KEY:
            if 'perturb' in DATASET:
                if 'test' in DATASET:
                    print(f"Main File: {WSQ_DIR}/seq2seq_data/test_perturb.csv")
                    iter_item = pd.read_csv(f"{WSQ_DIR}/seq2seq_data/test_perturb.csv")
                elif 'dev' in DATASET:
                    print(f"Main File: {WSQ_DIR}/seq2seq_data/dev_perturb.csv")
                    iter_item = pd.read_csv(f"{WSQ_DIR}/seq2seq_data/dev_perturb.csv")                    
                else:
                    assert False
            elif DATASET=='train':
                iter_item = pd.read_csv(f'{WSQ_DIR}/annotated/train.csv')
            elif DATASET=='test':
                iter_item = pd.read_csv(f'{WSQ_DIR}/annotated/test.csv')
            elif 'dev' in DATASET:
                iter_item = pd.read_csv(f'{WSQ_DIR}/annotated/dev.csv')
            else:
                print("Dataset should be in ['train','test','dev']")
                assert False
        else:
            print('KEY should be either bothWikiSQL or bothWTQ')
            assert False
    else:
        f = open(f'../../../graph4nlp/pytorch/WikiTableQA/data.json')
        iter_item = json.load(f)
        print("Loaded SQL")

    
    for idx in range(len(iter_item)):
        # Load table
        if 'both' in KEY:
            if 'WTQ' in KEY:
                if 'perturb' in DATASET:
                    question_col = 'utterance'
                    answer_col   = 'targetValue' if 'orig' in DATASET else 'targetValue_new'
                    DATASET_ABBR = 'dev' if 'dev' in DATASET else 'train'
                    pathname     = iter_item.iloc[idx]['context'] if 'orig' in DATASET else\
                        f"csv_perturbed/{DATASET_ABBR}/{iter_item.iloc[idx]['id']}.csv"
                else:
                    question_col = 'utterance'
                    answer_col   = 'targetValue'
                    pathname     = iter_item.iloc[idx]['context']
                question = iter_item.iloc[idx][question_col]
                answer   = iter_item.iloc[idx][answer_col]
                try:
                    pd_table = pd.read_csv(f"{WTQ_DIR}/{pathname}")
                except:
                    try:
                        pd_table = pd.read_csv(f"{WTQ_DIR}/{pathname[:-4]}.tsv", delimiter='\t')
                    except:
                        try:
                            pd_table = pd.read_csv(f"{WTQ_DIR}/{pathname}", quotechar="'")
                        except:
                            assert False
            elif 'WikiSQL' in KEY:
                pd_table, question, answer = get_sql_table(iter_item.iloc[idx], DATASET)
            else:
                assert False
        else:
            pd_table, question, answer = get_item_sql(iter_item[idx])
        
        question_lst.append(question)
        answer_lst.append(answer)
        table_lst.append(pd_table)
  
    return question_lst, answer_lst, table_lst
