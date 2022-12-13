import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--set',     type=str,   required=True)
parser.add_argument('--dataset', type=str,   required=True)
args = parser.parse_args()

SET = args.set
DATASET = args.dataset

file1 = open(f'/home/lily/lyf6/graph4nlp/examples/pytorch/table_qa/output/{DATASET}_perturb_{SET}_changed_gt.txt', 'r')
line1 = list(file1.readlines())
line1 = list(map(lambda s: s.strip('\n'), line1))

file2 = open(f'/home/lily/lyf6/graph4nlp/examples/pytorch/table_qa/output/{DATASET}_perturb_{SET}_changed_pred.txt', 'r')
line2 = list(file2.readlines())
line2 = list(map(lambda s: s.strip('\n'), line2))

file3 = open(f'/home/lily/lyf6/graph4nlp/examples/pytorch/table_qa/output/{DATASET}_perturb_{SET}_orig_gt.txt', 'r')
line3 = list(file3.readlines())
line3 = list(map(lambda s: s.strip('\n'), line3))

file4 = open(f'/home/lily/lyf6/graph4nlp/examples/pytorch/table_qa/output/{DATASET}_perturb_{SET}_orig_pred.txt', 'r')
line4 = list(file4.readlines())
line4 = list(map(lambda s: s.strip('\n'), line4))

idx_cap = min([len(i) for i in [line1,line2,line3,line4]])
line1, line2, line3, line4 = line1[:idx_cap], line2[:idx_cap], line3[:idx_cap], line4[:idx_cap]

result_df = pd.DataFrame({'ans_orig': line3, 'ans_pert': line1, 'pred_orig': line4, 'pred_pert': line2})
result_df['ans_orig'] = result_df['ans_orig'].apply(lambda x: x.strip('[]').strip("'"))
acc_orig = np.mean(np.array([a.lower().strip()==p.lower().strip() for (a,p) in zip(result_df['ans_orig'], result_df['pred_orig'])]))
acc_pert = np.mean(np.array([a.lower().strip()==p.lower().strip() for (a,p) in zip(result_df['ans_pert'], result_df['pred_pert'])]))
per_chng = np.mean(np.array([o.lower().strip()==p.lower().strip() for (o,p) in zip(result_df['pred_orig'], result_df['pred_pert'])]))
print(f"Acc (Orig): {acc_orig}, Acc (Pert): {acc_pert}, % Change: {per_chng}")
