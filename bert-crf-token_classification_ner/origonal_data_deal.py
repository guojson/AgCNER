import torch

from utils import eval_object
from config import *
import pandas as pd
import numpy as np

bert_path = r'E:\guo\NER\BERT-BiLSTM-CRF-NER-pytorch-master\bert-base-chinese'

Tokenizer = eval_object(model_dict[MODEL][0])

tokenizer = Tokenizer.from_pretrained(bert_path)
ClassifyClass = eval_object(model_dict[MODEL][1])
bert = ClassifyClass.from_pretrained(bert_path)

# input_ids = tokenizer.convert_tokens_to_ids(list('我是中国人'))

# print(input_ids)


df = pd.read_csv('data/resume/test-50.csv', engine='python', encoding=csv_encoding, error_bad_lines=False, nrows=n_nums)

sentences = df[csv_rows[0]].tolist()

print(sentences)
input_ids = [tokenizer.convert_tokens_to_ids(list(sen)) for sen in sentences]

print(input_ids)

results = torch.tensor([])
for ids in input_ids:
    bert_output = bert(torch.Tensor([ids]).type(torch.long))  # [bs, seq_len, num_labels]
    sequence_output = bert_output[0]
    sent_level = torch.mean(sequence_output, dim=1)
    results = torch.cat((results, sent_level), 0)
np.save('resume-obert-hn.npy', results.detach().numpy())
