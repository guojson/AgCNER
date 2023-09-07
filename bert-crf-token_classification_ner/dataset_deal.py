import ast

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from config import csv_encoding, n_nums, csv_rows
import numpy as np
# df = pd.read_csv('data/agcner/bert_test_data_predict.csv', engine='python', encoding=csv_encoding,
#                  error_bad_lines=False,
#                  nrows=n_nums)
# sentences = df['sen'].tolist()
# sentences_gold = df['label_decode'].tolist()
# sentences_pred = df['pred_decode'].tolist()
# # print(sentences_gold)
# # print(sentences_pred)
# sentences = [ast.literal_eval(sent) for sent in sentences]
# sent_gold = [ast.literal_eval(sent) for sent in sentences_gold]
# sent_pred = [sent.split(',') for sent in sentences_pred]



y_true = []
y_pred = []
# for i in range(len(sentences)):
#     y_true.extend(sent_gold[i])
#     y_pred.extend(sent_pred[i])

with open('data/agcner/test_results.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        y_true.append(line.strip().split(' ')[1])
        y_pred.append(line.strip().split(' ')[2])


labels = ['I-PART', 'B-BEL', 'I-STRAINS', 'B-DRUG', 'I-ORG', 'B-STRAINS', 'I-PET', 'B-ORG', 'B-COM', 'B-FER', 'I-CLA',
          'I-PER', 'I-DRUG', 'B-DIS', 'I-BEL', 'B-PART', 'O', 'B-PER', 'I-REA', 'I-DIS', 'B-PET', 'B-CLA', 'I-FER',
          'B-CRO', 'I-CRO', 'B-REA', 'I-COM']

cm = confusion_matrix(y_true, y_pred, labels=labels)

np.savetxt('data/agcner/test_result_matrix.csv',cm,delimiter=',')

print(cm)

sns.heatmap(cm,annot=True ,fmt="d",xticklabels=labels,yticklabels=labels)
plt.title('confusion matrix')  # 标题
plt.xlabel('Predict lable')  # x轴
plt.ylabel('True lable')  # y轴
plt.show()


# with open('data/agcner/test_results.txt', 'w', encoding='utf-8', ) as f:
#     for ori_tokens, oril, prel in zip(sentences, sent_gold, sent_pred):
#         for ot, ol, pl in zip(ori_tokens, oril, prel):
#             f.write(str(ot) + ' ' + str(ol) + ' ' + str(pl) + '\n')
