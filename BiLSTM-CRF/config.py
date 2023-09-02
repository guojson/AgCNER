import datetime
import os
import time

data_dir = os.getcwd() + '/data/agcner/'
train_dir = data_dir + 'train.npz'
dev_dir = data_dir + 'dev.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'dev', 'test']
vocab_path = data_dir + 'vocab.npz'
exp_dir = os.getcwd() + '/experiments/agcner/'
model_dir = exp_dir + 'bilstm-crf.pth'
log_dir = exp_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
case_dir = os.getcwd() + '/case/agcner/bilstm_crf.txt'

max_vocab_size = 1000000

n_split = 5
dev_split_size = 0.1
batch_size = 32
embedding_size = 128
hidden_size = 384
drop_out = 0.6
lr = 0.001
betas = (0.9, 0.999)
lr_step = 5
lr_gamma = 0.8

epoch_num = 30
min_epoch_num = 5
patience = 0.0002
patience_num = 5

gpu = '0'

# --------------------------IDCNN---------------------------------------------------#
kernel_size = 3
dilation_l = None
dilation = 1
blocks = 1
drop_penalty = 1e-4
input_dropout = 0.5
middle_dropout = 0.2
hidden_dropout = 0.2
filters = 128
labels = ['CRO', 'DIS', 'PET', 'CUL', 'COM', 'DRUG', 'PER', 'PART', 'ORG', 'PAOG', 'BIS', 'FER', 'OTH']
label2id = {
    "O": 0,
    "B-CRO": 1,
    "I-CRO": 2,
    "B-DIS": 3,
    "I-DIS": 4,
    "B-PET": 5,
    "I-PET": 6,
    "B-CUL": 7,
    "I-CUL": 8,
    "B-COM": 9,
    "I-COM": 10,
    "B-DRUG": 11,
    "I-DRUG": 12,
    "B-PER": 13,
    "I-PER": 14,
    "B-OTH": 15,
    "I-OTH": 16,
    "B-PART": 17,
    "I-PART": 18,
    "B-ORG": 19,
    "I-ORG": 20,
    "B-PAOG": 21,
    "I-PAOG": 22,
    "B-BIS": 23,
    "I-BIS": 24,
    "B-FER": 25,
    "I-FER": 26
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
