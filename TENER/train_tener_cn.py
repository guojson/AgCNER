from multiprocessing import freeze_support

from models.TENER import TENER
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.embeddings import StaticEmbedding
from modules.pipe import  CNNERPipe

import argparse
from modules.callbacks import EvaluateCallback

device = 0
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='agcner', choices=['weibo', 'resume', 'ontonotes', 'msra','agcner'])

args = parser.parse_args()

dataset = args.dataset
if dataset == 'resume':
    n_heads = 4
    head_dims = 64
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 50
elif dataset == 'weibo':
    n_heads = 4
    head_dims = 32
    num_layers = 1
    lr = 0.001
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'ontonotes':
    n_heads = 4
    head_dims = 48
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'msra':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100
elif dataset == 'agcner':
    n_heads = 6
    head_dims = 80
    num_layers = 2
    lr = 0.0007
    attn_type = 'adatrans'
    n_epochs = 100

pos_embed = None

batch_size = 32
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True

dropout=0.15
fc_dropout=0.4

encoding_type = 'bio'
name = 'caches/{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)


@cache_results(name, _refresh=False)
def load_data():
    # 替换路径
    if dataset == 'ontonotes':
        paths = {'train':'../data/OntoNote4NER/train.char.bmes',
                 "dev":'../data/OntoNote4NER/dev.char.bmes',
                 "test":'../data/OntoNote4NER/test.char.bmes'}
        min_freq = 2
    elif dataset == 'weibo':
        paths = {'train': '../data/WeiboNER/train.all.bmes',
                 'dev':'../data/WeiboNER/dev.all.bmes',
                 'test':'../data/WeiboNER/test.all.bmes'}
        min_freq = 1
    elif dataset == 'resume':
        paths = {'train': 'data/ResumeNER/train.char.bmes',
                 'dev':'data/ResumeNER/dev.char.bmes',
                 'test':'data/ResumeNER/test.char.bmes'}
        min_freq = 1
    elif dataset == 'msra':
        paths = {'train': '../data/MSRANER/train_dev.char.bmes',
                 'dev':'../data/MSRANER/test.char.bmes',
                 'test':'../data/MSRANER/test.char.bmes'}
    elif dataset == 'agcner':
        paths = {'train': 'data/agcner/train.char.bmes',
                 'dev': 'data/agcner/dev.char.bmes',
                 'test': 'data/agcner/test.char.bmes'}
        min_freq = 2
    data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)
    embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name=r'E:/guo/NER/Flat-Lattice-Transformer-master/gigaword_chn.all.a2b.uni.ite50.vec',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    bi_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'),
                               model_dir_or_name=r'E:/guo/NER/Flat-Lattice-Transformer-master/gigaword_chn.all.a2b.bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=min_freq,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    return data_bundle, embed, bi_embed

def main():

    data_bundle, embed, bi_embed = load_data()
    print(data_bundle)

    model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                           d_model=d_model, n_head=n_heads,
                           feedforward_dim=dim_feedforward, dropout=dropout,
                            after_norm=after_norm, attn_type=attn_type,
                           bi_embed=bi_embed,
                            fc_dropout=fc_dropout,
                           pos_embed=pos_embed,
                            scale=attn_type=='transformer')

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    callbacks = []
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
    evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

    if warmup_steps>0:
        warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
        callbacks.append(warmup_callback)
    callbacks.extend([clip_callback, evaluate_callback])

    trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                      num_workers=0, n_epochs=n_epochs, dev_data=data_bundle.get_dataset('dev'),
                      metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                      dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                      use_tqdm=True, print_every=300, save_path=None)
    trainer.train(load_best_model=False)

if __name__ == '__main__':
    main()