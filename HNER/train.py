import pickle

from sklearn.metrics import confusion_matrix
import numpy as np
import conlleval
from seqlab.train_model import create_config, train_model

if __name__ == '__main__':
    # name = "agcner_ner"
    # dirpath= "."
    #
    # data_path = "data/agcner/agcner.pk"
    #
    # # create config
    # config = create_config(name, dirpath=dirpath, data_path=data_path, max_epoch=30,
    #                        model_name='ckiplab/bert-base-chinese')
    #
    # # train model
    # train_model(config)
    from seqlab.inference import load_model

    checkpoint = "agcner_ner-v2.ckpt"

    # load model
    model = load_model(checkpoint)

    with open(r'data/agcner/agcner.pk', 'rb') as f:
        data = pickle.load(f)

    sent_list = []
    for sent in data['test']:
        sent_list.append(sent['tokens'])

    gold_List = []
    for sent in data['test']:
        gold_List.append(sent['tags'])
    # # prediction
    # tokens = list("小麦的小麦赤霉病是如何防治更好")
    #
    # # prediction = model.extract_entities([tokens])
    prediction = model.extract_predictions(sent_list)
    #
    print(prediction)
    sents = []
    y_true = []
    y_pred = []
    for i in range(len(gold_List)):
        sents.extend(sent_list[i])
        y_true.extend(gold_List[i])
        y_pred.extend(prediction[i])
    labels = ['I-PART', 'B-BEL', 'I-STRAINS', 'B-DRUG', 'I-ORG', 'B-STRAINS', 'I-PET', 'B-ORG', 'B-COM', 'B-FER',
              'I-CLA',
              'I-PER', 'I-DRUG', 'B-DIS', 'I-BEL', 'B-PART', 'O', 'B-PER', 'I-REA', 'I-DIS', 'B-PET', 'B-CLA', 'I-FER',
              'B-CRO', 'I-CRO', 'B-REA', 'I-COM']
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    np.savetxt('data/agcner/test_result_matrix.csv', cm, delimiter=',')

    with open('data/agcner/predict_results.txt', 'w', encoding='utf-8', ) as f:
        for i in range(len(sents)):
            f.write(sents[i] + ' ' + y_true[i] + ' ' + y_pred[i] + '\n')

    eval_list = []
    for ori_tokens, oril, prel in zip(sent_list, gold_List, prediction):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)
