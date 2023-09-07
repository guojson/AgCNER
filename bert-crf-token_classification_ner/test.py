# -*- coding: utf-8 -*-
import pandas as pd
from sys import platform
from torch.utils.data import DataLoader

import conlleval
from model import BertModel
from utils import *
from dataset import DataPrecessForSentence
from config import *

bert_path_or_name = model_dict[MODEL][-1]


def main():
    device = torch.device("cuda:1")
    Tokenizer = eval_object(model_dict[MODEL][0])
    tokenizer = Tokenizer.from_pretrained(bert_path_or_name)
    # tokenizer = tokenizer.from_pretrained(bert_path_or_name)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    print(target_file)
    if platform == "linux" or platform == "linux2":

        checkpoint = torch.load(target_file)
    else:
        checkpoint = torch.load(target_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print(f"\t* Loading test data {test_file}...")

    test_dataset = DataPrecessForSentence(tokenizer, test_file)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    # model = BertModelTest().to(device)
    model = BertModel().to(device)
    model.load_state_dict(checkpoint["model"])

    # test_loader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=10 )
    print(20 * "=", " Testing model on device: {} ".format(device), 20 * "=")
    # batch_time, total_time, accuracy, all_labels, all_pred = test(model, test_loader)
    batch_time, total_time, accuracy, all_labels, all_pred = test_data(model, test_loader, tokenizer)
    decoded_preds = ner_decode(all_pred)
    all_pred = [[j.item() for j in i.cpu()] for i in all_pred]
    all_labels = [[j.item() for j in i.cpu()] for i in all_labels]
    print(
        "\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%\n".format(batch_time,
                                                                                                            total_time,
                                                                                                            (
                                                                                                                    accuracy * 100)))
    df = pd.read_csv(test_file, engine='python', encoding=csv_encoding, error_bad_lines=False)

    sentences = df[csv_rows[0]].tolist()
    sentences = [list(sent_) for sent_ in sentences]

    df['pred'] = all_pred
    df['pred_decode'] = decoded_preds

    #
    label2id = load_json(json_dict)
    id2label = {v: k for k, v in label2id.items()}

    if all_labels:
        # decode_labels = [[id2label[j].replace('M', 'I').replace('E', 'I') for j in i] for i in all_labels]
        # decode_preds = [[id2label[j].replace('M', 'I').replace('E', 'I') for j in i] for i in all_pred]
        decode_labels = [[id2label[j] for j in i] for i in all_labels]
        decode_preds = [[id2label[j] for j in i] for i in all_pred]
        eval_list = []
        for ori_tokens, oril, prel in zip(sentences, decode_labels, decode_preds):
            for ot, ol, pl in zip(ori_tokens, oril, prel):
                eval_list.append(f"{ot} {ol} {pl}\n")
            eval_list.append("\n")

        # eval the model
        counts = conlleval.evaluate(eval_list)
        conlleval.report(counts, logger=logging)

        dict_all, dict_every_type = my_metrics(decode_labels, decode_preds)

        print(dict_all)
        print(dict_every_type)

    df['label_entities'] = [get_entities(i, id2label, 'bio') for i in all_pred]
    df.to_csv(test_pred_out, index=False, encoding='utf-8')


# def compute_metrics():

if __name__ == "__main__":
    main()
