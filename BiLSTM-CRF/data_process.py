import os
import json
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.files = config.files

    def data_process(self):
        for file_name in self.files:
            self.get_examples(file_name)

    def get_examples(self, mode):
        """
        将json文件每一行中的文本分离出来，存储为words列表
        标记文本对应的标签，存储为labels
        words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
        labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        if not self.data_dir.endswith('/clue/'):
            input_dir = self.data_dir + str(mode) + '.txt'
        else:
            input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        with open(input_dir, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            if self.data_dir.endswith('/clue/'):
                # 先读取到内存中，然后逐行处理
                for line in f.readlines():
                    # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                    json_line = json.loads(line.strip())

                    text = json_line['text']
                    words = list(text)
                    # 如果没有label，则返回None
                    label_entities = json_line.get('label', None)
                    labels = ['O'] * len(words)

                    if label_entities is not None:
                        for key, value in label_entities.items():
                            for sub_name, sub_index in value.items():
                                for start_index, end_index in sub_index:
                                    assert ''.join(words[start_index:end_index + 1]) == sub_name
                                    if start_index == end_index:
                                        labels[start_index] = 'S-' + key
                                    else:
                                        labels[start_index] = 'B-' + key
                                        labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                    word_list.append(words)
                    label_list.append(labels)
            else:
                sent_ = []
                tag_ = []
                for line in f.readlines():
                    if line != '\n':
                        word, label = line.rstrip().split(' ')
                        sent_.append(word)
                        tag_.append(label)
                    else:
                        if len(sent_) > 0 and len(tag_) > 0 and len(sent_) == len(tag_):
                            word_list.append(sent_)
                            label_list.append(tag_)
                        sent_, tag_ = [], []
                if len(sent_) > 0 and len(tag_) > 0 and len(sent_) == len(tag_):
                    word_list.append(sent_)
                    label_list.append(tag_)

            # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("-------- {} data process DONE!--------".format(mode))
