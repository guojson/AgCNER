from data import build_corpus
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, \
    bilstm_train_and_eval, ensemble_evaluate


def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir='data/agcner/1-8')
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir='data/agcner/1-8')
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir='data/agcner/1-8')

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )
    # print(hmm_pred)
    sent = []
    y_true = []
    y_hmm_pred = []
    for i in range(len(test_word_lists)):
        sent.extend(test_word_lists[i])
        y_true.extend(test_tag_lists[i])
        y_hmm_pred.extend(hmm_pred[i])

    with open('data/agcner/1-8/test_hmm_results.txt', 'w', encoding='utf-8', ) as f:
       for i in range(len(sent)):

           f.write(sent[i]+' '+y_true[i]+' '+y_hmm_pred[i]+'\n')


    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )
    y_crf_pred = []
    for i in range(len(test_word_lists)):
        y_crf_pred.extend(crf_pred[i])

    with open('data/agcner/1-8/test_crf_results.txt', 'w', encoding='utf-8', ) as f:
        for i in range(len(sent)):
            f.write(sent[i] + ' ' + y_true[i] + ' ' + y_crf_pred[i] + '\n')

    ensemble_evaluate(
        [hmm_pred,crf_pred],
        test_tag_lists
    )


if __name__ == "__main__":
    main()
