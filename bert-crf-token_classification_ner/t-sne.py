import codecs
import json
import os
import shutil
import tempfile

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

import pandas as pd


# digits = datasets.load_digits(n_class=6)
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape
#
# '''显示原始数据'''
# n = 20  # 每行20个数字，每列20个数字
# img = np.zeros((10 * n, 10 * n))
# for i in range(n):
#     ix = 10 * i + 1
#     for j in range(n):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# vectors_AgCNER = np.load('emb_origanal_BERT/vectors_AgCNER.npy')
# vectors_ccks2017 = np.load('emb_origanal_BERT/vectors_ccks2017.npy')
# vectors_msra = np.load('emb_origanal_BERT/vectors_clue.npy')
# vectors_resume = np.load('emb_origanal_BERT/vectors_resume.npy')

# vectors_AgCNER = np.load('emb_word2vec/vectors_word2vec_average_AgCNER.npy')
# vectors_ccks2017 = np.load('emb_word2vec/vectors_word2vec_average_ccks2017.npy')
# vectors_msra = np.load('emb_word2vec/vectors_word2vec_average_clue.npy')
# vectors_resume = np.load('emb_word2vec/vectors_word2vec_average_resume.npy')

def sentence_level_vision():
    flag = '{}-obert-hn.npy'

    vectors_AgCNER = np.load(flag.format('agcner'))
    # vectors_ccks2017 = np.load('emb_fineturning_BERT/vectors_ccks2017.npy')
    # vectors_msra = np.load('emb_fineturning_BERT/vectors_clue.npy')
    vectors_resume = np.load(flag.format('resume'))

    X = []
    X.extend(vectors_AgCNER)
    X.extend(vectors_resume)

    X = np.array(X)

    y = []

    y.extend([0 for i in range(50)])
    y.extend([1 for i in range(50)])
    # y.extend([2 for i in range(100)])
    # y.extend([3 for i in  range(100)])

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, early_exaggeration=5)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    np.savetxt(flag + '.csv', X_norm, delimiter=',')
    # # plt.figure(figsize=(8, 8))
    # #
    # #
    # #
    # # for i in range(X_norm.shape[0]):
    # #     print(X_norm[i, 0], X_norm[i, 1],X_norm[i,2],str(y[i]))
    # #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
    # #              fontdict={'weight': 'bold', 'size': 9})
    # # plt.xticks([])
    # # plt.yticks([])
    # # plt.show()
    #
    # # 数据１
    data1 = X_norm[0:50]
    # data的值如下：
    # [[ 0  1  2]
    #  [ 3  4  5]
    #  [ 6  7  8]
    #  [ 9 10 11]
    #  [12 13 14]
    #  [15 16 17]
    #  [18 19 20]
    #  [21 22 23]]
    x1 = data1[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y1 = data1[:, 1]  # [ 1  4  7 10 13 16 19 22]
    # z1 = data1[:, 2]  # [ 2  5  8 11 14 17 20 23]
    #
    # 数据２
    data2 = X_norm[50:100]
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    # z2 = data2[:, 2]

    plt.figure(figsize=(20, 20), dpi=100)

    plt.scatter(x1, y1, s=100, c='deeppink', marker='o', label="AgCNER")
    plt.scatter(x2, y2, s=100, c='darkblue', marker='+', label="Resume")
    # 2.5 添加图例
    plt.legend(loc="best")
    # 2.4 图像保存
    plt.savefig("./f-r-a.png")
    # 3.图像显示
    plt.show()
    # # # 数据3
    # # data3 = X_norm[200:300]
    # # x3 = data3[:, 0]
    # # y3 = data3[:, 1]
    # # z3 = data3[:, 2]
    # #
    # # # 数据3
    # # data4 = X_norm[300:400]
    # # x4 = data4[:, 0]
    # # y4 = data4[:, 1]
    # # z4 = data4[:, 2]
    #
    # # 绘制散点图
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x1, y1, z1, c='#30A9DE', label='AgCNER')
    # ax.scatter(x2, y2, z2, c='#E53A40', label='resume')
    # # ax.scatter(x3, y3, z3, c='#285943', label='clue')
    # # ax.scatter(x4, y4, z4, c='#9055A2', label='resume')
    #
    # # 绘制图例
    # ax.legend(loc='best')
    #
    # # 添加坐标轴(顺序是Z, Y, X)
    # # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # # ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    # dirpath = tempfile.mkdtemp()
    # images = []
    # for angle in range(0, 360, 5):
    #     ax.view_init(30, angle)
    #     fname = os.path.join(flag + str(angle) + '_100.png')
    #     plt.savefig(fname, dpi=100, format='png', bbox_inches='tight')
    #     images.append(imageio.imread(fname))
    # imageio.mimsave(flag + '_100.gif', images)
    # shutil.rmtree(dirpath)
    #
    # # 展示
    # plt.show()


def word_level_vision():
    entity_vectors_AgCNER = json.load(open('embedding/or_char_entity_vectors_AgCNER.json', encoding='utf-8'))
    entity_vectors_resume = json.load(open('embedding/or_char_entity_vectors_resume.json', encoding='utf-8'))

    entity_AgCNER_list = {}
    for index, entity in enumerate(entity_vectors_AgCNER):
        if entity['entity'] not in entity_AgCNER_list.keys():
            entity_AgCNER_list[entity['entity']] = entity['rep']

    entity_resume_list = {}
    for index, entity in enumerate(entity_vectors_resume):
        if entity['entity'] not in entity_resume_list.keys():
            entity_resume_list[entity['entity']] = entity['rep']

    with codecs.open('embedding/or_char_entity_vectors_map_AgCNER.json', 'w', encoding='utf-8') as file_obj:
        json.dump(entity_AgCNER_list, file_obj, indent=4, ensure_ascii=False)

    X = []

    X.extend(np.array([entity_AgCNER_list[i] for i in list(entity_AgCNER_list.keys())[:100]]))
    X.extend(np.array([entity_resume_list[i] for i in list(entity_resume_list.keys())[:100]]))
    X = np.array(X)
    y = []

    y.extend([0 for i in range(100)])
    y.extend([1 for i in range(100)])

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, early_exaggeration=5)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    np.savetxt('embedding/or_char_entity_vectors.csv', X_norm, delimiter=',')

    # 数据１
    data1 = X_norm[0:100]

    x1 = data1[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y1 = data1[:, 1]  # [ 1  4  7 10 13 16 19 22]

    # 数据２
    data2 = X_norm[100:200]
    x2 = data2[:, 0]
    y2 = data2[:, 1]

    plt.scatter(x1, y1, s=200, label='AgCNER', c='blue', marker='.', alpha=None, edgecolors='white')
    # 显示图例
    plt.legend()

    plt.scatter(x2, y2, s=200, label='Resume', c='red', marker='.', alpha=None, edgecolors='white')
    plt.legend()  # 每次都要执行

    plt.show()


def AgCNER_word_vision():

    # 可视化单个实体
    or_char_entity_vectors_map_AgCNER_ = json.load(
        open('embedding/or_char_entity_vectors_map_AgCNER.json', encoding='utf-8'))

    X = []

    for info in or_char_entity_vectors_map_AgCNER_.values():
        X.append(info)
    Y = []
    for info in or_char_entity_vectors_map_AgCNER_.keys():
        Y.append(info)

    name = ['one']
    test = pd.DataFrame(columns=name, data=Y)  # 数据有三列，列名分别为one,two,three
    test.to_csv('embedding/or_char_entity_AgCNE.csv', encoding='utf-8')

    entity_name = list(or_char_entity_vectors_map_AgCNER_.keys())
    # for info in ft_char_entity_vectors_map_AgCNER_:
    #     X.append(info['emb'])

    X = np.array(X)
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsne.fit_transform(X)

    X_tsne = tsne.embedding_

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    np.savetxt('embedding/or_char_entity_map_vectors.csv', X_norm, delimiter=',')

    # 数据１
    data1 = X_norm[0:100]

    x1 = data1[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y1 = data1[:, 1]  # [ 1  4  7 10 13 16 19 22]

    plt.scatter(x1, y1, s=200, label='AgCNER', c='blue', marker='.', alpha=None, edgecolors='white')

    # for i in range(len(x1)):
    #     plt.annotate(entity_name[i], xy=(x1[i], y1[i]), xytext=(x1[i] + 0.1, y1[i] + 0.1))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标

    # 显示图例
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # word_level_vision()
    # AgCNER_word_vision()
    sentence_level_vision()

