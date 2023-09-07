import pickle


def convert_to_pk(dataset):
    with open(dataset + '.txt', 'r', encoding='utf8') as f:
        datas = f.read().rstrip()
    sentences = datas.split('\n\n')

    datas = []
    for r in sentences:
        lines = r.split('\n')
        data = {'tokens': [], 'tags': []}
        for index, line in enumerate(lines):
            text = line.strip().split(" ")
            word = text[0]
            tag = text[1]
            data['tokens'].append(word)
            data['tags'].append(tag)
        datas.append(data)
    return datas


if __name__ == '__main__':
    datas = {}

    datas['train'] = convert_to_pk('train')
    datas['dev'] = convert_to_pk('dev')
    datas['test'] = convert_to_pk('test')

    datas['tag_to_id'] = {'O': 0, 'B-CRO': 1, 'I-CRO': 2, 'B-DIS': 3, 'I-DIS': 4, 'B-PET': 5, 'I-PET': 6,
                          'B-STRAINS': 7, 'I-STRAINS': 8, 'B-COM': 9, 'I-COM': 10,
                          'B-DRUG': 11, 'I-DRUG': 12, 'B-PER': 13, 'I-PER': 14, 'B-CLA': 15, 'I-CLA': 16, 'B-PART': 17,
                          'I-PART': 18, 'B-ORG': 19, 'I-ORG': 20, 'B-REA': 21,
                          'I-REA': 22, 'B-BEL': 23, 'I-BEL': 24, 'B-FER': 25, 'I-FER': 26}

    with open('agcner.pk', 'wb') as f:
        pickle.dump(datas, f)

    with open('agcner.pk', 'rb') as f:
        dd = pickle.load(f)

    print(dd)
