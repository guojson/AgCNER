# AgCNER
****
A publicly available dataset and code for Chinese agricultural diseases and pests

# 1 Note 
During the under review period, the dataset has been uploaded to figshare and can be viewed by editors and reviewers. It
will be fully released here as soon as the paper was accepted
# 2 Dataset
The large-scale corpus for Chinese ADP-NER task named AgCNER contains 13 categories, 206,992 entities, and 66,553 instances with 3,909,293 characters. Compared with other datasets,
AgCNER maintains the best performance in terms of the number of categories, entities, samples, and characters. Moreover, this is the first publicly available corpus for this domain-specific field.
## 2.1 Entity Tags
![image](https://github.com/guojson/AgCNER/assets/44044833/c5f3e4cb-dc6e-472d-acb9-f4fd34febb39)
## 2.2 Examples
![image](https://github.com/guojson/AgCNER/assets/44044833/e5574498-c9cc-4b41-9026-f04b17c093cc)
## 2.3 Statistic
![image](https://github.com/guojson/AgCNER/assets/44044833/00f240c6-e8c1-47ce-bd32-373a463550e6)

## 2.4 Results
![image](https://github.com/guojson/AgCNER/assets/44044833/aaf5285a-c19a-4ace-8ea4-1dc8c758708f)

# 3 Code
The full code, including HMM, CRF, BiLSTM-CRF, IDCNN-CRF, BiLSTM-Attention-CRF, Lattice-LSTM, TENER, FLAT, NFLAT, Graph4CNER,
BERT-BiLSTM-CRF,BERT-IDCNN-CRF,HNER, with their outputs for AgCNER have been released.

##### 3.1 HMM and CRF
The code for HMM and CRF was listed in file "HMM and CRF"

Before training, you can change the data_dir into your own data path, then it would be worked by using following command:

```
python main.py

```
#### 3.2 BiLSTM-CRF, BiLSTM-Attention-CRF, and IDCNN-CRF
The code was released in file "BiLST-CRF"
Before training, specific model should be confirmed in run.py, then it would be worked by using following command:

```
python run.py
```

#### 3.3 Lattice-LSTM
Lattice LSTM for Chinese NER. Character based LSTM with Lattice embeddings as input.
****

The pretrained character and word embeddings are the same with the embeddings in the baseline of RichWordSegmentor

Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

Word(Lattice) embeddings (ctb.50d.vec): [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing)

the code can be run by using following command:
```
python main.py
```
#### 3.4 TENER

This project needs the natural language processing python package fastNLP. You can install by the following command:
```
pip install fastNLP
```
You can run the code by the following command
```
python train_tener_cn.py --dataset agcner
```
#### 3.5 FLAT
code for ACL 2020 paper: FLAT: Chinese NER Using Flat-Lattice Transformer. 

you can go [here](https://fastnlp.readthedocs.io/zh/latest/) to know more about FastNLP.

How to run the code?
====
1. Download the character embeddings and word embeddings.

      Character and Bigram embeddings (gigaword_chn.all.a2b.{'uni' or 'bi'}.ite50.vec) : [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

      Word(Lattice) embeddings: 
      
      yj, (ctb.50d.vec) : [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
      
      ls, (sgns.merge.word.bz2) : [Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)

2. Modify the `paths.py` to add the pretrained embedding and the dataset
3. Run following commands
```
python preprocess.py (add '--clip_msra' if you need to train FLAT on MSRA NER dataset)
cd V0 (without Bert) / V1 (with Bert)
python flat_main.py --dataset <dataset_name> (agcner)
```
#### 3.5 NFLAT
before training, fastNLP should be also installed using following command:
```
pip install fastNLP
```
1. Download the pretrained character embeddings and word embeddings and put them in the data folder.
    * Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Bi-gram embeddings (gigaword_chn.all.a2b.bi.ite50.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Word(Lattice) embeddings (ctb.50d.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * If you want to use a larger word embedding, you can refer to [Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors) and [Tencent AI Lab Embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html)

2. Modify the `utils/paths.py` to add the pretrained embedding and the dataset.

3. Long sentence clipping for the datasets MSRA and Ontonotes, or your datasets, run the command:
```bash
python sentence_clip.py
```

4. Merging char embeddings and word embeddings:
```bash
python char_word_mix.py
```

5. Model training and evaluation
    * Weibo dataset
    ```shell
    python main.py --dataset agcner
    ```
#### 3.6 Graph4CNER

##### Pretrained Embeddings:
Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec) can be downloaded in [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D).

Word embeddings (sgns.merge.word) can be downloaded in [Google Drive](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) or
[Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw).

##### Usage：

:one: Download the character embeddings and word embeddings and put them in the `data/embeddings` folder.

:two: Modify the `run_main.sh` by adding your train/dev/test file directory.

:three: `sh run_main.sh`. Note that the default hyperparameters is may not be the optimal hyperparameters, and you need to adjust these.

:four: Enjoy it! :smile:

#### 3.7 BERT-BiLSTM-CRF and BERT-IDCNN-CRF

The code for BERT-BiLSTM-CRF and BERT-IDCNN-CRF was listed in BERT-BiLSTM-CRF.

1. Pretrained Embeddings:
 * pre-trained language model BERT for Chinese NER can be downloaded in [Huggingface](https://www.huggingface.co/)
2. select the training model, BERT-BiLSTM-CRF or BERT-IDCNN-CRF
3. Model training and evaluation
    * agcner dataset
    ```shell
    python ner.py
    ```
#### HNER 
Hierarchical Transformer Model for Scientific Named Entity Recognition

****
1. convert dataset with `.txt` into `.pk` by using following command:
```angular2html
python  convert_to_pk.py
```
2. Model training and evaluation
   * agcner dataset
   ```angular2html
    name = "agcner_ner"
    dirpath= "."
    data_path = "data/agcner/agcner.pk"
    # create config
    config = create_config(name, dirpath=dirpath, data_path=data_path, max_epoch=30,
                           model_name='ckiplab/bert-base-chinese')
    # train model
    train_model(config)
   ```
3. Testing and predicting
   ```angular2html
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
  
    prediction = model.extract_predictions(sent_list)
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
   ```

# 4 ssAgBERT
Fine-turned BERT for named entity recognition of agricultural diseases and pests was also released in https://github.com/guojson/AgBERT.git

# 5 Thanks
   * LatticeLSTM: https://github.com/tyistyler/LatticeLSTM_torch_1.4.git
   * TENER: https://github.com/fastnlp/TENER.git
   * FLAT: https://github.com/LawsonAbs/Flat-ner.git
   * NFLAT: https://github.com/CoderMusou/NFLAT4CNER.git
   * Graph4CNER: https://github.com/DianboWork/Graph4CNER.git
   * HNER: https://github.com/urchade/HNER.git
   * HMM and CRF: https://github.com/luopeixiang/named_entity_recognition.git