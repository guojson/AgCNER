import config
import torch

from Vocabulary import Vocabulary

# if config.gpu != '':
#     device = torch.device(f"cuda:{config.gpu}")
# else:
device = torch.device("cpu")

vocab = Vocabulary(config)
vocab.get_vocab()
word='水稻恶苗病茎节受害状及侧生的不定根'
word = list(word)
token_len = len(word)


word_id = [[vocab.word_id(w_) for w_ in word]]

label2id = config.label2id

word_id = torch.tensor(word_id, dtype=torch.long).to(device)

masks = torch.ByteTensor(1, token_len).fill_(0)
masks[0, :len(word_id[0])] = torch.tensor([1] * len(word_id[0]), dtype=torch.uint8).to(device)



model = torch.load(config.model_dir)
model.to(device)

y_pred = model.forward(word_id)
labels_pred = model.crf.decode(y_pred, mask=masks)
pred = [[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred]
print(pred)
