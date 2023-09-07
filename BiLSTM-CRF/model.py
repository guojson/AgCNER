import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F

from attention import Attention


class BiLSTM_CRF(nn.Module):
    def __init__(self, config, embedding_size, hidden_size, vocab_size, target_size, drop_out):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(target_size, batch_first=True)

    def forward(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embeddings)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    def forward_with_crf(self, input_ids, input_mask, input_tags):
        tag_scores = self.forward(input_ids)
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1)
        return tag_scores, loss

class IDCNN_CRF(nn.Module):

    def __init__(self, config, embedding_size, hidden_size, vocab_size, target_size, drop_out):
        super(IDCNN_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.dilation_l = config.dilation_l

        if self.dilation_l is None:
            self.dilation_l = [1, 2, 1]
        self.num_blocks = config.blocks

        self.drop_penalty = config.drop_penalty
        self.num_labels = target_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.filters = config.filters
        cnn_kernel_size = config.kernel_size
        padding_word = int(cnn_kernel_size / 2)
        self.conv0 = nn.Conv1d(in_channels=embedding_size,
                               out_channels=self.filters,
                               kernel_size=cnn_kernel_size,
                               padding=padding_word)
        self.cov_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=embedding_size,
                      kernel_size=cnn_kernel_size,
                      padding=padding_word * dilation,
                      dilation=dilation) for dilation in self.dilation_l
        ])
        self.conv_layers_size = len(self.cov_layers)

        # 全连接层
        self.dense = nn.Linear(in_features=(embedding_size * self.num_blocks),
                               out_features=self.num_labels)

        self.i_drop = nn.Dropout(config.input_dropout)
        self.m_drop = nn.Dropout(config.middle_dropout)
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embedding_size)

        # 全连接层
        # self.classifier = nn.Linear(in_features=self.hidden_size, out_features=target_size)

        self.crf = CRF(target_size, batch_first=True)

    def forward(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        feature = self.i_drop(embeddings)
        feature = feature.permute(0, 2, 1)
        conv0 = self.conv0(feature)
        conv0 = F.relu(conv0)
        conv_layer = conv0
        conv_outputs = []
        for _ in range(self.num_blocks):
            for j, mdv in enumerate(self.cov_layers):
                conv_layer = mdv(conv_layer)
                conv_layer = F.relu(conv_layer)
                if j == self.conv_layers_size - 1:
                    conv_layer = self.m_drop(conv_layer)
                    conv_outputs.append(conv_layer)
        layer_concat = torch.cat(conv_outputs, 1)
        layer_concat = layer_concat.permute(0, 2, 1)

        return self.dense(layer_concat)

    def forward_with_crf(self, input_ids, input_mask, input_tags):
        tag_scores = self.forward(input_ids)
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1)
        return tag_scores, loss


class BiLSTM_Attention_CRF(nn.Module):

    def __init__(self, config, embedding_size, hidden_size, vocab_size, target_size, drop_out):
        super(BiLSTM_Attention_CRF, self).__init__()
        self.hidden_size = hidden_size
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, target_size)
        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(target_size, batch_first=True)
        self.attention = Attention(embed_dim=hidden_size * 2, hidden_dim=128,
                                   out_dim=hidden_size * 2)  # embed_dim 和 out_dim为标签长度

    def forward(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embeddings)
        out, _ = self.attention(sequence_output, sequence_output)


        # M = F.tanh(sequence_output)  # [128, 32, 256]
        # # M = torch.tanh(torch.matmul(H, self.u))
        # alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        # out = sequence_output * alpha  # [128, 32, 256]
        # out = torch.sum(out, 1)  # [128, 256]
        # out = F.relu(out)
        out = self.fc1(out)

        tag_scores = self.classifier(out)
        return tag_scores

    def forward_with_crf(self, input_ids, input_mask, input_tags):
        tag_scores = self.forward(input_ids)

        loss = self.crf(tag_scores, input_tags, input_mask) * (-1)
        return tag_scores, loss
