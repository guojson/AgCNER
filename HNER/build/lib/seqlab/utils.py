import torch
from torch.nn.utils.rnn import pad_sequence


def select_first_subword(hidden_state, subword_mask, seq_lengths, padding_value=-1):

    dim = hidden_state.size(-1)

    subword_mask = subword_mask.unsqueeze(-1)

    mask_sel = torch.masked_select(hidden_state, subword_mask).view(-1, dim)

    valid_seq = torch.split(mask_sel, seq_lengths)

    padded = pad_sequence(valid_seq, batch_first=True,
                          padding_value=padding_value)

    word_mask = padded.sum(-1) != padding_value * padded.shape[-1]

    return {
        'first_subword': padded,
        'word_mask': word_mask
    }
