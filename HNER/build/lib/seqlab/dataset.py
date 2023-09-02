import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class DataSeqLab(object):
    """
    Data processing for sequence labelling

    Args:
        tokenizer : huggingface tokenizer
        label_to_id : dictionnary mapping
    """
    def __init__(self, tokenizer, tag_to_id):

        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id

    def tokenize_and_align_labels(self, sample):
        """
        Align tokens and labels

        Args:
            sample (dict): a data sample with tokens and tags.

        Returns:
            dict
        """

        tokens = sample['tokens']

        # for prediction when labels are missing
        if 'tags' in sample:
            tags = sample['tags']
        else:
            tags = len(tokens) * ['O']

        encoded_sentence = []
        aligned_labels = []
        word_label = []
        iob_labels = []

        for t, n in zip(tokens, tags):

            encoded_token = self.tokenizer.tokenize(t)

            if len(encoded_token) < 1:
                encoded_token = [self.tokenizer.unk_token]

            encoded_token = self.tokenizer.convert_tokens_to_ids(encoded_token)

            n_subwords = len(encoded_token)

            if len(encoded_sentence) + len(encoded_token) > 512:
                break

            encoded_sentence.extend(encoded_token)

            aligned_labels.extend(
                [self.tag_to_id[n]] + (n_subwords - 1) * [-1]
            )

            word_label.append(self.tag_to_id[n])
            iob_labels.append(n)

        assert len(encoded_sentence) == len(aligned_labels)

        encoded_sentence = torch.LongTensor(encoded_sentence)
        aligned_labels = torch.LongTensor(aligned_labels)
        word_label = torch.LongTensor(word_label)

        lengths = len(iob_labels)

        return {
            'input_ids': encoded_sentence, 'aligned_labels': aligned_labels,
            'iob_labels': iob_labels, 'seq_length': lengths, 'word_label': word_label
        }

    def collate_fn(self, batch_as_list):
        """
        Batchification

        Args:
            batch_as_list (list[dict]): a list of dict with "tokens" and "tags"

        Returns:
            Input tensors for model
        """

        batch = [self.tokenize_and_align_labels(b) for b in batch_as_list]

        input_ids = pad_sequence([b['input_ids'] for b in batch],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)

        aligned_labels = pad_sequence(
            [b['aligned_labels'] for b in batch], batch_first=True, padding_value=-1)

        iob_labels = [b['iob_labels'] for b in batch]

        word_label = pad_sequence(
            [b['word_label'] for b in batch], batch_first=True, padding_value=0)

        seq_length = [b['seq_length'] for b in batch]

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()

        subword_mask = aligned_labels != -1

        return {
            'input_ids': input_ids,
            'aligned_labels': aligned_labels,
            'attention_mask': attention_mask,
            'seq_length': seq_length,
            'subword_mask': subword_mask,
            'word_label': word_label,
            'iob_labels': iob_labels
        }

    def create_dataloader(self, data, batch_size=2, num_workers=1, **kwgs):
        """
        Create a torch dataloader

        Args:
            data (list[dict]): [{'tokens': [...], 'tags': [O, O, O]}]
            batch_size (int): batch_size for dataloader
            num_workers (int): num_workers for dataloader

        Returns:
            Torch dataloader
        """
        return DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn, **kwgs)