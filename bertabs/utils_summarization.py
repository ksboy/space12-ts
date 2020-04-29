import os
from collections import deque

import torch
from torch.utils.data import Dataset
import json, csv

# ------------
# Data loading
# ------------

class CNNDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, filepath="", prefix="train"):
        """ We initialize the class by listing all the documents to summarize.
        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).
        """
        # assert os.path.isfile(filepath)
        # self.filepath = filepath
        self.sources = open(os.path.join(filepath, "test.source"),encoding='utf-8').readlines()
        self.targets = open(os.path.join(filepath, "test.target"),encoding='utf-8').readlines()

    def __len__(self):
        """ Returns the number of documents. """
        # with open(self.path, encoding="utf-8") as source:
        #     raw_story = source.readlines()
        # return len(raw_story)
        return len(self.sources)

    def __getitem__(self, idx):
        return str(idx), self.sources[idx], self.targets[idx]


class SpaceDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, filepath="", prefix="train"):
        """ We initialize the class by listing all the documents to summarize.
        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).
        """
        assert os.path.isfile(filepath)
        self.filepath = filepath

    def __len__(self):
        """ Returns the number of documents. """
        # with open(self.path, encoding="utf-8") as source:
        #     raw_story = source.readlines()
        # return len(raw_story)
        count = 0
        for index, line in enumerate(open(self.filepath,'r')):
            if line=='\n' or line=='': continue
            count += 1
        return count

    def __getitem__(self, idx):
        with open(self.filepath ,encoding='utf-8') as f:
            reader = csv.DictReader(f)
            title = reader.fieldnames
            for index, row in enumerate(reader):
                if index == idx:
                    # print (row )
                    return str(idx), [row['con_text']], [row['con_title']]


# --------------------------
# Encoding and preprocessing
# --------------------------


def fit_to_block_size(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask


def encode_for_summarization(story_lines, summary_lines, tokenizer):
    """ Encode the story and summary lines, and join them
    as specified in [1] by using `[SEP] [CLS]` tokens to separate
    sentences.
    """
    story_lines_token_ids = [tokenizer.encode(line) for line in story_lines]
    story_token_ids = [token for sentence in story_lines_token_ids for token in sentence]
    summary_lines_token_ids = [tokenizer.encode(line) for line in summary_lines]
    summary_token_ids = [token for sentence in summary_lines_token_ids for token in sentence]

    return story_token_ids, summary_token_ids


def compute_token_type_ids(batch, separator_token_id):
    """ Segment embeddings as described in [1]

    The values {0,1} were found in the repository [2].

    Attributes:
        batch: torch.Tensor, size [batch_size, block_size]
            Batch of input.
        separator_token_id: int
            The value of the token that separates the segments.

    [1] Liu, Yang, and Mirella Lapata. "Text summarization with pretrained encoders."
        arXiv preprint arXiv:1908.08345 (2019).
    [2] https://github.com/nlpyang/PreSumm (/src/prepro/data_builder.py, commit fac1217)
    """
    batch_embeddings = []
    for sequence in batch:
        sentence_num = -1
        embeddings = []
        for s in sequence:
            if s == separator_token_id:
                sentence_num += 1
            embeddings.append(sentence_num % 2)
        batch_embeddings.append(embeddings)
    return torch.tensor(batch_embeddings)
