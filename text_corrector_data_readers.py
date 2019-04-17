from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from data_reader import DataReader, PAD_TOKEN, EOS_TOKEN, GO_TOKEN, EOS_ID
from itertools import izip


class PTBDataReader(DataReader):
    """
    DataReader used to read in the Penn Treebank dataset.
    """

    UNKNOWN_TOKEN = "<unk>"  # already defined in the source data

    DROPOUT_WORDS = {"a", "an", "the"}
    DROPOUT_PROB = 0.25

    REPLACEMENTS = {"there": "their", "their": "there"}
    REPLACEMENT_PROB = 0.25

    def __init__(self, config, train_path):
        super(PTBDataReader, self).__init__(
            config, train_path, special_tokens=[PAD_TOKEN, GO_TOKEN, EOS_TOKEN])

        self.UNKNOWN_ID = self.token_to_id[PTBDataReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):

        for line in self.read_tokens(path):
            source = []
            target = []

            for token in line:
                target.append(token)

                # Randomly dropout some words from the input.
                dropout_word = (token in PTBDataReader.DROPOUT_WORDS and
                                random.random() < PTBDataReader.DROPOUT_PROB)
                replace_word = (token in PTBDataReader.REPLACEMENTS and
                                random.random() <
                                PTBDataReader.REPLACEMENT_PROB)

                if replace_word:
                    source.append(PTBDataReader.REPLACEMENTS[token])
                elif not dropout_word:
                    source.append(token)

            yield source, target

    def unknown_token(self):
        return PTBDataReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.rstrip().lstrip().split()


class MovieDialogReader(DataReader):
    """
    DataReader used to read and tokenize data from the Cornell open movie
    dialog dataset.
    """

    UNKNOWN_TOKEN = "UNK"

    DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}  # Add "to"

    REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                    "than": "then"}
    # Add: "be":"to"

    def __init__(self, config, train_path=None, token_to_id=None,
                 dropout_prob=0.25, replacement_prob=0.25, dataset_copies=2):
        super(MovieDialogReader, self).__init__(
            config, path=train_path, token_to_id=token_to_id,
            special_tokens=[
                PAD_TOKEN, GO_TOKEN, EOS_TOKEN,
                MovieDialogReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)


        self.UNKNOWN_ID = self.token_to_id[MovieDialogReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, path):
        for tokens in self.read_tokens(path):
            source = []
            target = []

            for token in tokens:
                target.append(token)

                # Randomly dropout some words from the input.
                dropout_token = (token in MovieDialogReader.DROPOUT_TOKENS and
                                random.random() < self.dropout_prob)
                replace_token = (token in MovieDialogReader.REPLACEMENTS and                        
                                random.random() < self.replacement_prob)

                if replace_token:
                    source.append(MovieDialogReader.REPLACEMENTS[token])
                elif not dropout_token:
                    source.append(token)

            yield source, target

    def unknown_token(self):
        return MovieDialogReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.lower().strip().split()

class SentencePairReader(DataReader):

    UNKNOWN_TOKEN = "UNK"

    def __init__(self, config, path=None, token_to_id=None, dataset_copies=2):
        super(SentencePairReader, self).__init__(
            config, path=path, token_to_id=token_to_id,
            special_tokens=[
                PAD_TOKEN, GO_TOKEN, EOS_TOKEN,
                SentencePairReader.UNKNOWN_TOKEN],
            dataset_copies=dataset_copies)
        
        self.UNKNOWN_ID = self.token_to_id[SentencePairReader.UNKNOWN_TOKEN]

    def read_samples_by_string(self, input_path, target_path):
        with open(input_path, "r") as fa, open(target_path, "r")  as fb:
            for line_a, line_b in izip(fa, fb):
                yield line_a.strip().split(), line_b.strip().split()

    def unknown_token(self):
        return SentencePairReader.UNKNOWN_TOKEN

    def read_tokens(self, path):
        with open(path, "r") as f:
            for line in f:
                yield line.lower().strip().split()

    def read_samples(self, input_path, target_path):
         for source_words, target_words in self.read_samples_by_string(input_path, target_path):         
             source = [self.convert_token_to_id(word) for word in source_words]
             target = [self.convert_token_to_id(word) for word in target_words]
             target.append(EOS_ID)                                                    
 
             yield source, target    
       
    def build_dataset(self, input_path, target_path):
        dataset = [[] for _ in self.config.buckets]

        # Make multiple copies of the dataset so that we synthesize different
        # dropouts.
        for _ in range(self.dataset_copies):
            for source, target in self.read_samples(input_path, target_path):
                for bucket_id, (source_size, target_size) in enumerate(
                        self.config.buckets):
                    if len(source) < source_size and len(
                            target) < target_size:
                        dataset[bucket_id].append([source, target])
                        break

        return dataset

