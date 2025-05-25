import os
import torch

# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Corpus(object):
    def __init__(self, path, batch_size, max_sql, word_id=None, vocabulary=None):
        if word_id is not None and vocabulary is not None:
            self.word_id = word_id
            self.vocabulary = vocabulary
            self.train = self.tokenize(os.path.join(path, 'train.txt'), use_given_vocab=True)
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'), use_given_vocab=True)
        else:
            self.vocabulary = []
            self.word_id = {}
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.dset_flag = "train"
        self.max_sql = max_sql
        self.batch_size = batch_size
        print("size of train set: ", self.train.size(0))
        print("size of valid set: ", self.valid.size(0))
        self.train_batch_num = self.train.size(0) // self.batch_size["train"]
        self.valid_batch_num = self.valid.size(0) // self.batch_size["valid"]
        self.train = self.train.narrow(0, 0, self.batch_size["train"] * self.train_batch_num)
        self.valid = self.valid.narrow(0, 0, self.batch_size["valid"] * self.valid_batch_num)
        self.train = self.train.view(self.batch_size["train"], -1).t().contiguous()
        self.valid = self.valid.view(self.batch_size["valid"], -1).t().contiguous()

    def set_train(self):
        self.dset_flag = "train"
        self.train_si = 0

    def set_valid(self):
        self.dset_flag = "valid"
        self.valid_si = 0

    def tokenize(self, file_name, use_given_vocab=False):
        file_lines = open(file_name, 'r').readlines()
        ids = []
        for line in file_lines:
            words = line.strip().split() + ['<eos>']
            for word in words:
                if use_given_vocab:
                    idx = self.word_id.get(word, self.word_id.get('<unk>', 0))
                else:
                    if word not in self.word_id:
                        self.word_id[word] = len(self.vocabulary)
                        self.vocabulary.append(word)
                    idx = self.word_id[word]
                ids.append(idx)
        return torch.tensor(ids, dtype=torch.long)

    def get_batch(self):
        ## train_si and valid_si indicates the index of the start point of the current mini-batch
        if self.dset_flag == "train":
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(0) - self.train_si - 1)
            data_loader = self.train
            self.train_si = self.train_si + seq_len
        else:
            start_index = self.valid_si
            seq_len = min(self.max_sql, self.valid.size(0) - self.valid_si - 1)
            data_loader = self.valid
            self.valid_si = self.valid_si + seq_len
        # Load a truncation of word token ids
        data = data_loader[start_index:start_index + seq_len]
        target = data_loader[start_index + 1:start_index + seq_len + 1].view(-1)

        ## end_flag indicates whether a epoch (train or valid epoch) has been ended
        if self.dset_flag == "train" and self.train_si + 1 == self.train.size(0):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == "valid" and self.valid_si + 1 == self.valid.size(0):
            end_flag = True
            self.valid_si = 0
        else:
            end_flag = False

        # Using data to generate target, each time generate the next word
        return data, target, end_flag


if __name__ == "__main__":
    dataset = Corpus(path="../data/ptb")