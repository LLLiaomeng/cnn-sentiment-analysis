import torch
from torch.utils.data import Dataset
import numpy as np
import os
import re
from collections import defaultdict
import tarfile
import urllib.request

class subDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = torch.IntTensor(self.Data[index])
        label = self.Label[index]
        return data, label

def download_or_unzip():
    root = os.getcwd()
    path = os.path.join(root, "rt-polaritydata")
    if not os.path.isdir(path):
        tpath = os.path.join(root, "rt-polaritydata.tar")
        if not os.path.isfile(tpath):
            print('downloading')
            urllib.request.urlretrieve("https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz", tpath)
        with tarfile.open(tpath, 'r') as tfile:
            print('extracting')
            tfile.extractall(root)
    return os.path.join(path, '')

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def load_glove(filename, vocab, k=300):
    word_vec = defaultdict(float)
    with open(filename, "r", errors='ignore') as file:
        for line in file.readlines():
            word = line.split()[0]
            vec = line.split()[1:]
            if word in vocab:
                word_vec[word] = list(map(float, vec))
            else:
                word_vec[word] = np.random.uniform(-0.25, 0.25, k)          #Word not in glove is randomly initialized
    return word_vec


def load_data(filename, list_stopWords):
    sentences = []
    label = []
    cv = []
    vocab = defaultdict(float)
    for (file, ip) in filename.items():
        with open(file, "r", errors='ignore') as file:
            for line in file.readlines():
                cv.append(np.random.randint(0, 10))                 #cv is used to cross validation
                label.append(ip)
                a = clean_str(line).split()
                b = []
                for word in a:
                    if word not in list_stopWords:
                        vocab[word] = vocab[word] + 1
                        b.append(word)
                sentences.append(b)
    return vocab, label, sentences, cv


def W_idx(word_vec, k=300):
    word_idx = defaultdict(int)
    index = 1
    W = [np.zeros(k, dtype='float32').tolist()]
    for word in word_vec:
        W.append(word_vec[word])
        word_idx[word] = index
        index += 1
    return word_idx, W


def data_example(sentences, word_idx, max_len=30, k=300):
    data_idx = []
    for sentence in sentences:
        num = len(sentence)
        if (num > max_len):
            i = 1
            a = []
            for word in sentence:
                if i > max_len:
                    break
                else:
                    a.append(word_idx[word])
                i += 1
            data_idx.append(a)
        else:
            b = []
            for word in sentence:
                b.append(word_idx[word])
            for j in range(max_len - num):
                b.append(0)
            data_idx.append(b)
    return data_idx


def data_train_test(data_idx, label, cv, index):
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    idx = 0
    for i in cv:
        if i == index:
            test_data.append(data_idx[idx])
            test_label.append(label[idx])
        else:
            train_data.append(data_idx[idx])
            train_label.append(label[idx])
        idx += 1
    return train_data, test_data, train_label, test_label  