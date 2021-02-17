import re
from pathlib import Path

import numpy as np

from datamaestro import prepare_dataset
import torch
from torch.utils.data import Dataset

class DatasetIMDB(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return torch.tensor(data), target

    def __len__(self):
        return len(self.dataset)

class DatasetDBpedia(Dataset):
    def __init__(self, ds_dbpedia, word2id, OOVID=None, WORDS=None):
        self.vocab_itos = ds_dbpedia.get_vocab().itos
        self.dataset = ds_dbpedia
        self.word2id = word2id
        self.OOVID = len(word2id)-2 if OOVID is None else OOVID
        self.WORDS = re.compile(r"\S+") if WORDS is None else WORDS

    def __getitem__(self, index):
        target, tokens = self.dataset[index]
        text = " ".join([self.vocab_itos[token] for token in tokens])
        return self.tokenizer(text), target

    def __len__(self):
        return len(self.dataset)

    def tokenizer(self, t):
        return torch.tensor([self.word2id.get(x, self.OOVID) for x in re.findall(self.WORDS, t.lower())])


class DatasetLM1B(Dataset):
    def __init__(self, path_folder, word2id, OOVID=None, WORDS=None):
        self.word2id = word2id
        self.OOVID = len(word2id)-2 if OOVID is None else OOVID
        self.WORDS = re.compile(r"\S+") if WORDS is None else WORDS
        self.dataset = []
        
        for file in os.listdir(path_folder):
            with open(path_folder+"/"+file, "r") as f:
                for line in f.readlines():
                    self.dataset.append(line)


    def __getitem__(self, index):
        return self.tokenizer(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def tokenizer(self, t):
        return [self.word2id.get(x, self.OOVID) for x in re.findall(self.WORDS, t.lower())]


class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="utf-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    PAD = len(words)+1
    words.append("__OOV__")
    words.append("__PAD__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)
