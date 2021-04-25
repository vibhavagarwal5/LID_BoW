import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

lang = ['eng', 'fra', 'ita', 'deu', 'spa']

lang2id = {}
for i, l in enumerate(lang):
    lang2id[l] = i


def get_data(path):
    data = pd.read_csv(path, sep='\t', encoding='utf8',
                       index_col=0, names=['lang', 'text'])
    data = data[[True if len(s) > 10 else False for s in data['text']]]
    data = data[data['lang'].isin(lang)]
    return data


def get_trigrams(data, num_feats=200):
    vectorizer = CountVectorizer(analyzer='char',
                                 ngram_range=(3, 3), max_features=num_feats)
    X = vectorizer.fit_transform(data)
    grams = vectorizer.get_feature_names()
    return grams


def create_vocab(data):
    vocab = set()
    for l in lang:
        corpus = data[data.lang == l]['text']
        trigrams = get_trigrams(corpus)
        vocab.update(trigrams)
    return vocab


def save_pkl(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


class BoWDataset(Dataset):
    def __init__(self, data_type, path, args):
        self.data = pd.read_csv(os.path.join(path, f'{data_type}.csv'), encoding='utf8',
                                names=['lang', 'text'])
        with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        self.args = args
        self.vectorizer = CountVectorizer(analyzer='char',
                                          ngram_range=(3, 3),
                                          vocabulary=self.vocab)

    def create_feats(self, idx):
        X = self.vectorizer.fit_transform([self.data['text'][idx]]).toarray()
        # X = (X - X.min()) / (X.max() - X.min())

        y = lang2id[self.data['lang'][idx]]
        return X, y

    def __len__(self):
        return len(self.data['lang'])

    def __getitem__(self, index):
        X, y = self.create_feats(index)
        return X, y


class Collatefn:
    def __call__(self, batch):
        X = [i[0] for i in batch]
        X = np.array(X)
        X = (X - X.min()) / (X.max() - X.min())

        y = [i[1] for i in batch]

        X = torch.tensor(X).float()
        y = torch.tensor(y).long()
        return X, y


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--prepare_data", action="store_true",
                        help="To prepare data before training")
    parser.add_argument("--save", action="store_true",
                        help="Not to save the prepared data")
    parser.add_argument("--data_path", type=str,
                        default="./data", help="Path of the dataset")
    args = parser.parse_args()

    if args.prepare_data:
        '''
        Prepare data
        '''

        data = get_data(os.path.join(args.data_path, 'sentences.csv'))
        train, test = train_test_split(data, test_size=0.3, random_state=42)
        train, val = train_test_split(train, train_size=0.8, random_state=42)
        print(data.shape, train.shape, test.shape, val.shape)
        vocab = create_vocab(train)
        if args.save:
            save_pkl(vocab, os.path.join(args.data_path, 'vocab.pkl'))
            train.to_csv(os.path.join(args.data_path, 'train.csv'),
                         index=False, header=None)
            test.to_csv(os.path.join(args.data_path, 'test.csv'),
                        index=False, header=None)
            val.to_csv(os.path.join(args.data_path, 'val.csv'),
                       index=False, header=None)

    else:
        '''Testing'''
        from torch.utils.data import DataLoader

        dataset = BoWDataset('train', args.data_path, None)
        dataloader = DataLoader(dataset,
                                batch_size=4,
                                collate_fn=Collatefn(), shuffle=True)

        batch = next(iter(dataloader))
        X, y = batch
        print(X, X.shape)
        print(y, y.shape)
