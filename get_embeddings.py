import torch
import torch.nn as nn
import numpy as np
import json
from tokenizers import Tokenizer
from preprocessing import cyrillize, pattern
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import regexp_tokenize


def get_sub_word_tokenization_embedding(dim=100):
    tokenizer = Tokenizer.from_file("data/tokenizer_comments.json")
    token2ind = tokenizer.get_vocab()
    ind2token = lambda x: tokenizer.id_to_token(x)

    with open('data/unsupervised_comments.json', 'r', encoding="utf-8") as f:
        unsupervised_comments = json.load(f)

    tokenized_unsupervised_comments = [tokenizer.encode(c.lower()).tokens for c in unsupervised_comments]

    n_words = tokenizer.get_vocab_size()
    X=np.zeros((n_words,n_words))
    for s in ["[UNK]", "[PAD]", "[STR]", "[END]"]:
        X[token2ind[s], token2ind[s]] = 1
    for comment in tokenized_unsupervised_comments:
        for wi in range(len(comment)):
            if comment[wi] not in token2ind: continue
            i=token2ind[comment[wi]]
            for k in range(1,4+1):
                if wi-k>=0 and comment[wi-k] in token2ind:
                    j=token2ind[comment[wi-k]]
                    X[i,j] += 1
                if wi+k<len(comment) and comment[wi+k] in token2ind:
                    j=token2ind[comment[wi+k]]
                    X[i,j] += 1

    svd = TruncatedSVD(n_components=dim, n_iter=10)
    svd.fit(X)
    X_reduced = svd.transform(X)

    return lambda comment: np.stack([X_reduced[token2ind[token]] for token in tokenizer.encode(comment.lower()).tokens])


def get_noise_dampening_embedding(dim, device):
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, dropout_p=0.1):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(dropout_p)

        def forward(self, input):
            embedded = self.dropout(self.embedding(input))
            output, hidden = self.gru(embedded)
            return output, hidden

        def save(self, filepath):
            torch.save(self.state_dict(), filepath)

        @classmethod
        def load(cls, filepath, input_size, hidden_size, dropout_p=0.1):
            model = cls(input_size, hidden_size, dropout_p)
            model.load_state_dict(torch.load(filepath))
            return model

    SOW_token = ''
    EOW_token = ''
    UNK_token = '�'

    alphabet_for_generation = 'абвгдежзийклмнопрстуфхцчшщьъюяabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+[]\';.,/`~"<>|1234567890'
    alphabet = alphabet_for_generation
    alphabet += SOW_token
    alphabet += EOW_token
    alphabet += UNK_token

    char2ind = {}
    for i, c in enumerate(alphabet):
        char2ind[c] = i

    def indexesFromWord(word):
        return [(char2ind[c] if c in char2ind else char2ind[UNK_token]) for c in word]

    def tensorFromWord(word):
        indexes = indexesFromWord(word)
        indexes.append(char2ind[EOW_token])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

    encoder = EncoderRNN.load("data/embedding_encoder_100_000_smaller_alphabet.pth", 96, 128)
    encoder.to(device)
    encoder.eval()

    def embedding(word):
        if word == '[PAD]':
            return torch.zeros(1, 1, 128)
        with torch.no_grad():
            input_tensor = tensorFromWord(word)
            _, encoder_hidden = encoder(input_tensor)
        return encoder_hidden

    # reducing dims with svd on the unsupervised comments
    with open('data/unsupervised_comments.json', 'r', encoding="utf-8") as f:
        unsupervised_comments = json.load(f)
    vocabulary = set([t for comment in unsupervised_comments for t in regexp_tokenize(comment.lower(), pattern)])
    X = np.vstack([embedding(t).cpu().flatten() for t in vocabulary])
    svd = TruncatedSVD(n_components=dim, n_iter=10)
    svd.fit(X)

    def comment_embedding(comment):
        tokens = [
            svd.transform(embedding(word).cpu().flatten(end_dim=1)).flatten()
            for word in regexp_tokenize(comment.lower(), pattern)
        ]
        if len(tokens) != 0:
            return np.vstack(tokens)
        return []

    return comment_embedding