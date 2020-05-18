import numpy as np
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
import nltk
nltk.download('abc')
from nltk.corpus import abc
import itertools,re

corpus = []

for text_id in abc.fileids():
        raw_text = list(itertools.chain.from_iterable(abc.sents(text_id)))
        text = ' '.join(raw_text)
        text = text.lower()
        text.replace('\n', ' ')
        text = re.sub('[^a-z ]+', '', text)
        corpus.append([w for w in text.split() if w != ''])
        
from collections import Counter
import random, math

def subsample_frequent_words(corpus):
    filtered_corpus = []
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {word: word_counts[word]/float(sum_word_counts) for word in word_counts}
    for text in corpus:
        filtered_corpus.append([])
        for word in text:
            if random.random() < (1+math.sqrt(word_counts[word] * 1e3)) * 1e-3 / float(word_counts[word]):
                filtered_corpus[-1].append(word)
    return filtered_corpus

tokenized_corpus = subsample_frequent_words(corpus)
vocabulary = set(itertools.chain.from_iterable(tokenized_corpus))

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
index2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
num_epochs = 5
learning_rate = 0.005

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()    
    print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
