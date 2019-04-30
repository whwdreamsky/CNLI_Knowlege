# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

"""
Utility functions.
"""

import logging
import nltk
import os
import json
import numpy as np
import tensorflow as tf
from collections import Counter
from nltk.tokenize.regexp import RegexpTokenizer
from RTEDataset import RTEDataset
import classifiers
from CNLIDataset import CNLIDataset
tokenizer = nltk.tokenize.TreebankWordTokenizer()
UNKNOWN = '_UNK_'
PADDING = '_PAD_'
GO = '_EOS_'  # it's called "GO" but actually serves as a null alignment


#UNKNOWN = u'_UNK'
#PADDING = u'_PAD'
#GO = u'_GO'  # it's called "GO" but actually serves as a null alignment
NOMATCH = -1
def get_tokenizer(language):
    """
    Return the tokenizer function according to the language.
    """
    language = language.lower()
    if language == 'en':
        tokenize = tokenize_english
    elif language == 'pt':
        tokenize = tokenize_portuguese
    else:
        ValueError('Unsupported language: %s' % language)

    return tokenize


def tokenize_english(text):
    """
    Tokenize a piece of text using the Treebank tokenizer

    :return: a list of strings
    """
    return tokenizer.tokenize(text)


def tokenize_portuguese(text):
    """
    Tokenize the given sentence in Portuguese. The tokenization is done in
    conformity  with Universal Treebanks (at least it attempts so).

    :param text: text to be tokenized, as a string
    """
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    # more structured patterns come first
    [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|    # emails
    (?:[\#@]\w+)|                     # Hashtags and twitter user names
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    \b\d+(?:[-:.,]\w+)*(?:[.,]\d+)?\b|
        # numbers in format 999.999.999,999, or hyphens to alphanumerics
    \.{3,}|                           # ellipsis or sequences of dots
    (?:\w+(?:\.\w+|-\d+)*)|           # words with dots and numbers, possibly followed by hyphen number
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

    return tokenizer.tokenize(text)


def tokenize_corpus(pairs):
    """
    Tokenize all pairs.

    :param pairs: a list of tuples (sent1, sent2, relation)
    :return: a list of tuples as in pairs, except both sentences are now lists
        of tokens
    """
    tokenized_pairs = []
    for sent1, sent2, label in pairs:
        tokens1 = tokenize_english(sent1)
        tokens2 = tokenize_english(sent2)
        tokenized_pairs.append((tokens1, tokens2, label))

    return tokenized_pairs


def count_parameters():
    """
    Count the number of trainable tensorflow parameters loaded in
    the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        logging.debug('%s: %d params' % (variable.name, variable_params))
        total_params += variable_params
    return total_params


def count_corpus_tokens(pairs):
    """
    Examine all pairs ans extracts all tokens from both text and hypothesis.

    :param pairs: a list of tuples (sent1, sent2, relation) with tokenized
        sentences
    :return: a Counter of lowercase tokens
    """
    c = Counter()
    for sent1, sent2, _ in pairs:
        c.update(t.lower() for t in sent1)
        c.update(t.lower() for t in sent2)

    return c


def config_logger(verbose):
    """
    Setup basic logger configuration

    :param verbose: boolean
    :return:
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='%(message)s', level=level)


def get_logger(name='logger'):
    """
    Setup and return a simple logger.
    :return:
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.propagate = False

    return logger




def get_model_class(params):
    """
    Return the class of the model object

    :param params: saved parameter dictionary
    :return: a subclass of classifiers.DecomposableNLIModel
    """
    if params.get('model') == 'lstm':
        model_class = classifiers.LSTMClassifier
    else:
        model_class = classifiers.MultiFeedForwardClassifier

    assert issubclass(model_class, classifiers.DecomposableNLIModel)
    return model_class


def create_label_dict(pairs):
    """
    Return a dictionary mapping the labels found in `pairs` to numbers
    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a dict
    """
    labels = set(pair[2] for pair in pairs)
    mapping = zip(labels, range(len(labels)))
    return dict(mapping)


def convert_labels(pairs, label_map):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :param label_map: dictionary mapping label strings to numbers
    :return: a numpy array
    """
    print(label_map)
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)

def _pad_wordpair_to_indices(paded_sent1,paded_sent2,wordspair,
                                word_dict,max_len=None,flag='first'):
    #if max_len1 is None or:
    #    maxlen = paded_sent
    #print(paded_sent1.shape)
    wordindex_pair = []
    for i,item in enumerate(wordspair):
        wordindexpair = []
        for wp in item:
            w1 = word_dict[wp[0]]
            w2 = word_dict[wp[1]]
            #print(w1,w2)
            #print("sent1:  ")
            #print(paded_sent1[i])
            #print("sent2:  ")
            #print(paded_sent2[i])
            try:
                index_1 = list(paded_sent1[i]).index(w1)
                index_2 = list(paded_sent2[i]).index(w2)
                #print(index_1,"**********",index_2)
            except:
                continue
            wordindexpair.append((index_1,index_2))
        wordindex_pair.append(wordindexpair)
    
    shape = (len(wordspair),max_len)
    wordpairpaded = np.full(shape,NOMATCH, dtype=np.int32)
    for i ,item in enumerate(wordindex_pair):
        for pair in item:
            if flag=='first':
                wordpairpaded[i][pair[0]] = pair[1]
            else:
                wordpairpaded[i][pair[1]] = pair[0]
    #print(wordpairpaded)
    return wordpairpaded

def _produce_attention_bias(sent1_wordpair,sent2_wordpair,max_len1,max_len2):
    shape = (sent1_wordpair.shape[0],max_len1,max_len2)
    wordpairpaded = np.full(shape,0, dtype=np.float32)
    for (x,y),value in np.ndenumerate(sent1_wordpair):
        if value !=NOMATCH:
            wordpairpaded[x][y][value] +=1
    for (x,y),value in np.ndenumerate(sent2_wordpair):
        if value !=NOMATCH:
            wordpairpaded[x][value][y] +=1

    return wordpairpaded



def getchar_seq(tokens,sizes,maxlen_char,chardict):
    # 因为这里 要经过一个 PADDing 的操作，所以实际上 size 的长度是大于token 的
    charseqs = []
    for size,token in zip(sizes,tokens):
        charseq = []
        for i in range(size):
            if i>= len(token):
                charseq.append(getCharSequence("",chardict,maxlen_char))
            else:
                charseq.append(getCharSequence(token[i],chardict,maxlen_char))
        charseqs.append(charseq)
    return charseqs

def getCharSequence(word,chardict,maxlen_char=10):
    sequence = np.full([maxlen_char],chardict['_PAD_CHAR_'])
    for i,c in enumerate(word):
        if i>= maxlen_char:
            break
        if c in chardict:
            sequence[i] = chardict[c]
        else:
            sequence[i] = chardict["_UNK_CHAR_"]
    return sequence.tolist()
def getchar_seq(tokenlist,maxlen_word,maxlen_char,chardict):
    # 因为这里 要经过一个 PADDing 的操作，所以实际上 size 的长度是大于token 的
    charseqs = []
    for tokens in tokenlist:
        charseq = []
        for i in range(maxlen_word):
            if i>= len(tokens):
                charseq.append(getCharSequence("",chardict,maxlen_char))
            else:
                charseq.append(getCharSequence(tokens[i],chardict,maxlen_char))
        charseqs.append(charseq)
    return charseqs

def create_dataset(pairs,wordpairs,word_dict, label_dict=None,
                   max_len1=None, max_len2=None,maxlen_char=6):
    """
    Generate and return a RTEDataset object for storing the data in numpy format.

    :param pairs: list of tokenized tuples (sent1, sent2, label)
    :param word_dict: a dictionary mapping words to indices
    :param label_dict: a dictionary mapping labels to numbers. If None,
        labels are ignored.
    :param max_len1: the maximum length that arrays for sentence 1
        should have (i.e., time steps for an LSTM). If None, it
        is computed from the data.
    :param max_len2: same as max_len1 for sentence 2
    :return: RTEDataset
    """
    tokens1 = [pair[0] for pair in pairs]
    tokens2 = [pair[1] for pair in pairs]

    sentences1, sizes1,max_len1 = _convert_pairs_to_indices(tokens1, word_dict,
                                                   max_len1)
    sentences2, sizes2,max_len2 = _convert_pairs_to_indices(tokens2, word_dict,
                                                   max_len2)
    if label_dict is not None:
        labels = convert_labels(pairs, label_dict)
    else:
        labels = None
    sent1_charseq = getchar_seq(tokens1,max_len1,maxlen_char=maxlen_char,chardict=word_dict)
    sent2_charseq = getchar_seq(tokens2,max_len2,maxlen_char=maxlen_char,chardict=word_dict)

    return CNLIDataset(sentences1, sentences2, sizes1, sizes2, labels,sent1_charseq,sent2_charseq),max_len1,max_len2


def _convert_pairs_to_indices(sentences, word_dict, max_len=None,
                              use_null=True):
    """
    Convert all pairs to their indices in the vector space.

    The maximum length of the arrays will be 1 more than the actual
    maximum of tokens when using the NULL symbol.

    :param sentences: list of lists of tokens
    :param word_dict: mapping of tokens to indices in the embeddings
    :param max_len: maximum allowed sentence length. If None, the
        longest sentence will be the maximum
    :param use_null: prepend a null symbol at the beginning of each
        sentence
    :return: a tuple with a 2-d numpy array for the sentences and
        a 1-d array with their sizes
    """
    sizes = np.array([len(sent) for sent in sentences])
    # 注意这里的maxlen 要保证比之前大1
    if use_null:
        sizes += 1
    #    if max_len is not None:
    #        max_len += 1

    if max_len is None:
        max_len = sizes.max()

    shape = (len(sentences), max_len)
    print(len(word_dict))
    array = np.full(shape, word_dict[PADDING], dtype=np.int32)

    for i, sent in enumerate(sentences):
        indices = []
        for token in sent:
            if token in word_dict:
                indices.append(word_dict[token])
            else:
                indices.append(word_dict[UNKNOWN])
        #indices = [word_dict[token] for token in sent if token in word_dict]
        if use_null:
            indices = indices + [word_dict[GO]]
        if len(indices) > max_len:
            array[i,:] = indices[:max_len]
        else:
            array[i, :len(indices)] = indices


    return array, sizes,max_len


def load_parameters(dirname):
    """
    Load a dictionary containing the parameters used to train an instance
    of the autoencoder.

    :param dirname: the path to the directory with the model files.
    :return: a Python dictionary
    """
    filename = os.path.join(dirname, 'model-params.json')
    with open(filename, 'rb') as f:
        data = json.load(f)

    return data


def get_sentence_sizes(pairs):
    """
    Count the sizes of all sentences in the pairs
    :param pairs: a list of tuples (sent1, sent2, _). They must be
        tokenized
    :return: a tuple (sizes1, sizes2), as two numpy arrays
    """
    sizes1 = np.array([len(pair[0]) for pair in pairs])
    sizes2 = np.array([len(pair[1]) for pair in pairs])
    return (sizes1, sizes2)


def get_max_sentence_sizes(pairs1, pairs2):
    """
    Find the maximum length among the first and second sentences in both
    pairs1 and pairs2. The two lists of pairs could be the train and validation
    sets

    :return: a tuple (max_len_sentence1, max_len_sentence2)
    """
    train_sizes1, train_sizes2 = get_sentence_sizes(pairs1)
    valid_sizes1, valid_sizes2 = get_sentence_sizes(pairs2)
    train_max1 = max(train_sizes1)
    valid_max1 = max(valid_sizes1)
    max_size1 = max(train_max1, valid_max1)
    train_max2 = max(train_sizes2)
    valid_max2 = max(valid_sizes2)
    max_size2 = max(train_max2, valid_max2)

    return max_size1, max_size2


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def padding_attentionbias(attentionbias,sent1,sent2,wordpairweightdict,feathernum):

    padded_attentionbias = np.zeros([len(attentionbias),sent1.shape[1],sent2.shape[1],feathernum],dtype=float)
    attention_bias_attend = np.zeros([len(attentionbias),sent1.shape[1],sent2.shape[1]],dtype=float)
    for id,wordpair in enumerate(attentionbias):
        #print(len(wordpair))
        wordpair1 = wordpair[0]
        wordpair2 = wordpair[1]
        # 因为可能存在的不对称性，我们分开写，例如 w1 w2 存在，但是 w2 w1 不存在
        for wordindex in wordpair1:
            w1 = sent1[id][wordindex[0]]
            w2 = sent2[id][wordindex[1]]
            # print("index 1 %d , index 2 %d" %(wordindex[0],wordindex[1]))
            # print("word 1 %d , word 2 %d" %(w1,w2))
            padded_attentionbias[id][wordindex[0]][wordindex[1]] = wordpairweightdict[w1][w2]
            attention_bias_attend[id][wordindex[0]][wordindex[1]] = 1
            #if w1 in wordpairweightdict:
            #    if w2 in wordpairweightdict[w1]:
            #        #print(w1, w2)
            #        padded_attentionbias[id][wordindex[0]][wordindex[1]] = wordpairweightdict[w1][w2]
        for wordindex in wordpair2:
            w1 = sent1[id][wordindex[0]]
            w2 = sent2[id][wordindex[1]]
            # print("index 1 %d , index 2 %d" % (wordindex[0], wordindex[1]))
            # print("word 1 %d , word 2 %d" % (w1, w2))
            padded_attentionbias[id][wordindex[0]][wordindex[1]] = wordpairweightdict[w2][w1]
            attention_bias_attend[id][wordindex[0]][wordindex[1]] = 1
            #if w2 in wordpairweightdict:
            #    if w1 in wordpairweightdict[w2]:
            #        #print(w2, w1)
            #        padded_attentionbias[id][wordindex[0]][wordindex[1]] = wordpairweightdict[w2][w1]
    return padded_attentionbias,attention_bias_attend




def getCrossMatrix(sent1,sent2):
    cross_matrix = []
    for w1 in sent1:
        row_m = []
        for w2 in sent2:
            row_m.append(w1+';'+w2)
        cross_matrix.append(row_m)
    return cross_matrix
