# -*- coding: utf-8 -*-

"""
Functions for reading and writing data to and from files.
"""

import json
import os
import logging
import numpy as np
import nltk
import codecs
import pickle as  pkl
from collections import defaultdict

import utils


def write_word_dict(word_dict, dirname):
    """
    Write the word dictionary to a file.

    It is understood that unknown words are mapped to 0.
    """
    #words = [word for word in word_dict.keys() if word_dict[word] != 0]
    words = [word for word in word_dict.keys()]
    sorted_words = sorted(words, key=lambda x: word_dict[x])
    #text = '\n'.join(sorted_words)
    path = os.path.join(dirname, 'word-dict.txt')
    with open(path, 'wb') as f:
        for wd in sorted_words:
            f.write((wd+'\t'+str(word_dict[wd])+'\n').encode('utf-8'))
        #f.write(text.encode('utf-8'))


def read_word_dict(dirname):
    """
    Read a file with a list of words and generate a defaultdict from it.
    """
    filename = os.path.join(dirname, 'word-dict.txt')
    with open(filename, 'rb') as f:
        text = f.read().decode('utf-8')

    words = text.splitlines()
    index_range = range(1, len(words) + 1)
    return defaultdict(int, zip(words, index_range))

def read_word_dict_text(filename):
    """
    Read a file with a list of words and generate a defaultdict from it.
    """
    worddict = {}
    with codecs.open(filename, 'r','utf-8') as f:
        for line in f:
            tmp = line.strip().split()
            #print(tmp[0])
            if len(tmp)==2:
                if tmp[0] not in worddict:
                    worddict[tmp[0]] = int(tmp[1])
            elif len(tmp) == 1:
                worddict[tmp[0]] = len(worddict)

    return worddict


def write_extra_embeddings(embeddings, dirname):
    """
    Write the extra embeddings (for unknown, padding and null)
    to a numpy file. They are assumed to be the first three in
    the embeddings model.
    """
    path = os.path.join(dirname, 'extra-embeddings.npy')
    np.save(path, embeddings[:3])


def _generate_random_vector(size):
        """
        Generate a random vector from a uniform distribution between
        -0.1 and 0.1.
        """
        return np.random.uniform(-0.1, 0.1, size)

def load_pairs_text(filename, lowercase, language='en'):
    """
    Read sent1 \t sent2 \t label

    :param filename: path to the file
    :param lowercase: whether to convert content to lower case
    :param language: language to use tokenizer (only used if input is in
        TSV format)
    :return: a list of tuples (first_sent, second_sent, label)
    """
    logging.info('Reading data from %s' % filename)
    # we are only interested in the actual sentences + gold label
    # the corpus files has a few more things
    useful_data = []
    # the SNLI corpus has one JSON object per line
    with codecs.open(filename, 'r','utf-8') as f:
        tokenize = utils.get_tokenizer(language)
        for line in f:
            line = line.strip()
            if lowercase:
                line = line.lower()
            sent1, sent2, label = line.split('\t')
            if label == '-':
                continue
            tokens1 = tokenize(sent1)
            tokens2 = tokenize(sent2)
            tokens1 = ['_BOS_'] + tokens1
            tokens2 = ['_BOS_'] + tokens2
            useful_data.append((tokens1, tokens2, label))
    return useful_data

def load_embeddings(embeddings_path, vocabulary_path=None,
                    generate=False, load_extra_from=None,
                    normalize=False):
    """
    Load and return an embedding model in either text format or
    numpy binary format. The text format is used if vocabulary_path
    is None (because the vocabulary is in the same file as the
    embeddings).

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with vocabulary,
        if needed
    :param generate: whether to generate random embeddings for
        unknown, padding and null
    :param load_extra_from: path to directory with embeddings
        file with vectors for unknown, padding and null
    :param normalize: whether to normalize embeddings
    :return: a tuple (defaultdict, array)
    """
    assert not (generate and load_extra_from), \
        'Either load or generate extra vectors'

    logging.debug('Loading embeddings')
    if embeddings_path is None:
        wd = read_word_dict_text(vocabulary_path)
        embeddings = _generate_random_vector((len(wd),300))
    
    else:
        if vocabulary_path is None:
            wordlist, embeddings = load_text_embeddings(embeddings_path)
        else:
            wd, embeddings = load_binary_embeddings(embeddings_path,
                                                        vocabulary_path)

    if generate or load_extra_from:
        #mapping = zip(wordlist, range(3, len(wordlist) + 3))
        #print("1111111111111111111111111111111111111111")
        # always map OOV words to 0
        #wd = defaultdict(int, mapping)
        #wd[utils.PADDING] = 0
        #wd[utils.UNKNOWN] = 1
        #wd[utils.GO] = 2

        if generate:
            vector_size = embeddings.shape[1]
            extra = [_generate_random_vector(vector_size),
                     _generate_random_vector(vector_size),
                     _generate_random_vector(vector_size)]

        else:
            path = os.path.join(load_extra_from, 'extra-embeddings.npy')
            extra = np.load(path)

        embeddings = np.append(extra, embeddings, 0)

    else:
        #mapping = zip(wordlist, range(0, len(wordlist)))
        #wd = defaultdict(int, mapping)
        #wd = wordlist
        pass

    logging.debug('Embeddings have shape {}'.format(embeddings.shape))
    if normalize:
        embeddings = utils.normalize_embeddings(embeddings)

    return wd, embeddings


def load_binary_embeddings(embeddings_path, vocabulary_path):
    """
    Load any embedding model in numpy format, and a corresponding
    vocabulary with one word per line.

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with words
    :return: a tuple (wordlist, array)
    """
    vectors = np.load(embeddings_path)

    # 这里修改了文件导入的方式，这里的 worddict 使用的是
    #with open(vocabulary_path, 'rb') as f:
    #    text = f.read().decode('utf-8')
    #words = text.splitlines()
    wd = read_word_dict_text(vocabulary_path)
    #with open(vocabulary_path,'r') as f:
    #    wd = pkl.load(f)

    return wd, vectors

def write_pairs(sentence_pairs,path):
    """
    write to save sentence_pair
    """
    with open(path,"w") as f:
        for pair in sentence_pairs:
            result = []
            result.append(" ".join(pair[0]))
            result.append(" ".join(pair[1]))
            result.append(pair[2])
            r1 = "\t".join(result)+'\n'
            f.write(r1.encode('utf-8'))



def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            words.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)

    embeddings = np.array(vectors, dtype=np.float32)

    return words, embeddings

def load_wordnetweight(path):
    """
    输出格式: word1;word2 0 0 0 0
    :param path:
    :return:wordnetweight_dict,feathernum
    """
    with open(path,'r') as f:
        wordpairweights = pkl.load(f)
    return wordpairweights


def write_params(dirname, lowercase, language=None, model='mlp'):
    """
    Write system parameters (not related to the networks) to a file.
    """
    path = os.path.join(dirname, 'system-params.json')
    data = {'lowercase': lowercase,
            'model': model}
    if language:
        data['language'] = language
    with open(path, 'wb') as f:
        json.dump(data, f)


def write_label_dict(label_dict, dirname):
    """
    Save the label dictionary to the save directory.
    """
    path = os.path.join(dirname,'label-map.json')
    with open(path, 'wb') as f:
        json.dump(label_dict, f)


def load_label_dict(dirname):
    """
    Load the label dict saved with a model
    """
    path = os.path.join(dirname,'label-map.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_params(dirname):
    """
    Load system parameters (not related to the networks)
    :return: a dictionary
    """
    path = os.path.join(dirname, 'system-params.json')
    with open(path, 'rb') as f:
        return json.load(f)


def read_alignment(filename, lowercase):
    """
    Read a file containing pairs of sentences and their alignments.
    :param filename: a JSONL file
    :param lowercase: whether to convert words to lowercase
    :return: a list of tuples (first_sent, second_sent, alignments)
    """
    sentences = []
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            if lowercase:
                line = line.lower()
            data = json.loads(line)
            sent1 = data['sentence1']
            sent2 = data['sentence2']
            alignment = data['alignment']
            sentences.append((sent1, sent2, alignment))

    return sentences




def read_corpus(filename, lowercase, language='en'):
    """
    Read a JSONL or TSV file with the SNLI corpus

    :param filename: path to the file
    :param lowercase: whether to convert content to lower case
    :param language: language to use tokenizer (only used if input is in
        TSV format)
    :return: a list of tuples (first_sent, second_sent, label)
    """
    logging.info('Reading data from %s' % filename)
    # we are only interested in the actual sentences + gold label
    # the corpus files has a few more things
    useful_data = []
    wordpairs = []
    # the SNLI corpus has one JSON object per line
    with open(filename, 'rb') as f:
        if filename.endswith('.tsv') or filename.endswith('.txt'):
            #tokenize = utils.get_tokenizer(language)
            for line in f:
                line = line.decode('utf-8').strip()
                pair1 = []
                pair2 = []
                if lowercase:
                    line = line.lower()
                tmp = line.split('\t')
                if len(tmp)<3:
                    continue
                sent1, sent2, label = tmp[:3]
                #这里是 sent1 --> sent2 中的 wordpair
                #这里 wordpair 是用 index 做替换的，这个index 是该词在sent 中的位置
                #这里要对原始语料进行 tokenize

                # KIM wordpair 格式
                if len(tmp)>=4:
                    pair_item  = tmp[3].split(' ')
                    for pair in pair_item:
                        if pair.find('|')!=-1:
                            pair_split = pair.split('|')
                            if len(pair_split)==2:
                                pair1.append((int(pair_split[0]),int(pair_split[1])))
                if len(tmp)>=5:
                    pair_item  = tmp[4].split(' ')
                    for pair in pair_item:
                        if pair.find('|')!=-1:
                            pair_split = pair.split('|')
                            if len(pair_split)==2:
                                pair2.append((int(pair_split[0]),int(pair_split[1])))

                wordpairs.append((pair1,pair2))
                if label == '-':
                    continue
                #tokens1 = tokenize(sent1)
                #tokens2 = tokenize(sent2)
                tokens1 = sent1.split(' ')
                tokens2 = sent2.split(' ')
                useful_data.append([tokens1, tokens2, label])
        else:
            for line in f:
                line = line.decode('utf-8')
                if lowercase:
                    line = line.lower()
                data = json.loads(line)
                if data['gold_label'] == '-':
                    # ignore items without a gold label
                    continue

                sentence1_parse = data['sentence1_parse']
                sentence2_parse = data['sentence2_parse']
                label = data['gold_label']

                tree1 = nltk.Tree.fromstring(sentence1_parse)
                tree2 = nltk.Tree.fromstring(sentence2_parse)
                tokens1 = tree1.leaves()
                tokens2 = tree2.leaves()
                t = (tokens1, tokens2, label)
                useful_data.append(t)

    return useful_data,wordpairs



def getWordpairDict(filename,outputfile):
    wordpair_dict = {}
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            pair1 = []
            pair2 = []
            tmp = line.split('\t')
            if len(tmp)<3:
                continue
            sent1, sent2, label = tmp[:3]
            #这里是 sent1 --> sent2 中的 wordpair
            #这里 wordpair 是用 index 做替换的，这个index 是该词在sent 中的位置
            #这里要对原始语料进行 tokeni
            tokens1 = sent1.split(' ')
            tokens2 = sent2.split(' ')
            for w1 in tokens1:
                for w2 in tokens2:
                    wordpair_dict[w1+';'+w2] = len(wordpair_dict)
    with open(outputfile, 'wb') as f:
        for wd in wordpair_dict:
            f.write((wd+'\t'+str(wordpair_dict[wd])+'\n').encode('utf-8'))
    return wordpair_dict

