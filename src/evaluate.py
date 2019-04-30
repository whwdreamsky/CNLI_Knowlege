# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
from itertools import izip
import tensorflow as tf
import numpy as np
import utils
import ioutils
import os
from sklearn.metrics import classification_report

"""
Evaluate the performance of an NLI model on a dataset
"""


def print_errors(pairs, answers, label_dict):
    """
    Print the pairs for which the model gave a wrong answer,
    their gold label and the system one.
    """
    for pair, answer in izip(pairs, answers):
        label_str = pair[2]
        label_number = label_dict[label_str]
        if answer != label_number:
            sent1 = ' '.join(pair[0])
            sent2 = ' '.join(pair[1])
            print('Sent 1: {}\nSent 2: {}'.format(sent1, sent2))
            print('System label: {}, gold label: {}'.format(answer,
                                                            label_number))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # data_base = "/home/hwwang/workplace/deeplearning/textentailment/data/kim_wordnet_data/"
    # data_base1 = "/home/hwwang/workplace/project/nli/data/word_sequence/"
    # root_base = "/home/hwwang/workplace/project/DAM/"
    #data_base = "/home/oliver/Documents/workplace/project/data/breaknli/"
    #data_base1 = "/home/oliver/Documents/workplace/project/data/snli/snli/"
    data_base = "/home/oliver/Documents/workplace/project/data/cnli_back/blunlp/"
    root_base = "/home/oliver/Documents/workplace/project/DAM/"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Directory with saved model',default=root_base+'/model_weights/cnli_mlp_charcnn/')
    parser.add_argument('--dataset',
                        help='JSONL or TSV file with data to evaluate on',default=data_base+'cnli_dev_seg_ltp.txt')
    parser.add_argument('--dataset_lemma',
                        help='JSONL or TSV file with data to evaluate on', default=data_base + 'cnli_dev_seg_ltp.txt')
    parser.add_argument('--embeddings', help='Numpy embeddings file',default=data_base+'/char_embed/cnli_tencent.npy')

    parser.add_argument('--vocabulary',
                        help='Text file with embeddings vocabulary',default=data_base+'/char_embed/worddict.txt')
    parser.add_argument('-v',
                        help='Verbose', action='store_true', dest='verbose')
    parser.add_argument('-e',
                        help='Print pairs and labels that got a wrong answer',
                        action='store_true', dest='errors')
    args = parser.parse_args()

    utils.config_logger(verbose=args.verbose)
    params = ioutils.load_params(args.model)
    sess = tf.Session()

    model_class = utils.get_model_class(params)
    model = model_class.load(args.model, sess)

    # word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
    #                                                args.vocabulary,
    #                                                generate=False,
    #                                                load_extra_from=args.model,
    #                                                normalize=True)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
                                                    args.vocabulary,
                                                    generate=False,
                                                    load_extra_from=None,
                                                    normalize=False)

    model.initialize_embeddings(sess, embeddings)
    label_dict = ioutils.load_label_dict(args.model)

    pairs,wordpairs = ioutils.read_corpus(args.dataset, True,
                                params['language'])
    dataset,_,_ = utils.create_dataset(
                                pairs,wordpairs,
                                word_dict,
                                label_dict,
                                max_len1=model.maxlen1,
                                max_len2=model.maxlen2)

    print("Test Dataset Size :%d" ,dataset.num_items)
    loss, acc, answers,logits = model.evaluate(sess, dataset, True,batch_size=64)
    #print(answers)
    print(np.array(logits).shape)
    label_dict_inverse = {}
    for key,value in label_dict.items():
        label_dict_inverse[value] = key
    fout = open("t1.txt",'w')
    for ans in answers:
        fout.write(label_dict_inverse[ans])
        fout.write('\n')
    fout.close()
        #print(label_dict_inverse[ans])
    #print(logits)
    #logits = np.array(logits)
    #print(logits.shape)
    #np.savetxt("./logits.txt",logits)
    #for item in logits:
    #    print(item)
    print('Loss: %f' % loss)
    print('Accuracy: %f' % acc)
    target_names = []
    for i in range(len(label_dict_inverse)):
        target_names.append(label_dict_inverse[i])
    print(classification_report(dataset.labels,answers,target_names=target_names))

    if args.errors:
        print_errors(pairs, answers, label_dict)
