# -*- coding: utf-8 -*-

from __future__ import division, print_function

"""
Script to train an RTE LSTM.
"""

import sys
import argparse
import tensorflow as tf
import os
import ioutils
import utils
from classifiers import LSTMClassifier, MultiFeedForwardClassifier,\
    DecomposableNLIModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':

    #data_base = "/data/ceph/offline_training/hwwang/experiment/multiffn-nli-master/snli/snli_1.0/"
    data_base = "/home/hwwang/workplace/deeplearning/textentailment/data/snli/"
    root_base = "/home/hwwang/workplace/deeplearning/textentailment/multiffn-nli-master/"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings',
                        help='Text or numpy file with word embeddings',
                        default=data_base+'snli.npy')
    parser.add_argument('--train', help='JSONL or TSV file with training corpus',default=data_base+'t1.txt')#'train_s1_s2_label.txt')#'cnli_train_beta1_seg.txt')
    parser.add_argument('--validation',
                        help='JSONL or TSV file with validation corpus',default=data_base+'t2.txt')#'dev_s1_s2_label.txt')
    parser.add_argument('--save', help='Directory to save the model files',default=root_base+"/model_weights/")
    parser.add_argument('--model', help='Type of architecture',
                        choices=['lstm', 'mlp'],default='mlp')
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)',default=data_base+'vocab.txt')
    parser.add_argument('-e', dest='num_epochs', default=30, type=int,
                        help='Number of epochs')
    parser.add_argument('-b', dest='batch_size', default=32, help='Batch size',
                        type=int)
    parser.add_argument('-u', dest='num_units', help='Number of hidden units',
                        default=300, type=int)
    parser.add_argument('--no-proj', help='Do not project input embeddings to '
                                          'the same dimensionality used by '
                                          'internal networks',
                        action='store_false', dest='no_project')
    parser.add_argument('-d', dest='dropout', help='Dropout keep probability',
                        default=0.8, type=float)
    parser.add_argument('-aw', dest='attendweight', help='attention bias weight',
                        default=1, type=float)
    parser.add_argument('-c', dest='clip_norm', help='Norm to clip training '
                                                     'gradients',
                        default=10, type=float)
    parser.add_argument('-r', help='Learning rate', type=float, default=0.05,
                        dest='rate')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en',
                        help='Language (default en; only affects tokenizer)')
    parser.add_argument('--lower', help='Lowercase the corpus (use it if the '
                                        'embedding model is lowercased)',default=True,
                        action='store_true')
    parser.add_argument('--use-intra', help='Use intra-sentence attention',
                        action='store_true', dest='use_intra')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('--report', help='Number of batches between '
                                         'performance reports',
                        default=100, type=int)
    parser.add_argument('-v', help='Verbose', action='store_true',
                        dest='verbose',default=1)
    parser.add_argument('--optim', help='Optimizer algorithm',
                        default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])

    args = parser.parse_args()
    
    utils.config_logger(args.verbose)
    logger = utils.get_logger('train')
    logger.debug('Training with following options: %s' % ' '.join(sys.argv))

    train_pairs,train_wordpairs = ioutils.read_corpus(args.train, args.lower, args.lang)
    valid_pairs,valid_wordpairs = ioutils.read_corpus(args.validation, args.lower, args.lang)

    #ioutils.write_pairs(train_pairs,args.save+'//train.txt')
    #ioutils.write_pairs(valid_pairs,args.save+'//valid.txt')
    # whether to generate embeddings for unknown, padding, null

    
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                    True, normalize=True)

    logger.info('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)
    train_data = utils.create_dataset(train_pairs,train_wordpairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs,valid_wordpairs, word_dict, label_dict)
   
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    ioutils.write_params(args.save, lowercase=args.lower, language=args.lang,
                         model=args.model)
    ioutils.write_label_dict(label_dict, args.save)
    ioutils.write_extra_embeddings(embeddings, args.save)

    msg = '{} sentences have shape {} (firsts) and {} (seconds)'
    logger.debug(msg.format('Training',
                            train_data.sentences1.shape,
                            train_data.sentences2.shape))
    logger.debug(msg.format('Validation',
                            valid_data.sentences1.shape,
                            valid_data.sentences2.shape))

    
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    logger.info('Creating model')
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    if args.model == 'mlp':
        model = MultiFeedForwardClassifier(args.num_units, 3, vocab_size,
                                           embedding_size,
                                           use_intra_attention=args.use_intra,
                                           training=True,
                                           project_input=args.no_project,
                                           optimizer=args.optim)
    else:
        model = LSTMClassifier(args.num_units, 3, vocab_size,
                               embedding_size, training=True,
                               project_input=args.no_project,
                               optimizer=args.optim)

    model.initialize(sess, embeddings)

    # this assertion is just for type hinting for the IDE
    assert isinstance(model, DecomposableNLIModel)

    total_params = utils.count_parameters()
    logger.debug('Total parameters: %d' % total_params)

    logger.info('Starting training')
    model.train(sess, train_data, valid_data, args.save, args.rate,
                args.num_epochs, args.batch_size, args.dropout, args.l2,
                args.clip_norm, args.report,args.attendweight)

