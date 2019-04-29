import tensorflow as tf
import codecs
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
# 读入数据
database = "/home/oliver/Documents/workplace/project/data/snli/snli/"
vocabfile = database + "vocab.txt"
embedfile = database + "snli.npy"
trainfile = database + "t1.txt"
def getSNLIData(datafile):
    sent1 = []
    sent2 = []
    labels = []
    with codecs.open(datafile,'r',encoding="utf-8") as f:
        for line in f:
            tmp= line.strip().split('\t')
            if len(tmp)<2:
                continue
            s1 = tmp[0]
            s2 = tmp[1]
            sent1.append(s1.split(' '))
            sent2.append(s2.split(' '))
            labels.append(tmp[2])
    return sent1,sent2,labels
def read_word_dict_text(filename):
    """
    Read a file with a list of words and generate a defaultdict from it.
    """
    worddict = {}
    with open(filename) as f:
        for line in f:
            tmp = line.strip().split()
            if len(tmp)==2:
                if tmp[0] not in worddict:
                    worddict[tmp[0]] = int(tmp[1])
            elif len(tmp) == 1:
                worddict[tmp[0]] = len(worddict)

    return worddict
worddict = read_word_dict_text(vocabfile)
sent1,sent2,labels= getSNLIData(trainfile)
def convert_and_pad(sents):
    maxlen = max([len(sent) for sent in sents])
    sents_id = np.full((len(sents),maxlen),worddict["_PAD_"])
    for i,sent in enumerate(sents):
        for j,w in enumerate(sents[i]):
            if w in worddict:
                sents_id[i][j] = worddict[w]
            else:
                sents_id[i][j] = worddict["_UNK_"]
    return sents_id
sent1_paded = convert_and_pad(sent1)
sent2_paded = convert_and_pad(sent2)
le = LabelEncoder()
lables_id = le.fit_transform(labels)
print(le.classes_)
def train_input_fn():
    def parser(sent1,sent2,label):
        features = {"sent1:"sent1,"sent2":sent2,"label":label}
        return features,label

    dataset = tf.data.Dataset.from_tensor_slices((
                        sent1_paded,
                        sent2_paded,
                        lables_id))

    dataset = dataset.shuffle()
    dataset = dataset.batch(32)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((
                        sent1_paded,
                        sent2_paded,
                        lables_id))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


head = tf.contrib.estimator.multi_class_head(3)
def model_fn(features,labels,mode,params):
    def cnn_encoder(inputs,reuse=False,training=True):
        inputs = tf.expand_dims(inputs,axis=-1)
        with tf.variable_scope("cnn_encoder",reuse=reuse) \
            as cnnscope:
            conv = tf.layers.conv1d(inputs=inputs,
                              reuse=reuse,
                              filters=32,
                              kernel_size=[2,3,4],
                              padding="same",
                              activation=tf.nn.relu)
            pool = tf.reduce_max(input_tensor=conv,axis=-1)
            dropout_pool = tf.layers.dropout(inputs=pool,
                                               rate=0.2,training=training)
        return dropout_pool

    def lstm_encoder(inputs,reuse=True,training=True):
        initializer = tf.contrib.layers.xavier_initializer()
        lstm = tf.nn.rnn_cell.LSTMCell(300,initializer=initializer)
        with tf.variable_scope("lstm_encoder",reuse=reuse) \
            as lstmscope:
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm,lstm,
                                                    inputs,
                                                    dtype=tf.float32,
                                                    scope=lstmscope)
            output_fw,output_bw  = outputs
            concate_output = tf.concat([outputfw,output_bw],axis=-1)
        return concate_output
    

    embeddings = tf.Variable(params["embedding_initializer"],trainable=False)
    sent1_embed = tf.nn.embedding_lookup(embeddings,features['sent1'])
    sent2_embed = tf.nn.embedding_lookup(embeddings,features['sent2'])
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    sent1_embed_drop = tf.layers.dropout(inputs=sent1_embed,\
                                        rate=0.2,
                                        training=training)
    sent2_embed_drop = tf.layers.dropout(inputs=sent2_embed,
                                         rate=0.2,
                                         training=training)
    sent1_cnn = cnn_encoder(sent1_embed_drop,False,training)
    sent2_cnn = cnn_encoder(sent2_embed_drop,True,training)
    sent1_lstm = lstm_encoder(sent1_embed_drop,False,training)
    sent2_lstm = lstm_encoder(sent2_embed_drop,True,training)
    concat_embed = tf.concate([sent1_cnn,sent1_embed,sent2_cnn,sent2_embed])
    logits = tf.layers.dense(concat_embed,3,activation=tf.nn.relu)

    if labels is not None:
        labels = tf.reshape(labels,[-1,1])
    optimizer = tf.train.AdamOptimizer()
    def _train_op_fn(loss):
        return optimizer.minimize(loss,tf.train.get_global_step())
    
    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits, 
        train_op_fn=_train_op_fn)
    