# coding=utf-8
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python import debug as tf_debug


def mlp_layer(x, hiddendim, name, activation=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = tf.reshape(x, [-1, x.get_shape()[-1]])
        n_inputs = int(x.get_shape()[-1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, hiddendim), stddev=stddev)
        # W = tf.Variable(tf.random_normal([n_inputs,hiddendim],stddev=0.1),name="weights")
        W = tf.get_variable(initializer=init, name="weights")
        # 偏置设成0 就可以了
        b = tf.get_variable(name="bias", initializer=tf.zeros([hiddendim]))
        z = tf.matmul(x, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        elif activation == "sigmoid":
            return tf.nn.sigmoid(z)
        elif activation == "tanh":
            return tf.nn.tanh(z)
        elif activation == "softmax":
            return tf.nn.softmax(z)
    return z


def softmax3d(x):
    original_shape = x.get_shape()
    x_reshape = tf.reshape(x, [-1, original_shape[2]])
    # c_reshape /= 100
    # cshape = c_reshape.get_shape()
    d = tf.nn.softmax(x_reshape)
    d = tf.reshape(d, [-1, original_shape[1], original_shape[2]])
    return d


class DAM(object):
    def __init__(self, config):
        self.max_len_sen1 = config["max_len_1"]
        self.max_len_sen2 = config["max_len_2"]
        self.n_class = config["n_class"]
        self.hiddendim = config["hiddendim"]
        self.vocabsize = config["vocabsize"]
        self.embeddim = config["embeddim"]
        self.dropoutrate = config["dropoutrate"]
        self.learning_rate = config["learning_rate"]
        self.clip_value = config["clip_value"]
        # inputdata
        self.sent1 = tf.placeholder(tf.int32, [None, self.max_len_sen1])
        self.sent2 = tf.placeholder(tf.int32, [None, self.max_len_sen2])
        self.labels = tf.placeholder(tf.int32, [None])
        self.embedding_pretrain = tf.placeholder(tf.float32, [self.vocabsize, self.embeddim])

    def build_graph(self):
        with tf.name_scope("embedding") as scope:
            embeddings_var = tf.Variable(tf.random_uniform([self.vocabsize, self.embeddim], -1, 1), trainable=False)
            embeddings_var.assign(self.embedding_pretrain)
            embedd_1 = tf.nn.embedding_lookup(embeddings_var, self.sent1)
            embedd_2 = tf.nn.embedding_lookup(embeddings_var, self.sent2)
        # project
        with tf.name_scope("projection")as scope:
            orginalshape_1 = embedd_1.get_shape()
            orginalshape_2 = embedd_2.get_shape()
            embedd_1 = tf.reshape(embedd_1, [-1, self.embeddim])
            embedd_2 = tf.reshape(embedd_2, [-1, self.embeddim])
            embedd_project_1 = mlp_layer(embedd_1, self.embeddim, name="project", activation="relu")
            embedd_project_2 = mlp_layer(embedd_2, self.embeddim, name="project", activation="relu", reuse=True)
            embedd_project_1 = tf.reshape(embedd_project_1, [-1, orginalshape_1[1], orginalshape_1[2]])
            embedd_project_2 = tf.reshape(embedd_project_2, [-1, orginalshape_2[1], orginalshape_2[2]])
        # Attend
        with tf.name_scope("Attend") as scope:
            self.attend = tf.matmul(embedd_project_1, tf.transpose(embedd_project_2, [0, 2, 1]))
            self.attend1_softmax = softmax3d(self.attend)
            self.attend2_softmax = softmax3d(tf.transpose(self.attend, [0, 2, 1]))
            self.alpha = tf.matmul(self.attend1_softmax, embedd_project_2)
            self.beta = tf.matmul(self.attend2_softmax, embedd_project_1)
        # compare
        with tf.name_scope("Compare") as scope:
            v1 = tf.concat([embedd_project_1, self.alpha, embedd_project_1 - self.alpha, embedd_project_1 * self.alpha],
                           axis=-1)
            v2 = tf.concat([embedd_project_2, self.beta, embedd_project_2 - self.beta, embedd_project_2 * self.beta],
                           axis=-1)
            # G
            v1 = tf.nn.dropout(v1, self.dropoutrate)
            v2 = tf.nn.dropout(v2, self.dropoutrate)
            v1_g = mlp_layer(v1, 4 * self.embeddim, name="g_function", activation="relu")
            v2_g = mlp_layer(v2, 4 * self.embeddim, name="g_function", activation="relu", reuse=True)
            v1_g = tf.reshape(v1_g, [-1, self.max_len_sen1, 4 * self.embeddim])
            v2_g = tf.reshape(v2_g, [-1, self.max_len_sen2, 4 * self.embeddim])
        # aggregate
        with tf.name_scope("aggregate"):
            v1_g = tf.reduce_sum(v1_g, axis=1)
            v2_g = tf.reduce_sum(v2_g, axis=1)
            v_all = tf.concat([v1_g, v2_g], axis=-1)
        # classify
        with tf.name_scope("classify"):
            v_all = tf.nn.dropout(v_all, self.dropoutrate)
            v_all = mlp_layer(v_all, self.embeddim, name="h_out_1", activation="relu")
            self.output = mlp_layer(v_all, self.n_class, name="h_out_2", activation="")
        # loss
        # 之前的损失是每个的损失，这里要取均值
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
            # 优化器
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # clip TO DO
            # self.training_op = self.optimizer.minimize(self.loss)
            gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.clip_value)
            self.training_op = self.optimizer.apply_gradients(zip(gradients, v))
        # 预测
        with tf.name_scope("prediction"):
            self.preprob = tf.nn.softmax(self.output)
            self.prediction = tf.argmax(self.preprob, axis=1)


def fetch_batch(train_X, train_y, batch_index, batch_size):
    return train_X[batch_index * batch_size:(batch_index + 1) * batch_size], \
           train_y[batch_index * batch_size:(batch_index + 1) * batch_size]


def shuffledata(train_X, train_y):
    index = [i for i in range(len(train_X))]
    np.random.shuffle(index)
    # return [item  ]
    return train_X[index], train_y[index]


def getFeedDict(x, y):
    sen1 = np.stack([item[0] for item in x])
    sen2 = np.stack([item[1] for item in x])
    return {dam_model.sent1: sen1, dam_model.sent2: sen2, dam_model.labels: y,
            dam_model.embedding_pretrain: embeedingmatrix}

def load_glove_vector(filename,worddict):
    model = {}
    global embedd_dim
    with open(filename) as f:
        for line in f:
            tmp = line.strip().split(' ')
            word = tmp[0]
            emstr = tmp[1:]
            if word not in worddict:
                continue
            embedd_dim = len(emstr)
            model[word] = np.array([float(item) for item in emstr],dtype='float32')
    model["_UNK_"] = np.random.randn(embedd_dim,)
    model["_PAD_"] = np.random.randn(embedd_dim,)
    return model
def getEmbeedingMatrix(word2vec_model,worddict_inverse):
    trans_matrix = np.zeros([len(worddict),embedd_dim])
    for i in range(len(worddict_inverse)):
        if worddict_inverse[i] in word2vec_model:
            trans_matrix[i] = word2vec_model[worddict_inverse[i]]
        else:
            trans_matrix[i] = word2vec_model["_UNK_"]
    return trans_matrix


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
train_file_path = "/home/hwwang/workplace/deeplearning/textentailment/data/snli/train_sen1_sen2.txt"
traindata = pd.read_csv(train_file_path, sep='\t')
sent_1 = list(traindata["sen1"])
sent_2 = list(traindata["sen2"])
sent_1 = [str(item) for item in sent_1]
sent_2 = [str(item) for item in sent_2]
labels = list(traindata["label"])
sent_1 = sent_1[:5000]
sent_2 = sent_2[:5000]
labels = labels[:5000]
# traindata.columns
textall = list(sent_1)
textall.extend(sent_2)
print(len(textall))

# def getVocabulary(texts)
max_feather = 60000
embedd_dim = 300
tokenize = Tokenizer(num_words=max_feather)
tokenize.fit_on_texts(textall)
worddict = tokenize.word_index
sent1_seq = tokenize.texts_to_sequences(sent_1)
sent2_seq = tokenize.texts_to_sequences(sent_2)
max_len_1 = max([len(item) for item in sent1_seq])
max_len_2 = max([len(item) for item in sent2_seq])
print(max_len_1, max_len_2)
sent1_pad = pad_sequences(sent1_seq, maxlen=max_len_1, padding="post")
sent2_pad = pad_sequences(sent2_seq, maxlen=max_len_2, padding="post")
labelencoder = LabelEncoder()
labelencoder.fit(labels)
y_labes = labelencoder.transform(labels)
labels_dict = labelencoder.classes_
worddict['_PAD_'] = 0
worddict['_UNK_'] = len(worddict)
worddict_inverse = {}
for key, value in worddict.items():
    worddict_inverse[value] = key
# make Embeddings
vector_path = "/home/hwwang/workplace/deeplearning/resource/glove.840B.300d.txt"
word2vec_model = load_glove_vector(vector_path, worddict)
embeedingmatrix = getEmbeedingMatrix(word2vec_model, worddict_inverse)
x_all = []
for sen1, sen2 in zip(sent1_pad, sent2_pad):
    x_all.append((sen1, sen2))
x_train, x_test, y_train, y_test = train_test_split(x_all, y_labes, test_size=0.2)
config = {
    "hiddendim": 200,
    "vocabsize": len(worddict),
    "embeddim": embedd_dim,
    "n_class": len(labels_dict),
    "learning_rate": 0.001,
    "dropoutrate": 0.2,
    "max_len_1": max_len_1,
    "max_len_2": max_len_2,
    "clip_value": 10
}
epochs = 10
batch_size = 64
tf.reset_default_graph()
dam_model = DAM(config)
dam_model.build_graph()
init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('loss', dam_model.loss)
file_writer = tf.summary.FileWriter("./log/", tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    for epoch in range(epochs):
        # x_train,y_train = shuffledata(x_train,y_train)
        n_batches = int(np.ceil(len(x_train) / batch_size))
        print("epoch :  ", epoch)
        for b_idx in range(n_batches):
            x_batch, y_batch = fetch_batch(x_train, y_train, b_idx, batch_size)
            # print(x_batch)
            # print(y_batch)
            sess.run(dam_model.training_op, feed_dict=getFeedDict(x_batch, y_batch))
            # summary_str = loss_summary.eval(feed_dict=getFeedDict(x_batch,y_batch))
            step = epoch * n_batches + b_idx
            # file_writer.add_summary(summary_str,step)
            if b_idx % 100 == 0:
                print("idx  ", b_idx, ":", sess.run(dam_model.loss, feed_dict=getFeedDict(x_batch, y_batch)))
        # print(sess.run(my_dnn.prediction,feed_dict={my_dnn.X:x_test,my_dnn.labels:y_test}))
        # print(sess.run(my_dnn.loss,feed_dict={my_dnn.X:x_test,my_dnn.labels:y_test}))
        # a.append(sess.run(my_dnn.output,feed_dict={my_dnn.X:x_test,my_dnn.labels:y_test}))
        acc_test = accuracy_score(y_test, sess.run(dam_model.prediction, feed_dict=getFeedDict(x_test, y_test)))
        acc_train = accuracy_score(y_train, sess.run(dam_model.prediction, feed_dict=getFeedDict(x_train, y_train)))
        print("acc_train:", acc_train)
        print("acc_test:", acc_test)
