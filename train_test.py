# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
import datatime
import data_helpers
from text_cnn import TextCNN

from tensorflow.contrib import learn

positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"   # 好评txt
negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"   # 差评txt

dev_sample_percentage = 0.1                                     # 原文10fold，这个数据太少，10%作为测试集，容易过拟合
embedding_dim = 128                                             # 词向量长度
filter_sizes = ['3', '4', '5']                                  # conv滤波器高度
num_filters = 128                                               # 每个conv滤波器的个数
dropout_keep_prob = 0.5                                         # FC层的drop_out
l2_reg_lambda = 0.                                              # FC层的l2_lambda

batch_size = 64
num_epochs = 200

allow_soft_placement = True                                     # 有gpu用gpu，没有就用cpu
log_device_placement = False                                    # 不打印日志

def preprocess():

    x_text, y_test = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)

    # 字典单词个数，并以此构建每个单词identity card
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # 划分训、测试集
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    del x, y, x_shuffled, y_shuffled

    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):

                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.drop_out_keep: dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.drop_out_keep: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # mini_batch
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                # 每100次使用以此验证集
                if current_step % 100 == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
