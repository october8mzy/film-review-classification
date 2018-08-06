# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

class TextCNN(object):

	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.):

		# 输入、输出
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input-x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input-y')
		self.drop_out_keep = tf.placeholder(tf.float32, name='dropout_keep_prob')

		# 加入L2正则化的损失初始化
		l2_loss = tf.constant(0.)

		# embedding
		with tf.name_scope('embedding'):

			self.W = tf.get_variable('W', 
									[vocab_size, embedding_size], 
									initializer=tf.random_normal_initializer(srddev=0.5))
			# tf.embedding_lookup
			# tf.expand_dims
			self.embedded_chars_expanded = tf.expand_dims(tf.nn.embedding_lookup(self.W, self.input_x), -1)

			pool_outputs = []

			for filter_size in filter_sizes:
				with tf.name_scope('conv-maxpool-%s' % str(filter_size)):
					W = tf.get_variable('W', 
										[filter_size, embedding_size, 1, num_filters],
										initializer=tf.truncated_normal_initializer(stddev=0.1))
					b = tf.get_variable('b',
										[num_filters],
										initializer=tf.constant_initializer(0.1))
					conv = tf.nn.conv2d(self.embedded_chars_expanded,
										W,
										strides=[1, 1, 1, 1],
										padding='VALID',
										name='conv')
					h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
					pooled = tf.nn.max_pool(h,
											ksize=[1, vocab_size - filter_size + 1],
											strides=[1, 1, 1, 1],
											padding='VALID',
											name='pool'
											)
					pool_outputs.append(pooled)

			# 在最后一个通道进行合并操作，类似inception module
			num_filters_total = len(filter_sizes) * num_filters
			self.h_pool = tf.concat(pool_outputs, -1)
			self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

			with tf.name_scope('dropout'):

				self.h_drop = tf.nn.dropout(self.h_pool_flat, self.drop_out_keep)

			with tf.name_scope('output'):

				W = tf.get_variable('W',
									[num_filters_total, num_classes],
									initializer=tf.random_normal_initializer(stddev=0.1))
				b = tf.get_variable('b',
									[num_classes],
									initializer=tf.constant_initializer(0.1))
				
				l2_loss += tf.nn.l2_loss(W)
				# 不对b进行正则化了！没啥用
				# l2_loss += tf.nn.l2_loss(b)
				self.scores = tf.nn.xw_plus_b(self.h_drop,
											W,
											b,
											name='scores')
				self.predictions = tf.argmax(tf.scores, axis=1, name='predictions')

			# 求解损失函数
			with tf.name_scope('loss'):

				losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
				self.loss = tf.reduce_mean(losses) + 0.5 * l2_reg_lambda * l2_loss

			# 求解准确度
			with tf.name_scope('accuracy'):
				correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
				self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
				



