#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:57:27 2017

@author: jesusbm
"""
import tensorflow as tf;
from erreur_introduite import mnist, get_modified_data, get_modified_batch



x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#training

ycorr = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=ycorr, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  #batch_xs, batch_ys = mnist.train.next_batch(100)
  batch_xs, batch_ys = get_modified_batch(0.001,100)
  sess.run(train_step, feed_dict={x: batch_xs, ycorr: batch_ys})
  
#evaluate

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ycorr,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#xt=mnist.test.images
#yt=mnist.test.labels
xt,yt=get_modified_data(0.001,mnist.test)
print(sess.run(accuracy, feed_dict={x: xt[:1000], ycorr: yt[:1000]}))
