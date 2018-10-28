#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:19:13 2017

@author: jesusbm
"""
import numpy as np
import time
import tensorflow as tf

class Cnn(object):
    
    def __init__(self):
        pass
        
    #fonction pour introduire des erreurs sur une image
    def pixel_bit_error(self,pe, image, nbits):
        N = len(image)
        err_image = np.zeros_like(image)
        pixel_error = np.random.choice([0, 1], N, p=[1-pe, pe])
        for i in range(N):
            if pixel_error[i]:
                bit = np.random.randint(0, nbits)
                nrm = 2**nbits - 1.0
                err_image[i] = (int(image[i]*nrm) ^ (2**bit))/nrm
            else:
                err_image[i] = image[i]
        return err_image

    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
    def conv2d(self,x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self,x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
      

    def run(self, train_data, test_data, resolution, error, generations, batchSize):

        train_images=train_data
        test_images=test_data
        
        
        train_labels=np.load("./data/train_labels.npy")
        test_labels=np.load("./data/test_labels.npy")
        
        
        
        sess = tf.InteractiveSession()
        
        x = tf.placeholder(tf.float32, shape=[None, resolution*resolution])
        ycorr = tf.placeholder(tf.float32, shape=[None, 10])
        
        x_image = tf.reshape(x, [-1, resolution, resolution, 1])
        
        #First Convolutional Layer
        
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        
        #Second Convolutional Layer
        
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        
        #Densely Connected Layer
        
        W_fc1 = self.weight_variable([int(np.ceil(resolution/4)) * int(np.ceil(resolution/4)) * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, int(np.ceil(resolution/4))*int(np.ceil(resolution/4))*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        #Dropout
        
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        #Readout Layer
        
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        
        start=time.time() #pour calculer le temps de calcul
        
        #training
        
        #methode d'optimisation des poids: AdamOptimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ycorr, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ycorr, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          
          for i in range(generations):
            indices=np.random.choice(55000,batchSize) #on s'entraine sur un echantillon d'images choisies aléatoirement (taille echantillon = batchSize)
            if (error==0):
                batch0 = [train_images[j] for j in indices]
            else:
                batch0 = [self.pixel_bit_error(error,train_images[j],8) for j in indices]
            batch1= [train_labels[k] for k in indices]
            print(i)
            train_step.run(feed_dict={x: batch0, ycorr: batch1, keep_prob: 1})
        
          #test
          print('test accuracy %g' % accuracy.eval(feed_dict={x: test_images, ycorr: test_labels, keep_prob: 1.0}))
          end=time.time()
          print("temps de calcul (s):", end-start)
    
      




