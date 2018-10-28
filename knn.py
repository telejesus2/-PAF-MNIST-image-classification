#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:18:55 2017

@author: jesusbm
"""

import tensorflow as tf
import numpy as np
import time

class Knn(object):
    
    def __init__(self, k):
        
        self.K=k
        
    def run(self, train_data, test_data,resolution ):
     
        train_images=train_data
        test_images=test_data
      
        train_labels=np.load("./data/train_labels.npy")
        test_labels=np.load("./data/test_labels.npy")
        
        xtr = tf.placeholder(tf.float32, [None, resolution*resolution])
        xte=tf.placeholder(tf.float32,[resolution*resolution]) #
        ycorr = tf.placeholder(tf.float32, [None, 10])
       
        #nombre de voisins
        nearest_neighbors=tf.Variable(tf.zeros([self.K]))
        
        #norme L2
        distance = tf.negative(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)),axis=1)) 
        # on veut maximiser distance
        values,indices=tf.nn.top_k(distance,k=self.K,sorted=False)
        
        nn = []
        for i in range(self.K):
            nn.append(tf.argmax(ycorr[indices[i]], 0)) #nn contient les k voisins les plus proches
        
        nearest_neighbors=nn
        
        # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
        #y, idx, count = unique_with_counts(x)
        #y ==> [1, 2, 4, 7, 8]
        #idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
        #count ==> [2, 1, 3, 1, 2]
        y, idx, count = tf.unique_with_counts(nearest_neighbors)
        
        start=time.time()  #pour calculer le temps de calcul
        
        #pred contient le voisin le plus frequent parmi les k les plus proches
        pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
        
        X_training, Y_training = train_images[:20000], train_labels[:20000] 
        X_test, Y_test  =test_images[:1000], test_labels[:1000]

        accuracy=0
        with tf.Session() as sess:
            
            for i in range(X_test.shape[0]):
                predicted_value=sess.run(pred,feed_dict={xtr:X_training,ycorr:Y_training,xte:X_test[i,:]})
        
                print("Test",i,"Prediction",predicted_value,"True Class:",np.argmax(Y_test[i]))
        
                if predicted_value == np.argmax(Y_test[i]):
                    #si la prediction est juste
                    accuracy += 1
            accuracy=accuracy/len(X_test)
            end=time.time()
            print("Calculation completed")
            print(self.K,"-th neighbors' Accuracy is:",accuracy)
            print("temps de calcul (s):", end-start)
            
