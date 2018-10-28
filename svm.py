#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:40:48 2017

@author: jesusbm
"""
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
import time

class Svm(object):
    def __init__(self):
        pass
        
    def run(self, train_data, test_data, resolution):
     
        train_images=train_data
        test_images=test_data

        train_labels=np.load("./data/train_labels.npy")
        test_labels=np.load("./data/test_labels.npy")
        
    
        start=time.time() #pour calculer le temps de calcul
        
        clf = SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073) #classifieur avec kernel gaussien
      
        train_labels=[np.argmax(i) for i in train_labels]
        clf.fit(train_images,train_labels)        #apprentissage
        
        print("apprentissage fini")
            
        predicted = clf.predict(test_images) #test
        
        test_labels=[np.argmax(i) for i in test_labels]
    
        print("Accuracy: %0.4f" % metrics.accuracy_score(test_labels, predicted))
        
        end=time.time()
        print("temps de calcul (s):", end-start)
    
    
 
        
    