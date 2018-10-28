#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:05:09 2017

@author: jesusbm
"""
from cnn import Cnn
from knn import Knn
from svm import Svm
from Caliente import Caliente
import numpy as np

class Jeello(object):
    
    def __init__(self, nbit, resolution, error):
        print("welcome to our paf")
        
        x = input("enter your favorite algorithm: 1-CNN 2-SVM 3-KNN 4-CALIENTE")
        self.run(x,nbit, resolution, error)
    
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
    
    def run(self,x,nbit, resolution, error):
        
        nbits=str(nbit)
        test_data= np.load( "./data/"+ nbits+"bit" +"/"+nbits+ "bit" + "_"+ str(resolution) + "x" + str(resolution) +"_test_images.npy")
        train_data= np.load( "./data/"+ nbits+"bit" +"/"+nbits+ "bit" + "_"+ str(resolution) + "x" + str(resolution) +"_train_images.npy")

        x=int(x)                           
        if (error!=0):
            test_data= np.array([self.pixel_bit_error(error, i, nbit) for i in test_data])
                
        
        if (x==1):
            generations= input("enter the number of generations")
            batchSize= input("enter the size of each batch")
            generations=int(generations)
            batchSize=int(batchSize)
            Jesus=Cnn()
            Jesus.run(train_data, test_data, resolution, error, generations, batchSize)
        if (x==2):
            if (error!=0):
                train_data= np.array([self.pixel_bit_error(error, i, nbit) for i in train_data])
            Jesus=Svm()
            Jesus.run(train_data, test_data,resolution)
        if (x==3):
            if (error!=0):
                train_data= np.array([self.pixel_bit_error(error, i, nbit) for i in train_data])
            k=input("k ?")
            k=int(k)
            Jesus=Knn(k)
            Jesus.run(train_data,test_data,resolution)
        if (x==4):
            Jesus=Caliente([],error)
            batchSize= input("enter the size of each batch")
            generations= input("enter the number of generations")
            generations=int(generations)
            batchSize=int(batchSize)
            Jesus.run( train_data, test_data, resolution, generations, batchSize)
            
            

        