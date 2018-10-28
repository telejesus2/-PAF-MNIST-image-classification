#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:20:00 2017


Module qui reutilise les fonctions de lecture de tensorflow pour reformatter les images dans MNIST

"""

import numpy as np
from scipy.signal import convolve2d


#fonction qui change le nombre de bits par pixel d'une image
def down_sample(image, nbits):
    down_sampled_image = np.zeros_like(image)
    for i in range(len(image)):
        #formule pour reduire le nombre de bits, elle marche
        down_sampled_image[i] = np.floor((image[i] * 255) / (2**(8-nbits)))/(2**nbits-1) 
    return down_sampled_image
  
    
#fonction pour sauvegarder des bases de données d'images modifiées    
def save_data(nbits,images): 
    down_sampled_images = np.zeros_like(images)
    for i in range(len(down_sampled_images)):
        down_sampled_images[i] = down_sample(images[i],nbits)
    
    np.save("./data/1bit/1bit_14x14_train_images.npy", down_sampled_images)


#fonction qui réduit à moitié la résolution d'une image, grâce à un filtre passe-bas
def resize_by_half(image):
    
    #lowpass
    kernel = 0.25 * np.ones(4).reshape((2,2))
    filtered = convolve2d(image, kernel)[1:, 1:]
    
    #downsampling
    resized = [[filtered[i][j] for j in range(0, len(filtered[i]),2)] for i in range(0, len(filtered), 2)]
    return resized