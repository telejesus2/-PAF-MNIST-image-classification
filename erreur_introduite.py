#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:29:28 2017

@author: jesusbm
"""




import numpy as np

#fonction qui renvoie une image modifiée. Elle prends en entrée l'image, le nombre de bits par pixel de l'image, et la probabilité que chaque pixel soit modifié
def pixel_bit_error(pe, image, nbits):
    N = len(image)
    err_image = np.zeros_like(image)
    pixel_error = np.random.choice([0, 1], N, p=[1-pe, pe])
    for i in range(N):
        #print(pixel_error[i])
        if pixel_error[i]:
            #print("MOD")
            bit = np.random.randint(0, nbits)
            #print(int(image[i]), bit)
            nrm = 2**nbits - 1.0
            err_image[i] = (int(image[i]*nrm) ^ (2**bit))/nrm
        else:
            err_image[i] = image[i]
    return err_image

#fonction pour sauvegarder des bases de données d'images modifiées    
def save_modified_data(pe, images, nbits):
    modified_images = np.zeros_like(images)
    for i in range(len(modified_images)):
        modified_images[i] = pixel_bit_error(pe, images[i], nbits)
    np.save("./data/8bit_14x14_error/8bit_14x14_error_50_test_images.npy", modified_images)
