#!/usr/bin/python3

import hstego
import numpy as np
import scipy.signal
import imageio
from scipy import misc, signal

input_image = 'files/1.pgm'


def HILL(I):  
    HF1 = np.array([                                                             
        [-1, 2, -1],                                                             
        [ 2,-4,  2],                                                             
        [-1, 2, -1]                                                              
    ])                                                                           
    H2 = np.ones((3, 3)).astype(np.float)/3**2                                   
    HW = np.ones((15, 15)).astype(np.float)/15**2                                
                                                                                 
    R1 = scipy.signal.convolve2d(I, HF1, mode='same', boundary='symm')
    W1 = scipy.signal.convolve2d(np.abs(R1), H2, mode='same', boundary='symm')
    rho=1./(W1+10**(-10))
    cost = scipy.signal.convolve2d(rho, HW, mode='same', boundary='symm')
    return cost     

I = imageio.imread(input_image)
costs = HILL(I)
print(costs)

hstego.embed(input_image, costs, 'files/message.txt', 's3cr3t', 'files/stego.png')
hstego.extract('files/stego.png', 's3cr3t', 'files/output.txt')

print(open('files/output.txt', 'r').read())




