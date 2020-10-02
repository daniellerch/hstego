#!/usr/bin/env python3

import sys
import imageio
import hstegolib
import scipy.signal
import numpy as np


# {{{ HILL()
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
# }}}

# {{{ help()
def help():
    print("\nUsage:")
    print("  ", sys.argv[0], "embed <input msg file> <input cover image> <output stego image> [password]")
    print("  ", sys.argv[0], "extract <input stego image> <output msg file> [password]")

    print("\nExample:")
    print("  ", sys.argv[0], "embed input-secret.txt cover.png stego.png p4ssw0rd")
    print("  ", sys.argv[0], "extract stego.png output-secret.txt p4ssw0rd")
    print("")
    sys.exit(0)
# }}}



if __name__ == "__main__":

    if len(sys.argv)<=1:
        help()

    elif sys.argv[1] == "embed":
        if len(sys.argv) < 5:
            help()

        msg_file_path = sys.argv[2]
        input_img_path = sys.argv[3]
        output_img_path = sys.argv[4]
        password = sys.argv[5]
        I = imageio.imread(input_img_path)
        cost = HILL(I)
        hstegolib.embed(input_img_path, cost, msg_file_path, password, output_img_path, payload=0.25)

    elif sys.argv[1] == "extract":
        if len(sys.argv) < 3:
            help()

        stego_img_path = sys.argv[2]
        output_msg_path = sys.argv[3]
        password = sys.argv[4]
        I = imageio.imread(stego_img_path)
        hstegolib.extract(stego_img_path, password, output_msg_path, payload=0.25)

    else:
        help()





