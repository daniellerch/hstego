#!/usr/bin/env python3

import os
import sys
import getpass
import imageio
import hstegolib

SPATIAL_EXT = ["png", "pgm", "tif"]

# TODO: Limit payload

def help():
    print("\nUsage:")
    print("  hstego.py embed <msg file> <cover image> <output stego image> [password]")
    print("  hstego.py extract <stego image> <output msg file> [password]")
    print("  hstego.py capacity <image>")

    print("\nExample:")
    print("  hstego.py embed input-secret.txt cover.png stego.png p4ssw0rd")
    print("  hstego.py extract stego.png output-secret.txt p4ssw0rd")
    print("")
    sys.exit(0)

def is_ext(path, extensions):
    fn, ext = os.path.splitext(path)
    if ext[1:].lower() in extensions:
        return True
    return False

def same_extension(path1, path2):
    fn1, ext1 = os.path.splitext(path1)
    fn2, ext2 = os.path.splitext(path2)
    if ext1!=ext2:
        return False
    return True


if __name__ == "__main__":

    if len(sys.argv)<=1:
        help()

    elif sys.argv[1] == "embed":
        if len(sys.argv) < 5:
            help()

        msg_file_path = sys.argv[2]
        input_img_path = sys.argv[3]
        output_img_path = sys.argv[4]

        if len(sys.argv) == 6:
            password = sys.argv[5]
        else:
            password = getpass.getpass(prompt="Password: ")
 

        #if not same_extension(input_img_path, output_img_path):
        #    print("Error, input and output images should have the same extension")
        #    sys.exit(-1)

        
        #try:
        if is_ext(output_img_path, SPATIAL_EXT):

            hill = hstegolib.HILL()
            hill.embed(input_img_path, msg_file_path, password, output_img_path)

        elif is_ext(input_img_path, "jpg"):

            juniw = hstegolib.J_UNIWARD()
            juniw.embed(input_img_path, msg_file_path, password, output_img_path)

        else:
            print("File extension not supported")

        #except Exception as e:
        #    print("Error, information can not be embedded:", e)



    elif sys.argv[1] == "extract":
        if len(sys.argv) < 4:
            help()

        stego_img_path = sys.argv[2]
        output_msg_path = sys.argv[3]

        if len(sys.argv) == 5:
            password = sys.argv[4]
        else:
            password = getpass.getpass(prompt="Password: ")

        
        #try:
        if is_ext(stego_img_path, SPATIAL_EXT):

            hill = hstegolib.HILL()
            hill.extract(stego_img_path, password, output_msg_path)

        elif is_ext(stego_img_path, "jpg"):

            juniw = hstegolib.J_UNIWARD()
            juniw.extract(stego_img_path, password, output_msg_path)

        else:
            print("File extension not supported")


        #except Exception as e:
        #    print("Error, information can not be extracted:", e)

    elif sys.argv[1] == "capacity":
        img_path = sys.argv[2]
        if is_ext(img_path, SPATIAL_EXT):
            I = imageio.imread(img_path)
            print("Capacity:", hstegolib.spatial_capacity(I), "bytes")
        elif is_ext(img_path, "jpg"):
            jpg = hstegolib.jpeg_load(img_path)
            print("Capacity:", hstegolib.jpg_capacity(jpg), "bytes")
        else:
            print("File extension not supported")
 


    elif sys.argv[1] == "stc-test":
        hstegolib.stc_test(100)

    else:
        help()





