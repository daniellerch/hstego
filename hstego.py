#!/usr/bin/env python3

import os
import sys
import getpass
import imageio
import hstegolib

from PIL import Image, ImageTk
import numpy as np


def help():
    print("\nUsage:")
    print(f"  {sys.argv[0]} embed <msg file> <cover image> <output stego image> [password]")
    print(f"  {sys.argv[0]} extract <stego image> <output msg file> [password]")
    print(f"  {sys.argv[0]} capacity <image> [--raw]")

    print("\nExample:")
    print(f"  {sys.argv[0]} embed input-secret.txt cover.png stego.png p4ssw0rd")
    print(f"  {sys.argv[0]} extract stego.png output-secret.txt p4ssw0rd")
    print("")
    sys.exit(0)

def same_extension(path1, path2):
    fn1, ext1 = os.path.splitext(path1)
    fn2, ext2 = os.path.splitext(path2)
    if ext1!=ext2:
        return False
    return True

def get_cover_capacity(img_path):
    image_capacity = 0
    hstegolib.validate_image_resource(img_path)

    if hstegolib.is_ext(img_path, hstegolib.SPATIAL_EXT):
        I = np.asarray(Image.open(img_path))
        image_capacity = hstegolib.spatial_capacity(I)
    else:
        jpg = hstegolib.jpeg_load(img_path)
        image_capacity = hstegolib.jpg_capacity(jpg)
    return image_capacity


def cli_excepthook(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.excepthook = cli_excepthook

    if len(sys.argv)<=1:
        has_gui = False
        window = None
        try:
            import hstegogui
            window = hstegogui.init()
            has_gui = True
        except:
            print("WARNING: Graphical user interface cannot be initialized.")
            import traceback
            traceback.print_exc()


        if has_gui:
            hstegogui.run(window)
        else:
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

 
        with open(msg_file_path, 'rb') as f:
            content = f.read()
        try:
            msg_len = len(content.encode())
        except:
            msg_len = len(content)

        if msg_len > get_cover_capacity(input_img_path):
            print('The message is too long for the selected cover')
            sys.exit(0)

       
        if hstegolib.is_ext(output_img_path, hstegolib.SPATIAL_EXT):

            suniw = hstegolib.S_UNIWARD()
            suniw.embed(input_img_path, msg_file_path, password, output_img_path)

        elif hstegolib.is_ext(input_img_path, "jpg"):

            juniw = hstegolib.J_UNIWARD()
            juniw.embed(input_img_path, msg_file_path, password, output_img_path)

        else:
            print("File extension not supported")



    elif sys.argv[1] == "extract":
        if len(sys.argv) < 4:
            help()

        stego_img_path = sys.argv[2]
        output_msg_path = sys.argv[3]

        if len(sys.argv) == 5:
            password = sys.argv[4]
        else:
            password = getpass.getpass(prompt="Password: ")

        
        if hstegolib.is_ext(stego_img_path, hstegolib.SPATIAL_EXT):

            suniw = hstegolib.S_UNIWARD()
            suniw.extract(stego_img_path, password, output_msg_path)

        elif hstegolib.is_ext(stego_img_path, "jpg"):

            juniw = hstegolib.J_UNIWARD()
            juniw.extract(stego_img_path, password, output_msg_path)

        else:
            print("File extension not supported")


    elif sys.argv[1] == "capacity":
        if len(sys.argv) < 3:
            help()

        img_path = sys.argv[2]
        raw_output = len(sys.argv) == 4 and sys.argv[3] == "--raw"
        hstegolib.validate_image_resource(img_path)

        if hstegolib.is_ext(img_path, hstegolib.SPATIAL_EXT):
            I = imageio.imread(img_path)
            capacity = hstegolib.spatial_capacity(I)
        elif hstegolib.is_ext(img_path, "jpg"):
            jpg = hstegolib.jpeg_load(img_path)
            capacity = hstegolib.jpg_capacity(jpg)
        else:
            print("File extension not supported")
            sys.exit(1)

        if raw_output:
            print(capacity)
        else:
            print("Capacity:", capacity, "bytes")
 


    elif sys.argv[1] == "stc-test":
        hstegolib.stc_test(100)

    else:
        help()



