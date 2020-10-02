#!/usr/bin/env python3

import sys
import getpass
import imageio
import hstegolib

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

        if len(sys.argv) == 6:
            password = sys.argv[5]
        else:
            password = getpass.getpass(prompt="Password: ")
        
        hstegolib.HILL_embed(input_img_path, msg_file_path, password, output_img_path, payload=0.10)

    elif sys.argv[1] == "extract":
        if len(sys.argv) < 4:
            help()

        stego_img_path = sys.argv[2]
        output_msg_path = sys.argv[3]

        if len(sys.argv) == 5:
            password = sys.argv[4]
        else:
            password = getpass.getpass(prompt="Password: ")

        I = imageio.imread(stego_img_path)
        hstegolib.HILL_extract(stego_img_path, password, output_msg_path, payload=0.10)

    else:
        help()





