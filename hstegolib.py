
import os
import sys
import glob
import copy
import errno
import random
import struct
import base64
import hashlib
import imageio
import numpy as np

from ctypes import *
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

jpg_candidates = glob.glob(os.path.join(os.path.dirname(__file__), 'jpeg_toolbox_extension.*.so'))
if not jpg_candidates:
    print("JPEG Toolbos library not found:", os.path.dirname(__file__))
    sys.exit(0)
jpeg = CDLL(jpg_candidates[0])

stc_candidates = glob.glob(os.path.join(os.path.dirname(__file__), 'stc_extension.*.so'))
if not stc_candidates:
    print("SRC library not found:", os.path.dirname(__file__))
    sys.exit(0)
stc = CDLL(stc_candidates[0])


# {{{ load()
def load(path, use_blocks=False):

    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT, 
                os.strerror(errno.ENOENT), path)

    jpeg.write_file.argtypes = c_char_p,
    jpeg.read_file.restype = py_object
    r = jpeg.read_file(path.encode())

    r["quant_tables"] = np.array(r["quant_tables"])

    for i in range(len(r["ac_huff_tables"])):
        r["ac_huff_tables"][i]["counts"] = np.array(r["ac_huff_tables"][i]["counts"])
        r["ac_huff_tables"][i]["symbols"] = np.array(r["ac_huff_tables"][i]["symbols"])

    for i in range(len(r["dc_huff_tables"])):
        r["dc_huff_tables"][i]["counts"] = np.array(r["dc_huff_tables"][i]["counts"])
        r["dc_huff_tables"][i]["symbols"] = np.array(r["dc_huff_tables"][i]["symbols"])

    if not use_blocks:
        chn = len(r["coef_arrays"])
        for c in range(chn):
            r["coef_arrays"][c] = np.array(r["coef_arrays"][c])
            h = r["coef_arrays"][c].shape[0]*8
            w = r["coef_arrays"][c].shape[1]*8
            r["coef_arrays"][c] = np.moveaxis(r["coef_arrays"][c], [0,1,2,3], [0,2,1,3])
            r["coef_arrays"][c] = r["coef_arrays"][c].reshape((h, w))

    return r
# }}}

# {{{ save()
def save(data, path, use_blocks=False):
    jpeg = CDLL(SO_PATH)
    jpeg.write_file.argtypes = py_object,c_char_p

    r = copy.deepcopy(data)
    r["quant_tables"] = r["quant_tables"].tolist()

    for i in range(len(r["ac_huff_tables"])):
        r["ac_huff_tables"][i]["counts"] = r["ac_huff_tables"][i]["counts"].tolist()
        r["ac_huff_tables"][i]["symbols"] = r["ac_huff_tables"][i]["symbols"].tolist()

    for i in range(len(r["dc_huff_tables"])):
        r["dc_huff_tables"][i]["counts"] = r["dc_huff_tables"][i]["counts"].tolist()
        r["dc_huff_tables"][i]["symbols"] = r["dc_huff_tables"][i]["symbols"].tolist()

    if not use_blocks:
        chn = len(r["coef_arrays"])
        for c in range(chn):
            h = r["coef_arrays"][c].shape[0]
            w = r["coef_arrays"][c].shape[1]
            r["coef_arrays"][c] = r["coef_arrays"][c].reshape((h//8, 8, w//8, 8))
            r["coef_arrays"][c] = np.moveaxis(r["coef_arrays"][c], [0,1,2,3], [0,2,1,3])
            r["coef_arrays"][c] = r["coef_arrays"][c].tolist()


    jpeg.write_file(r, path.encode())
# }}}


# {{{ encrypt()
def encrypt(plain_text, password):

    salt = get_random_bytes(AES.block_size)

    # use the Scrypt KDF to get a private key from the password
    private_key = hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

    cipher = AES.new(private_key, AES.MODE_CBC)
    cipher_text = cipher.encrypt(pad(plain_text, AES.block_size))
    enc = salt+cipher.iv+cipher_text

    return enc
# }}}

# {{{ decrypt()
def decrypt(cipher_text, password):

    salt = cipher_text[:AES.block_size]
    iv = cipher_text[AES.block_size:AES.block_size*2]
    cipher_text = cipher_text[AES.block_size*2:]

    # Fix padding
    mxlen = len(cipher_text)-(len(cipher_text)%AES.block_size)
    cipher_text = cipher_text[:mxlen]

    private_key = hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

    cipher = AES.new(private_key, AES.MODE_CBC, iv=iv)
    decrypted = cipher.decrypt(cipher_text)

    return decrypted
# }}}

# {{{ prepare_message()
def prepare_message(filename, password):

    f = open(filename, 'r')
    content_data = f.read()

    # Prepare a header with basic data about the message
    content_ver=struct.pack("B", 1) # version 1
    content_len=struct.pack("!I", len(content_data))
    content=content_ver+content_len+content_data.encode('utf8')

    # encrypt
    enc = encrypt(content, password)

    array=[]
    for b in enc:
        for i in range(8):
            array.append((b >> i) & 1)
    return array
# }}}

# {{{ embed()
def embed(input_img_path, cost_matrix,  msg_file_path, password, output_img_path, payload=0.10):

    I = imageio.imread(input_img_path)
    width, height = I.shape

    # Prepare cover image
    cover = (c_int*(width*height))()
    idx=0
    for j in range(height):
        for i in range(width):
            cover[idx] = I[i, j]
            idx += 1

    # Prepare costs
    INF = 2**31-1
    costs = (c_float*(width*height*3))()
    idx=0
    for j in range(height):
        for i in range(width):
            if cover[idx]==0:
                costs[3*idx+0] = INF
                costs[3*idx+1] = 0
                costs[3*idx+2] = cost_matrix[i, j]
            elif cover[idx]==255:
                costs[3*idx+0] = cost_matrix[i, j]
                costs[3*idx+1] = 0 
                costs[3*idx+2] = INF
            else:
                costs[3*idx+0] = cost_matrix[i, j]
                costs[3*idx+1] = 0
                costs[3*idx+2] = cost_matrix[i, j]
            idx += 1

    # Prepare message
    msg_bits = prepare_message(msg_file_path, password)
    if len(msg_bits)>width*height*payload:
        print("Message too long")
        sys.exit(0)

    m = int(width*height*payload)
    message = (c_ubyte*m)()
    for i in range(m):
        if i<len(msg_bits):
            message[i] = msg_bits[i]
        else:
            message[i] = 0

    # Hide message
    stego = (c_int*(width*height))()
    a = stc.stc_hide(width*height, cover, costs, m, message, stego)

    # Save output message
    idx=0
    for j in range(height):
        for i in range(width):
            I[i, j] = stego[idx]
            idx += 1
    imageio.imwrite(output_img_path, I)
# }}}   

# {{{ extract()
def extract(stego_img_path, password, output_msg_path, payload=0.10):

    I = imageio.imread(stego_img_path)
    width, height = I.shape

    # Prepare stego image
    stego = (c_int*(width*height))()
    idx=0
    for j in range(height):
        for i in range(width):
            stego[idx] = I[i, j]
            idx += 1

    # Extract the message
    n = width*height;
    m = int(n*payload)
    extracted_message = (c_ubyte*m)()
    s = stc.stc_unhide(n, stego, m, extracted_message)

    # Save the message
    enc = bytearray()
    idx=0
    bitidx=0
    bitval=0
    for b in extracted_message:
        if bitidx==8:
            #enc += chr(bitval)
            enc.append(bitval)
            #print("bitval:", bitval)
            bitidx=0
            bitval=0
        bitval |= b<<bitidx
        bitidx+=1

    enc = bytes(enc)   


    cleartext = decrypt(enc, password)
 
    # Extract the header and the message
    content_ver=struct.unpack_from("B", cleartext, 0)
    content_len=struct.unpack_from("!I", cleartext, 1)
    content=cleartext[5:content_len[0]+5]
    print(content)

    f = open(output_msg_path, 'w')
    f.write(content.decode())
    f.close()
# }}}


