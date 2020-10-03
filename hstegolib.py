
import os
import sys
import glob
import gzip
import copy
import errno
import random
import struct
import base64
import hashlib
import imageio
import scipy.signal
import scipy.fftpack
import numpy as np

from ctypes import *
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


INF = 2**31-1

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


# {{{ jpeg_load()
def jpeg_load(path, use_blocks=False):

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

# {{{ jpeg_save()
def jpeg_save(data, path, use_blocks=False):

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
def prepare_message(data, password):

    data = gzip.compress(data)
    data_len = struct.pack("!I", len(data))
    data = data_len + data

    # encrypt
    enc = encrypt(data, password)

    array=[]
    for b in enc:
        for i in range(8):
            array.append((b >> i) & 1)
    return array
# }}}


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

    cost[np.isnan(cost)] = INF
    cost[cost>INF] = INF

    return cost     
# }}}

# {{{ HILL_embed()
def HILL_embed(input_img_path, msg_file_path, password, output_img_path, payload=0.10):

    with open(msg_file_path, 'rb') as f:
        data = f.read()

    I = imageio.imread(input_img_path)

    if len(I.shape) == 2:
        height, width = I.shape
        n_channels = 1
        cost_matrix = [HILL(I)]
        I = I[..., np.newaxis]
        msg_bits = [prepare_message(data, password)]

    elif len(I.shape) == 3:
        height, width, _ = I.shape
        n_channels = 3
        cost_matrix = [HILL(I[:,:,0]), HILL(I[:,:,1]), HILL(I[:,:,2])]
        l = len(data)//3
        msg_bits = [ prepare_message(data[:l], password), 
                     prepare_message(data[l:2*l], password), 
                     prepare_message(data[2*l:], password) ]


    for channel in range(n_channels):

        if len(msg_bits[channel])>width*height*payload*8:
            print("Message too long:", len(msg_bits[channel]), "bits >", width*height*payload*8, "max bits")
            sys.exit(-1)

        # Prepare cover image
        cover = (c_int*(width*height))()
        idx=0
        for j in range(height):
            for i in range(width):
                cover[idx] = I[i, j, channel]
                idx += 1

        # Prepare costs
        costs = (c_float*(width*height*3))()
        idx=0
        for j in range(height):
            for i in range(width):
                if cover[idx]==0:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_matrix[channel][i, j]
                elif cover[idx]==255:
                    costs[3*idx+0] = cost_matrix[channel][i, j]
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = cost_matrix[channel][i, j]
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_matrix[channel][i, j]
                idx += 1


        m = int(width*height*payload)
        message = (c_ubyte*m)()
        for i in range(m):
            if i<len(msg_bits[channel]):
                message[i] = msg_bits[channel][i]
            else:
                message[i] = 0

        # Hide message
        stego = (c_int*(width*height))()
        a = stc.stc_hide(width*height, cover, costs, m, message, stego)

        # Save output message
        idx=0
        for j in range(height):
            for i in range(width):
                I[i, j, channel] = stego[idx]
                idx += 1

    if n_channels == 1:
        I = I.reshape(I.shape[:-1])

    imageio.imwrite(output_img_path, I)
# }}}   

# {{{ HILL_extract()
def HILL_extract(stego_img_path, password, output_msg_path, payload=0.10):

    I = imageio.imread(stego_img_path)

    if len(I.shape) == 2:
        height, width = I.shape
        n_channels = 1
        I = I[..., np.newaxis]

    elif len(I.shape) == 3:
        height, width, _ = I.shape
        n_channels = 3

    cleartext_list = []
    for channel in range(n_channels):

        # Prepare stego image
        stego = (c_int*(width*height))()
        idx=0
        for j in range(height):
            for i in range(width):
                stego[idx] = I[i, j, channel]
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
                enc.append(bitval)
                bitidx=0
                bitval=0
            bitval |= b<<bitidx
            bitidx+=1

        enc = bytes(enc)   

        cleartext = decrypt(enc, password)
        cleartext_list.append(cleartext)

    content = bytes()
    for cleartext in cleartext_list:
        content_len = struct.unpack_from("!I", cleartext, 0)[0]
        data = cleartext[4:content_len+4]
        content += gzip.decompress(data)

    f = open(output_msg_path, 'wb')
    f.write(content)
    f.close()
# }}}



# {{{ J_UNIWARD()
def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho')
    
def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def J_UNIWARD(coef_arrays, quant_tables, spatial):

    hpdf = np.array([
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,  
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,  
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,  
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])        

    sign = np.array([-1 if i%2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    F = []
    F.append(np.outer(lpdf.T, hpdf))
    F.append(np.outer(hpdf.T, lpdf))
    F.append(np.outer(hpdf.T, hpdf))


    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    spatial_impact = {}
    for i in range(8):
        for j in range(8):
            test_coeffs = np.zeros((8, 8))
            test_coeffs[i, j] = 1
            spatial_impact[i, j] = idct2(test_coeffs) * quant_tables[i, j]

    # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    wavelet_impact = {}
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX


    # Create reference cover wavelet coefficients (LH, HL, HH)
    pad_size = 16 # XXX
    spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')


    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
        RC.append(f)


    coeffs = coef_arrays
    k, l = coeffs.shape
    nzAC = np.count_nonzero(coef_arrays) - np.count_nonzero(coef_arrays[::8, ::8])

    rho = np.zeros((k, l))
    tempXi = [0.]*3
    sgm = 2**(-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size))
            sub_cols = list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size))

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:,sub_cols]
                wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub)+sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)


    rho[np.isnan(rho)] = INF
    rho[rho>INF] = INF

    return rho
# }}}

# {{{ J_UNIWARD_embed()
def J_UNIWARD_embed(input_img_path, msg_file_path, password, output_img_path, payload=0.10):

    with open(msg_file_path, 'rb') as f:
        data = f.read()

    I = imageio.imread(input_img_path)
    jpg = jpeg_load(input_img_path)

    if len(I.shape) == 2:
        height, width = I.shape
        n_channels = 1
        cost_matrix = [J_UNIWARD(jpg["coef_arrays"][0], jpg["quant_tables"][0], I)]
        I = I[..., np.newaxis]
        msg_bits = [prepare_message(data, password)]

    elif len(I.shape) == 3:
        height, width, _ = I.shape
        n_channels = 3
        cost_matrix = [J_UNIWARD(jpg["coef_arrays"][0], jpg["quant_tables"][0], I[:,:,0]), 
                       J_UNIWARD(jpg["coef_arrays"][1], jpg["quant_tables"][1], I[:,:,1]), 
                       J_UNIWARD(jpg["coef_arrays"][2], jpg["quant_tables"][1], I[:,:,2])]
        l = len(data)//3
        msg_bits = [ prepare_message(data[:l], password), 
                     prepare_message(data[l:2*l], password), 
                     prepare_message(data[2*l:], password) ]



    for channel in range(n_channels):

        if len(msg_bits[channel])>width*height*payload:
            print("Message too long:", len(msg_bits[channel]), "bits >", width*height*payload*8, "max bits")
            sys.exit(-1)

        # Prepare cover image
        cover = (c_int*(width*height))()
        idx=0
        for j in range(height):
            for i in range(width):
                cover[idx] = int(jpg["coef_arrays"][channel][i, j])
                idx += 1

        # Prepare costs
        costs = (c_float*(width*height*3))()
        idx=0
        for j in range(height):
            for i in range(width):
                if cover[idx]<=1016:
                    costs[3*idx+0] = INF
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_matrix[channel][i, j]
                elif cover[idx]>=1016:
                    costs[3*idx+0] = cost_matrix[channel][i, j]
                    costs[3*idx+1] = 0 
                    costs[3*idx+2] = INF
                else:
                    costs[3*idx+0] = cost_matrix[channel][i, j]
                    costs[3*idx+1] = 0
                    costs[3*idx+2] = cost_matrix[channel][i, j]
                idx += 1

        m = int(width*height*payload)
        message = (c_ubyte*m)()
        for i in range(m):
            if i<len(msg_bits[channel]):
                message[i] = msg_bits[channel][i]
            else:
                message[i] = 0

        # Hide message
        stego_coeffs = (c_int*(width*height))()
        a = stc.stc_hide(width*height, cover, costs, m, message, stego_coeffs)

        # Save output message
        idx=0
        for j in range(height):
            for i in range(width):
                jpg["coef_arrays"][channel][i, j] = stego_coeffs[idx]
                idx += 1

    jpeg_save(jpg, output_img_path)

# }}}   

# {{{ J_UNIWARD_extract()
def J_UNIWARD_extract(stego_img_path, password, output_msg_path, payload=0.10):

    jpg = jpeg_load(stego_img_path)
    I = imageio.imread(stego_img_path)

    if len(I.shape) == 2:
        height, width = I.shape
        n_channels = 1
        I = I[..., np.newaxis]

    elif len(I.shape) == 3:
        height, width, _ = I.shape
        n_channels = 3

    cleartext_list = []
    for channel in range(n_channels):

        # Prepare stego image
        stego = (c_int*(width*height))()
        idx=0
        for j in range(height):
            for i in range(width):
                stego[idx] = int(jpg["coef_arrays"][channel][i, j])
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
                enc.append(bitval)
                bitidx=0
                bitval=0
            bitval |= b<<bitidx
            bitidx+=1

        enc = bytes(enc)   

        cleartext = decrypt(enc, password)
        cleartext_list.append(cleartext)
 

    content = bytes()
    for cleartext in cleartext_list:
        content_len = struct.unpack_from("!I", cleartext, 0)[0]
        data = cleartext[4:content_len+4]
        content += gzip.decompress(data)

    f = open(output_msg_path, 'wb')
    f.write(content)
    f.close()
# }}}




