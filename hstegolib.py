#!/usr/bin/env python3

import os
import sys
import glob
import copy
import struct
import base64
import random
import imageio
import hashlib
import warnings

import scipy.signal
import scipy.fftpack
import scipy.ndimage
import numpy as np

from ctypes import *
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


from numba import jit

warnings.filterwarnings('ignore', category=RuntimeWarning)


SPATIAL_EXT = ["png", "pgm", "tif"]
MAX_PAYLOAD=0.05
INF = 2**31-1


base = os.path.dirname(__file__)

jpg_pattern = 'hstego_jpeg_toolbox_extension*.so'
stc_pattern = 'hstego_stc_extension*.so'

# running in a pyinstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS

jpg_candidates = glob.glob(os.path.join(base, jpg_pattern))
if not jpg_candidates and sys.platform == "linux": # devel mode
    jpg_candidates = glob.glob('build/lib.linux*/'+jpg_pattern)
if not jpg_candidates and sys.platform == "win32": # devel mode
    jpg_candidates = glob.glob('build/lib.win*/'+jpg_pattern)
if not jpg_candidates and sys.platform == "darwin": # devel mode
    jpg_candidates = glob.glob('build/lib.mac*/'+jpg_pattern)
if not jpg_candidates:
    print("JPEG Toolbox library not found:", base)
    sys.exit(0)
jpeg = CDLL(jpg_candidates[0])

stc_candidates = glob.glob(os.path.join(base, stc_pattern))
if not stc_candidates and sys.platform == "linux": # devel mode
    stc_candidates = glob.glob('build/lib.linux*/'+stc_pattern)
if not stc_candidates and sys.platform == "win32": # devel mode
    stc_candidates = glob.glob('build/lib.win*/'+stc_pattern)
if not stc_candidates and sys.platform == "darwin": # devel mode
    stc_candidates = glob.glob('build/lib.mac*/'+stc_pattern)
if not stc_candidates:
    print("STC library not found:", base)
    sys.exit(0)
stc = CDLL(stc_candidates[0])


# {{{ is_ext()
def is_ext(path, extensions):
    fn, ext = os.path.splitext(path)
    if ext[1:].lower() in extensions:
        return True
    return False
# }}}

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
    r["quant_tables"] = r["quant_tables"].astype('uint16').tolist()

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

# {{{ jpg_capacity()
def jpg_capacity(jpg):
    """ channel capacity in bytes """
    total_capacity = 0
    for channel in range(len(jpg["coef_arrays"])):
        nz_coeff = np.count_nonzero(jpg["coef_arrays"][channel])
        capacity = int((nz_coeff*MAX_PAYLOAD)/8)
        f = capacity // 16
        capacity = (f-1)*16
        capacity -= 16+16+4 # data for header
        if capacity<0:
            capacity = 0
        total_capacity += capacity
    return total_capacity
# }}}

# {{{ spatial_capacity()
def spatial_capacity(I):
    m = 1
    for i in I.shape:
        m *= i

    capacity = int((m*MAX_PAYLOAD)/8)
    f = capacity // 16
    capacity = (f-1)*16
    capacity -= 16+16+4 # data for header
    if capacity<0:
        capacity = 0

    return capacity
# }}} 



class Cipher:
    # {{{
    def __init__(self, password):
        self.password = password

    def open_file(self, path):
        with open(path, 'rb') as f:
            self.plaintext = f.read()

    def aes_encrypt(self):
        salt = get_random_bytes(AES.block_size)

        # use the Scrypt KDF to get a private key from the password
        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

        cipher = AES.new(private_key, AES.MODE_CBC)
        ciphertext = cipher.encrypt(pad(self.plaintext, AES.block_size))

        # Ciphertext with a 16+16 header 
        self.ciphertext = salt + cipher.iv + ciphertext

    def encrypt(self, input_path):
        """ encrypt file content """
        self.open_file(input_path)
        self.aes_encrypt()
        return self.ciphertext
  
    def aes_decrypt(self):
        salt = self.ciphertext[:AES.block_size]
        iv = self.ciphertext[AES.block_size:AES.block_size*2]
        ciphertext = self.ciphertext[AES.block_size*2:]

        # Fix padding
        mxlen = len(ciphertext)-(len(ciphertext)%AES.block_size)
        ciphertext = ciphertext[:mxlen]

        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

        cipher = AES.new(private_key, AES.MODE_CBC, iv=iv)
        decrypted = cipher.decrypt(ciphertext)
        self.decrypted = unpad(decrypted, AES.block_size)

    def decrypt(self, bytes_array):
        """ decrypt """
        try:
            self.ciphertext = bytes_array
            self.aes_decrypt()
            return self.decrypted
        except:
            print("WARNING: message not found")
            return bytearray()
    # }}}

class Stego:
    # {{{
    def __init__(self):
        pass

    def bytes_to_bits(self, data):
        array=[]
        for b in data:
            for i in range(8):
                array.append((b >> i) & 1)
        return array


    def hide_stc(self, cover_array, costs_array_m1, costs_array_p1, 
                 message_bits, mx=255, mn=0):

        cover = (c_int*(len(cover_array)))()
        for i in range(len(cover_array)):
            cover[i] = int(cover_array[i])

        # Prepare costs
        costs = (c_float*(len(costs_array_m1)*3))()
        # 0: cost of changing by -1
        # 1: cost of not changing
        # 2: cost of changing by +1
        for i in range(len(costs_array_m1)):
            if cover[i]<=mn:
                costs[3*i+0] = INF
                costs[3*i+1] = 0
                costs[3*i+2] = costs_array_p1[i]
            elif cover[i]>=mx:
                costs[3*i+0] = costs_array_m1[i]
                costs[3*i+1] = 0 
                costs[3*i+2] = INF
            else:
                costs[3*i+0] = costs_array_m1[i]
                costs[3*i+1] = 0
                costs[3*i+2] = costs_array_p1[i]


        m = len(message_bits)
        message = (c_ubyte*m)()
        for i in range(m):
            message[i] = message_bits[i]

        # Hide message
        stego = (c_int*(len(cover_array)))()
        _ = stc.stc_hide(len(cover_array), cover, costs, m, message, stego)

        # stego data to numpy
        stego_array = cover_array.copy()
        for i in range(len(cover_array)):
            stego_array[i] = stego[i]
     
        return stego_array


    def hide(self, message, cover_matrix, cost_matrix_m1, cost_matrix_p1,
             password_seed, mx=255, mn=0):
        random.seed(password_seed)

        message_bits = self.bytes_to_bits(message)

        height, width = cover_matrix.shape
        cover_array = cover_matrix.reshape((height*width,)) 
        costs_array_m1 = cost_matrix_m1.reshape((height*width,)) 
        costs_array_p1 = cost_matrix_p1.reshape((height*width,)) 

        # Hide data_len (32 bits) into 64 pixels (0.5 payload)
        data_len = struct.pack("!I", len(message_bits))
        data_len_bits = self.bytes_to_bits(data_len)

        # Shuffle
        indices = list(range(len(cover_array)))
        random.shuffle(indices)
        cover_array = cover_array[indices]
        costs_array_m1 = costs_array_m1[indices]
        costs_array_p1 = costs_array_p1[indices]

        stego_array_1 = self.hide_stc(cover_array[:64], 
            costs_array_m1[:64], costs_array_p1[:64], data_len_bits, mx, mn)
        stego_array_2 = self.hide_stc(cover_array[64:], 
            costs_array_m1[64:], costs_array_p1[64:], message_bits, mx, mn)
        stego_array = np.hstack((stego_array_1, stego_array_2))


        # Unshuffle
        stego_array[indices] = stego_array[list(range(len(cover_array)))]

        stego_matrix = stego_array.reshape((height, width))
      
        return stego_matrix


    def unhide_stc(self, stego_array, message_len):

        stego = (c_int*(len(stego_array)))()
        for i in range(len(stego_array)):
            stego[i] = int(stego_array[i])

        extracted_message = (c_ubyte*len(stego_array))()
        s = stc.stc_unhide(len(stego_array), stego, message_len, extracted_message)

        if len(extracted_message) < message_len:
            print("WARNING, inconsistent message lenght:", 
                  len(extracted_message), ">", message_len)
            return bytearray()

        # Message bits to bytes
        data = bytearray()
        idx=0
        bitidx=0
        bitval=0
        for i in range(message_len):
            if bitidx==8:
                data.append(bitval)
                bitidx=0
                bitval=0
            bitval |= extracted_message[i]<<bitidx
            bitidx+=1
            idx += 1
        if bitidx==8:
            data.append(bitval)

        data = bytes(data)
        return data


    def unhide(self, stego_matrix, password_seed):

        random.seed(password_seed)

        height, width = stego_matrix.shape
        stego_array = stego_matrix.reshape((height*width,))

        # Shuffle
        indices = list(range(len(stego_array)))
        random.shuffle(indices)
        stego_array = stego_array[indices]

        # Extract a 32-bits message lenght from a 64-pixel array
        data = self.unhide_stc(stego_array[:64], 32)
        data_len = struct.unpack_from("!I", data, 0)[0]
        
        data = self.unhide_stc(stego_array[64:], data_len)
        return data

    # }}}

class S_UNIWARD:
    # {{{ 
    def cost_fn(self, I):

        k, l = I.shape[:2]

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

        sgm = 1
        pad_size = 16 # XXX

        rho = np.zeros((k, l))
        for i in range(3):
            cover_padded = np.pad(I, (pad_size, pad_size), 'symmetric').astype('float32')

            R0 = scipy.signal.convolve2d(cover_padded, F[i], mode="same")

            X = scipy.signal.convolve2d(1./(np.abs(R0)+sgm), np.rot90(np.abs(F[i]), 2), 'same');

            if F[0].shape[0]%2 == 0:
                X = np.roll(X, 1, axis=0)

            if F[0].shape[1]%2 == 0:
                X = np.roll(X, 1, axis=1)

            X = X[(X.shape[0]-k)//2:-(X.shape[0]-k)//2, (X.shape[1]-l)//2:-(X.shape[1]-l)//2]
            rho += X

        rho_m1 = rho.copy()
        rho_p1 = rho.copy()

        rho_p1[rho_p1>INF] = INF
        rho_p1[np.isnan(rho_p1)] = INF
        rho_p1[I==255] = INF

        rho_m1[rho_m1>INF] = INF
        rho_m1[np.isnan(rho_m1)] = INF
        rho_m1[I==0] = INF

        return rho_m1, rho_p1 


    def embed(self, input_img_path, msg_file_path, password, output_img_path):

        with open(msg_file_path, 'rb') as f:
            data = f.read()

        I = imageio.imread(input_img_path)
        
        n_channels = 3
        if len(I.shape) == 2:
            n_channels = 1
            I = I[..., np.newaxis]

        cipher = Cipher(password)
        message = cipher.encrypt(msg_file_path)

        # Seed from password
        password_hash = hashlib.sha256(password.encode()).digest()
        password_seed = int.from_bytes(password_hash, 'big')

        # real capacity, without headers
        m = 1
        for i in I.shape:
            m *= i
        capacity = int((m*MAX_PAYLOAD)/8)
        if len(message) > capacity:
            print("ERROR, message too long:", len(message), ">", capacity)
            sys.exit(0)

        stego = Stego()

        if n_channels == 1:
            msg_bits = [ message ] 
        else:
            l = len(data)//3
            msg_bits = [ message[:l], message[l:2*l], message[2*l:] ]

        for c in range(n_channels):
            costs_m1, costs_p1 = self.cost_fn(I[:,:,c])
            I[:,:,c] = stego.hide(msg_bits[c], I[:,:,c], costs_m1, costs_p1, password_seed)

        if n_channels==1:
            imageio.imwrite(output_img_path, I[:,:,0])
        else:
            imageio.imwrite(output_img_path, I)


    def extract(self, stego_img_path, password, output_msg_path):

        I = imageio.imread(stego_img_path)
       
        n_channels = 3
        if len(I.shape) == 2:
            n_channels = 1
            I = I[..., np.newaxis]

        cipher = Cipher(password)
        stego = Stego()

        # Seed from password
        password_hash = hashlib.sha256(password.encode()).digest()
        password_seed = int.from_bytes(password_hash, 'big')

        ciphertext = []
        for c in range(n_channels):
            ciphertext += stego.unhide(I[:,:,c], password_seed)

        plain = cipher.decrypt(bytes(ciphertext))
        with open(output_msg_path, 'wb') as f:
            f.write(plain)

    # }}}


# {{{ Numba code for J-UNIWARD
A = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if i==0:
            A[i][j] = 0.35355339
        else:
            A[i][j] = 0.5*np.cos(np.pi*(2*j+1)*i/float(16))

@jit(nopython=True, cache=True)
def invDCT(m):
    # {{{
    return np.transpose(np.dot(np.transpose(np.dot(m,A)),A))
    # }}}


B = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        if j==0:
            B[i][j] = 0.35355339
        else:
            B[i][j] = 0.5*np.cos(np.pi*(2*i+1)*j/float(16))

@jit(nopython=True, cache=True)
def DCT(m):   
    # {{{
    return np.transpose(np.dot(np.transpose(np.dot(m,B)),B))
    # }}}


@jit(nopython=True, cache=True)
def uncompress(coeffs, quant):
    # {{{
    spatial = np.zeros(coeffs.shape)
    for block_i in range(coeffs.shape[0]//8):
        for block_j in range(coeffs.shape[1]//8):
            dct_block = coeffs[block_i*8:block_i*8+8, block_j*8:block_j*8+8]
            spatial_block = invDCT(dct_block*quant)+128
            spatial[block_i*8:block_i*8+8, block_j*8:block_j*8+8] = spatial_block
    return spatial
    # }}}

@jit(nopython=True, cache=True)
def compress(spatial, quant):
    # {{{
    dct = np.zeros(spatial.shape)
    for block_i in range(spatial.shape[0]//8):
        for block_j in range(spatial.shape[1]//8):
            spatial_block = spatial[block_i*8:block_i*8+8, block_j*8:block_j*8+8]
            dct_block = DCT(spatial_block-128)/quant
            dct[block_i*8:block_i*8+8, block_j*8:block_j*8+8] = dct_block
    return dct
    # }}}

def upsample(Cb, Cr, Y_shape):
    # {{{ Undo chroma upsampling
    height_r = Y_shape[0] / Cb.shape[0]
    width_r = Y_shape[1] / Cb.shape[1]
    if height_r > 1 or width_r > 1:
        Cb_upsampled = scipy.ndimage.zoom(Cb, (height_r, width_r), order=1)
        Cr_upsampled = scipy.ndimage.zoom(Cr, (height_r, width_r), order=1)
    else:
        Cb_upsampled, Cr_upsampled = Cb, Cr
    return Cb_upsampled, Cr_upsampled
    # }}}

def YCbCr_to_RGB(Y, Cb, Cr):
    # {{{
    Cbu, Cru = upsample(Cb, Cr, Y.shape)
    R = Y + 1.402 * (Cru-128)
    G = Y - 0.34414 * (Cbu-128)  - 0.71414 * (Cru-128) 
    B = Y + 1.772 * (Cbu-128)
    return np.stack((R, G, B), -1)
    # }}}


@jit(nopython=True, cache=True)
def cost_fn_fast(coeffs, wavelet_impact_array, RC, pad_size):
    # {{{
    k, l = coeffs.shape
    rho = np.zeros((k, l))
    tempXi = np.zeros((3, 23, 23))
    sgm = 2**(-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = np.array(list(range(row-mod_row-6+pad_size-1, row-mod_row+16+pad_size)))
            sub_cols = np.array(list(range(col-mod_col-6+pad_size-1, col-mod_col+16+pad_size)))

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:,sub_cols]
                wav_cover_stego_diff = wavelet_impact_array[f_index, mod_row, mod_col]
                tempXi[f_index] = np.abs(wav_cover_stego_diff) / (np.abs(RC_sub)+sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)

    return rho
    # }}}



# }}}

class J_UNIWARD:
    # {{{


    def cost_fn(self, coeffs, spatial, quant):

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
                spatial_impact[i, j] = invDCT(test_coeffs) * quant[i, j]


        # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
        # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
        #wavelet_impact = {}
        wavelet_impact_array = np.zeros((len(F), 8, 8, 23, 23))
        for f_index in range(len(F)):
            for i in range(8):
                for j in range(8):
                    #wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.) # XXX
                    wavelet_impact_array[f_index, i, j, :, :] = scipy.signal.correlate2d(spatial_impact[i, j], F[f_index], mode='full', boundary='fill', fillvalue=0.)




        # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1

        # Create reference cover wavelet coefficients (LH, HL, HH)
        pad_size = 16 # XXX
        spatial_padded = np.pad(spatial, (pad_size, pad_size), 'symmetric')
        #print(spatial_padded.shape)

        RC = []
        for i in range(len(F)):
            f = scipy.signal.correlate2d(spatial_padded, F[i], mode='same', boundary='fill')
            RC.append(f)
        RC = np.array(RC)

        rho = cost_fn_fast(coeffs, wavelet_impact_array, RC, pad_size)

        return rho



    def cost_polarization(self, rho, coeffs, spatial, quant):

        rho_m1 = rho.copy()
        rho_p1 = rho.copy()

        m = 0.65

        print("coeffs shape:", coeffs.shape)
        precover = scipy.signal.wiener(spatial, (3,3)) 
        print("precover shape:", coeffs.shape)
        coeffs_estim = compress(precover, quant)
        print("coeffs_estim shape:", coeffs_estim.shape)

        # polarize
        s = np.sign(coeffs_estim-coeffs)
        rho_p1[s>0] = m*rho_p1[s>0]
        rho_m1[s<0] = m*rho_m1[s<0]

        rho_p1[rho_p1>INF] = INF
        rho_p1[np.isnan(rho_p1)] = INF
        rho_p1[coeffs>1023] = INF

        rho_m1[rho_m1>INF] = INF
        rho_m1[np.isnan(rho_m1)] = INF
        rho_m1[coeffs<-1023] = INF


        return rho_m1, rho_p1


    def embed(self, input_img_path, msg_file_path, password, output_img_path):

        with open(msg_file_path, 'rb') as f:
            data = f.read()

        #I = imageio.imread(input_img_path)
        jpg = jpeg_load(input_img_path)
        if jpg["jpeg_components"] == 1:
            n_channels = 1
            I = uncompress(jpg["coef_arrays"][0], jpg["quant_tables"][0])
            I = I[..., np.newaxis]
            spatial_uncompress = I
        else:
            n_channels = 3
            Y = uncompress(jpg["coef_arrays"][0], jpg["quant_tables"][0])
            Cb = uncompress(jpg["coef_arrays"][1], jpg["quant_tables"][1])
            Cr = uncompress(jpg["coef_arrays"][2], jpg["quant_tables"][1])
            spatial_uncompress = (Y, Cb, Cr)
            I = YCbCr_to_RGB(Y, Cb, Cr)

        cipher = Cipher(password)
        message = cipher.encrypt(msg_file_path)

        # Seed from password
        password_hash = hashlib.sha256(password.encode()).digest()
        password_seed = int.from_bytes(password_hash, 'big')

        # Real capacity, without headers
        capacity = 0
        for channel in range(len(jpg["coef_arrays"])):
            nz_coeff = np.count_nonzero(jpg["coef_arrays"][channel])
            capacity += int((nz_coeff*MAX_PAYLOAD)/8)

        if len(message) > capacity:
            print("ERROR, message too long:", len(message), ">", capacity)
            sys.exit(0)


        stego = Stego()

        if n_channels == 1:
            msg_bits = [ message ] 
        else:
            l = len(data)//3
            msg_bits = [ message[:l], message[l:2*l], message[2*l:] ]

        for c in range(n_channels):
            quant = jpg["quant_tables"][0]

            if c > 2:
                quant = jpg["quant_tables"][1]

            cost = self.cost_fn(jpg["coef_arrays"][c], I[:,:,c], quant)
            costs_m1, costs_p1 = self.cost_polarization(
                          cost, jpg["coef_arrays"][c], 
                          spatial_uncompress[c], quant)

            jpg["coef_arrays"][c] = stego.hide(msg_bits[c], jpg["coef_arrays"][c], 
                          costs_m1, costs_p1, password_seed, mx=1016, mn=-1016)

        jpeg_save(jpg, output_img_path)



    def extract(self, stego_img_path, password, output_msg_path):

        I = imageio.imread(stego_img_path)
        jpg = jpeg_load(stego_img_path)
       
        n_channels = 3
        if len(I.shape) == 2:
            n_channels = 1
            I = I[..., np.newaxis]

        cipher = Cipher(password)
        stego = Stego()

        # Seed from password
        password_hash = hashlib.sha256(password.encode()).digest()
        password_seed = int.from_bytes(password_hash, 'big')

        ciphertext = []
        for c in range(n_channels):
            ciphertext += stego.unhide(jpg["coef_arrays"][c], password_seed)

        plain = cipher.decrypt(bytes(ciphertext))

        f = open(output_msg_path, 'wb')
        f.write(plain)
        f.close()
        
    # }}}




