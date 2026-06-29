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
import zlib
import warnings
import importlib.machinery

import scipy.signal
import scipy.fftpack
import scipy.ndimage
import numpy as np

from ctypes import *
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


from numba import jit

warnings.filterwarnings('ignore', category=RuntimeWarning)


SPATIAL_EXT = ["png", "pgm", "tif"]
MAX_PAYLOAD=0.05
INF = 2**31-1

# scrypt memory ~= 128 * n * r bytes.
# 2**18, r=8 => ~256 MiB per password derivation.
SCRYPT_N = 2**18
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 512 * 1024 * 1024
PASSWORD_SEED_SALT = b"HStego password seed"
HEADER_KEY_SALT = b"HStego header key v2"
HEADER_MAGIC = b"HS2\x00"
HEADER_PLAINTEXT_SIZE = 8
HEADER_SIZE = AES.block_size + AES.block_size + HEADER_PLAINTEXT_SIZE
HEADER_COVER_LEN = HEADER_SIZE * 8 * 2
CIPHER_MAGIC = b"HC1\x00"
CIPHER_HEADER_SIZE = 12
MAX_DECOMPRESSED_SIZE = 64 * 1024 * 1024


def _module_base_dir():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.abspath(sys._MEIPASS)
    return os.path.abspath(os.path.dirname(__file__))


def _native_library_patterns(module_name):
    patterns = [
        module_name + suffix
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ]
    patterns.extend([
        module_name + '*.so',
        module_name + '*.pyd',
        module_name + '*.dll',
        module_name + '*.dylib',
    ])
    return patterns


def _find_native_library(module_name):
    base = _module_base_dir()
    patterns = _native_library_patterns(module_name)
    search_dirs = [base]

    if not getattr(sys, 'frozen', False):
        search_dirs.extend(glob.glob(os.path.join(base, 'build', 'lib.*')))

    for search_dir in search_dirs:
        for pattern in patterns:
            candidates = sorted(glob.glob(os.path.join(search_dir, pattern)))
            if candidates:
                return candidates[0]
    return None


def _load_native_library(module_name, label):
    path = _find_native_library(module_name)
    if not path:
        print(label, "library not found:", _module_base_dir())
        sys.exit(1)
    return CDLL(path)


jpeg = _load_native_library(
    'hstego_jpeg_toolbox_extension', "JPEG Toolbox")
stc = _load_native_library(
    'hstego_stc_extension', "STC")


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

# {{{ jpg_channel_capacity()
def jpg_channel_capacity(coef_array):
    """ channel capacity in bytes """
    nz_coeff = np.count_nonzero(coef_array)
    capacity = int((nz_coeff*MAX_PAYLOAD)/8)
    f = capacity // 16
    capacity = (f-1)*16
    capacity -= 16+16+4 # data for header
    if capacity<0:
        capacity = 0
    return capacity
# }}}

# {{{ effective_payload_capacities()
def effective_payload_capacities(channel_capacities):
    """Channel payload capacities after reserving the global header."""
    capacities = list(channel_capacities)
    if capacities:
        capacities[0] = max(0, capacities[0] - HEADER_SIZE)
    return capacities
# }}}

# {{{ jpg_capacity()
def jpg_capacity(jpg):
    """ image capacity in bytes """
    channel_capacities = [
        jpg_channel_capacity(jpg["coef_arrays"][channel])
        for channel in range(len(jpg["coef_arrays"]))
    ]
    return sum(effective_payload_capacities(channel_capacities))
# }}}

# {{{ spatial_channel_capacity()
def spatial_channel_capacity(channel):
    m = 1
    for i in channel.shape:
        m *= i

    capacity = int((m*MAX_PAYLOAD)/8)
    f = capacity // 16
    capacity = (f-1)*16
    capacity -= 16+16+4 # data for header
    if capacity<0:
        capacity = 0

    return capacity
# }}}

# {{{ spatial_capacity()
def spatial_capacity(I):
    if len(I.shape) == 2:
        channel_capacities = [spatial_channel_capacity(I)]
    else:
        channel_capacities = [
            spatial_channel_capacity(I[:, :, channel])
            for channel in range(I.shape[2])
        ]
    return sum(effective_payload_capacities(channel_capacities))
# }}} 



class Cipher:
    # {{{
    def __init__(self, password):
        self.password = password

    def pack_plaintext(self, data):
        if len(data) > MAX_DECOMPRESSED_SIZE:
            raise ValueError("Message exceeds maximum size")
        compressed = zlib.compress(data, level=9)
        return CIPHER_MAGIC + struct.pack("!Q", len(data)) + compressed

    def unpack_plaintext(self, data):
        if len(data) < CIPHER_HEADER_SIZE:
            raise ValueError("Invalid payload")
        magic = data[:4]
        plaintext_size = struct.unpack("!Q", data[4:CIPHER_HEADER_SIZE])[0]
        if magic != CIPHER_MAGIC:
            raise ValueError("Invalid payload")
        if plaintext_size > MAX_DECOMPRESSED_SIZE:
            raise ValueError("Message exceeds maximum size")

        decompressor = zlib.decompressobj()
        plaintext = decompressor.decompress(
            data[CIPHER_HEADER_SIZE:], plaintext_size + 1)
        if len(plaintext) != plaintext_size or not decompressor.eof:
            raise ValueError("Invalid compressed payload")
        return plaintext

    def open_file(self, path):
        with open(path, 'rb') as f:
            self.plaintext = self.pack_plaintext(f.read())

    def aes_encrypt(self):
        salt = get_random_bytes(AES.block_size)

        # use the Scrypt KDF to get a private key from the password
        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt,
            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
            maxmem=SCRYPT_MAXMEM, dklen=32)

        cipher = AES.new(private_key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(self.plaintext)

        # Ciphertext with salt + nonce + authentication tag.
        self.ciphertext = salt + cipher.nonce + tag + ciphertext

    def encrypt(self, input_path):
        """ encrypt file content """
        self.open_file(input_path)
        self.aes_encrypt()
        return self.ciphertext
  
    def aes_decrypt(self):
        salt = self.ciphertext[:AES.block_size]
        nonce = self.ciphertext[AES.block_size:AES.block_size*2]
        tag = self.ciphertext[AES.block_size*2:AES.block_size*3]
        ciphertext = self.ciphertext[AES.block_size*3:]

        private_key = hashlib.scrypt(
            self.password.encode(), salt=salt,
            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
            maxmem=SCRYPT_MAXMEM, dklen=32)

        cipher = AES.new(private_key, AES.MODE_EAX, nonce=nonce)
        decrypted = cipher.decrypt_and_verify(ciphertext, tag)
        self.decrypted = self.unpack_plaintext(decrypted)

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


def derive_password_seed(password):
    password_hash = hashlib.scrypt(
        password.encode(), salt=PASSWORD_SEED_SALT,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM, dklen=32)
    return int.from_bytes(password_hash, 'big')


def derive_header_key(password):
    return hashlib.scrypt(
        password.encode(), salt=HEADER_KEY_SALT,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM, dklen=32)


def encrypt_header(password, payload_len):
    key = derive_header_key(password)
    nonce = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = struct.pack("!4sI", HEADER_MAGIC, payload_len)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + tag + ciphertext


def decrypt_header(password, header):
    if len(header) != HEADER_SIZE:
        raise ValueError("Invalid header size")
    nonce = header[:AES.block_size]
    tag = header[AES.block_size:AES.block_size*2]
    ciphertext = header[AES.block_size*2:]
    cipher = AES.new(derive_header_key(password), AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    magic, payload_len = struct.unpack("!4sI", plaintext)
    if magic != HEADER_MAGIC:
        raise ValueError("Invalid header")
    return payload_len


def split_lengths_by_capacity(total_len, capacities):
    total_capacity = sum(capacities)
    if total_capacity <= 0:
        raise ValueError("No usable channel capacity")
    if total_len > total_capacity:
        raise ValueError("Message exceeds channel capacity")

    lengths = [
        (total_len * capacity) // total_capacity
        for capacity in capacities
    ]
    remainder = total_len - sum(lengths)

    fractions = sorted(
        range(len(capacities)),
        key=lambda i: ((total_len * capacities[i]) % total_capacity, capacities[i]),
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for i in fractions:
            if lengths[i] < capacities[i]:
                lengths[i] += 1
                remainder -= 1
                changed = True
                if remainder == 0:
                    break
        if not changed:
            raise ValueError("Message exceeds channel capacity")
    return lengths


def split_by_capacity(message, capacities):
    lengths = split_lengths_by_capacity(len(message), capacities)
    chunks = []
    offset = 0
    for length in lengths:
        chunks.append(message[offset:offset + length])
        offset += length
    return chunks


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


    def hide_raw(self, message, cover_matrix, cost_matrix_m1, cost_matrix_p1,
                 password_seed, reserved_prefix=0, mx=255, mn=0):
        random.seed(password_seed)

        message_bits = self.bytes_to_bits(message)
        height, width = cover_matrix.shape
        cover_array = cover_matrix.reshape((height*width,))
        costs_array_m1 = cost_matrix_m1.reshape((height*width,))
        costs_array_p1 = cost_matrix_p1.reshape((height*width,))

        if reserved_prefix < 0 or reserved_prefix > len(cover_array):
            raise ValueError("Invalid reserved prefix")
        if len(message_bits) > len(cover_array) - reserved_prefix:
            raise ValueError("Message exceeds channel capacity")

        indices = list(range(len(cover_array)))
        random.shuffle(indices)
        cover_array = cover_array[indices]
        costs_array_m1 = costs_array_m1[indices]
        costs_array_p1 = costs_array_p1[indices]

        if len(message_bits) == 0:
            stego_array = cover_array.copy()
        else:
            prefix = cover_array[:reserved_prefix]
            stego_payload = self.hide_stc(
                cover_array[reserved_prefix:],
                costs_array_m1[reserved_prefix:],
                costs_array_p1[reserved_prefix:],
                message_bits, mx, mn)
            stego_array = np.hstack((prefix, stego_payload))

        stego_array[indices] = stego_array[list(range(len(cover_array)))]
        return stego_array.reshape((height, width))


    def hide_header_and_raw(self, header, message, cover_matrix,
                            cost_matrix_m1, cost_matrix_p1,
                            password_seed, mx=255, mn=0):
        random.seed(password_seed)

        header_bits = self.bytes_to_bits(header)
        message_bits = self.bytes_to_bits(message)
        height, width = cover_matrix.shape
        cover_array = cover_matrix.reshape((height*width,))
        costs_array_m1 = cost_matrix_m1.reshape((height*width,))
        costs_array_p1 = cost_matrix_p1.reshape((height*width,))

        if len(header) != HEADER_SIZE:
            raise ValueError("Invalid header size")
        if HEADER_COVER_LEN > len(cover_array):
            raise ValueError("No usable header capacity")
        if len(message_bits) > len(cover_array) - HEADER_COVER_LEN:
            raise ValueError("Message exceeds channel capacity")

        indices = list(range(len(cover_array)))
        random.shuffle(indices)
        cover_array = cover_array[indices]
        costs_array_m1 = costs_array_m1[indices]
        costs_array_p1 = costs_array_p1[indices]

        stego_header = self.hide_stc(
            cover_array[:HEADER_COVER_LEN],
            costs_array_m1[:HEADER_COVER_LEN],
            costs_array_p1[:HEADER_COVER_LEN],
            header_bits, mx, mn)

        if len(message_bits) == 0:
            stego_payload = cover_array[HEADER_COVER_LEN:].copy()
        else:
            stego_payload = self.hide_stc(
                cover_array[HEADER_COVER_LEN:],
                costs_array_m1[HEADER_COVER_LEN:],
                costs_array_p1[HEADER_COVER_LEN:],
                message_bits, mx, mn)

        stego_array = np.hstack((stego_header, stego_payload))
        stego_array[indices] = stego_array[list(range(len(cover_array)))]
        return stego_array.reshape((height, width))


    def unhide_raw(self, stego_matrix, password_seed, byte_len,
                   reserved_prefix=0):
        random.seed(password_seed)
        height, width = stego_matrix.shape
        stego_array = stego_matrix.reshape((height*width,))

        if byte_len < 0:
            return bytearray()
        if reserved_prefix < 0 or reserved_prefix > len(stego_array):
            return bytearray()
        if byte_len * 8 > len(stego_array) - reserved_prefix:
            return bytearray()

        indices = list(range(len(stego_array)))
        random.shuffle(indices)
        stego_array = stego_array[indices]
        return self.unhide_stc(stego_array[reserved_prefix:], byte_len * 8)


    def unhide_header(self, stego_matrix, password_seed):
        random.seed(password_seed)
        height, width = stego_matrix.shape
        stego_array = stego_matrix.reshape((height*width,))

        if HEADER_COVER_LEN > len(stego_array):
            return bytearray()

        indices = list(range(len(stego_array)))
        random.shuffle(indices)
        stego_array = stego_array[indices]
        return self.unhide_stc(stego_array[:HEADER_COVER_LEN], HEADER_SIZE * 8)


    def unhide_stc(self, stego_array, message_len):

        if message_len < 0 or message_len > len(stego_array):
            print("WARNING, inconsistent message length:",
                  len(stego_array), "<", message_len)
            return bytearray()

        if message_len == 0:
            return bytes()

        stego = (c_int*(len(stego_array)))()
        for i in range(len(stego_array)):
            stego[i] = int(stego_array[i])

        extracted_message = (c_ubyte*message_len)()
        s = stc.stc_unhide(len(stego_array), stego, message_len, extracted_message)

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

        if len(stego_array) < 64:
            print("WARNING: message not found")
            return bytearray()

        # Extract a 32-bits message length from a 64-pixel array
        data = self.unhide_stc(stego_array[:64], 32)
        if len(data) != 4:
            print("WARNING: message not found")
            return bytearray()

        data_len = struct.unpack_from("!I", data, 0)[0]
        if data_len % 8 != 0 or data_len > len(stego_array[64:]):
            print("WARNING, inconsistent message length:",
                  len(stego_array[64:]), "<", data_len)
            return bytearray()
        
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

        I = imageio.imread(input_img_path)
        
        n_channels = 3
        if len(I.shape) == 2:
            n_channels = 1
            I = I[..., np.newaxis]

        cipher = Cipher(password)
        message = cipher.encrypt(msg_file_path)
        header = encrypt_header(password, len(message))

        # Seed from password
        password_seed = derive_password_seed(password)

        channel_capacities = [
            spatial_channel_capacity(I[:, :, channel])
            for channel in range(n_channels)
        ]
        effective_capacities = effective_payload_capacities(channel_capacities)
        capacity = sum(effective_capacities)
        if len(message) > capacity:
            print("ERROR, message too long:", len(message), ">", capacity)
            sys.exit(0)

        stego = Stego()
        msg_bits = split_by_capacity(message, effective_capacities)

        for c in range(n_channels):
            costs_m1, costs_p1 = self.cost_fn(I[:,:,c])
            if c == 0:
                I[:,:,c] = stego.hide_header_and_raw(
                    header, msg_bits[c], I[:,:,c], costs_m1, costs_p1,
                    password_seed)
            else:
                I[:,:,c] = stego.hide_raw(
                    msg_bits[c], I[:,:,c], costs_m1, costs_p1,
                    password_seed)

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
        password_seed = derive_password_seed(password)

        try:
            header = stego.unhide_header(I[:, :, 0], password_seed)
            payload_len = decrypt_header(password, header)
            channel_capacities = [
                spatial_channel_capacity(I[:, :, channel])
                for channel in range(n_channels)
            ]
            effective_capacities = channel_capacities.copy()
            effective_capacities[0] = max(0, effective_capacities[0] - HEADER_SIZE)
            chunk_lengths = split_lengths_by_capacity(
                payload_len, effective_capacities)

            ciphertext = bytearray()
            for c in range(n_channels):
                if c == 0:
                    ciphertext += stego.unhide_raw(
                        I[:, :, c], password_seed, chunk_lengths[c],
                        reserved_prefix=HEADER_COVER_LEN)
                else:
                    ciphertext += stego.unhide_raw(
                        I[:, :, c], password_seed, chunk_lengths[c])
            plain = cipher.decrypt(bytes(ciphertext))
        except Exception:
            print("WARNING: message not found")
            plain = bytearray()

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

        #I = imageio.imread(input_img_path)
        jpg = jpeg_load(input_img_path)
        if jpg["jpeg_components"] == 1:
            n_channels = 1
            I = uncompress(jpg["coef_arrays"][0], jpg["quant_tables"][0])
            I = I[..., np.newaxis]
            spatial_uncompress = (I[:, :, 0],)
        else:
            n_channels = 3
            Y = uncompress(jpg["coef_arrays"][0], jpg["quant_tables"][0])
            Cb = uncompress(jpg["coef_arrays"][1], jpg["quant_tables"][1])
            Cr = uncompress(jpg["coef_arrays"][2], jpg["quant_tables"][1])
            spatial_uncompress = (Y, Cb, Cr)
            I = YCbCr_to_RGB(Y, Cb, Cr)

        cipher = Cipher(password)
        message = cipher.encrypt(msg_file_path)
        header = encrypt_header(password, len(message))

        # Seed from password
        password_seed = derive_password_seed(password)

        # Real capacity, without headers
        channel_capacities = [
            jpg_channel_capacity(jpg["coef_arrays"][channel])
            for channel in range(n_channels)
        ]
        effective_capacities = effective_payload_capacities(channel_capacities)
        capacity = sum(effective_capacities)

        if len(message) > capacity:
            print("ERROR, message too long:", len(message), ">", capacity)
            sys.exit(0)


        stego = Stego()
        msg_bits = split_by_capacity(message, effective_capacities)

        for c in range(n_channels):
            quant = jpg["quant_tables"][0]

            if c > 0:
                quant = jpg["quant_tables"][1]

            cost = self.cost_fn(jpg["coef_arrays"][c], I[:,:,c], quant)
            costs_m1, costs_p1 = self.cost_polarization(
                          cost, jpg["coef_arrays"][c], 
                          spatial_uncompress[c], quant)

            if c == 0:
                jpg["coef_arrays"][c] = stego.hide_header_and_raw(
                          header, msg_bits[c], jpg["coef_arrays"][c],
                          costs_m1, costs_p1, password_seed,
                          mx=1016, mn=-1016)
            else:
                jpg["coef_arrays"][c] = stego.hide_raw(
                          msg_bits[c], jpg["coef_arrays"][c],
                          costs_m1, costs_p1, password_seed,
                          mx=1016, mn=-1016)

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
        password_seed = derive_password_seed(password)

        try:
            header = stego.unhide_header(jpg["coef_arrays"][0], password_seed)
            payload_len = decrypt_header(password, header)
            channel_capacities = [
                jpg_channel_capacity(jpg["coef_arrays"][channel])
                for channel in range(n_channels)
            ]
            effective_capacities = channel_capacities.copy()
            effective_capacities[0] = max(0, effective_capacities[0] - HEADER_SIZE)
            chunk_lengths = split_lengths_by_capacity(
                payload_len, effective_capacities)

            ciphertext = bytearray()
            for c in range(n_channels):
                if c == 0:
                    ciphertext += stego.unhide_raw(
                        jpg["coef_arrays"][c], password_seed,
                        chunk_lengths[c], reserved_prefix=HEADER_COVER_LEN)
                else:
                    ciphertext += stego.unhide_raw(
                        jpg["coef_arrays"][c], password_seed, chunk_lengths[c])
            plain = cipher.decrypt(bytes(ciphertext))
        except Exception:
            print("WARNING: message not found")
            plain = bytearray()

        f = open(output_msg_path, 'wb')
        f.write(plain)
        f.close()
        
    # }}}

