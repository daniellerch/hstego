#!/usr/bin/env python3

import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hstegolib


def data_for(size):
    rng = random.Random(0x48535445474f)
    return bytes(rng.randrange(0, 256) for _ in range(size))


def payload_len(data):
    # Cipher payload length is compressed data plus salt, nonce and tag.
    return len(zlib.compress(data, level=9)) + 3 * hstegolib.AES.block_size


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <capacity-bytes> <output-secret>")

    capacity = int(sys.argv[1])
    output_secret = Path(sys.argv[2])

    # Keep a tiny margin so STC does not operate exactly at the declared limit.
    target_ciphertext_len = max(1, capacity - 8)

    lo = 0
    hi = target_ciphertext_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if payload_len(data_for(mid)) <= target_ciphertext_len:
            lo = mid
        else:
            hi = mid - 1

    payload = data_for(lo)
    ciphertext_len = payload_len(payload)
    slack = capacity - ciphertext_len
    if ciphertext_len > capacity:
        raise SystemExit(
            f"Generated payload exceeds capacity: {ciphertext_len} > {capacity}")
    if slack > max(32, capacity // 20):
        raise SystemExit(
            f"Generated payload is not close enough to capacity: "
            f"slack={slack}, capacity={capacity}")

    output_secret.write_bytes(payload)
    print(
        f"Near-capacity secret: plaintext={len(payload)} bytes, "
        f"ciphertext={ciphertext_len} bytes, capacity={capacity} bytes")


if __name__ == "__main__":
    main()
