#!/bin/bash
set -euo pipefail

TMPDIR=$(mktemp -d)
cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

run_case() {
    local title="$1"
    local input_secret="$2"
    local cover="$3"
    local stego="$4"
    local output_secret="$5"

    echo "------------------------------------------------------------------------"
    echo "$title"
    echo "------------------------------------------------------------------------"

    echo -n "Embed: "
    time ./hstego.py embed "$input_secret" "$cover" "$stego" p4ssw0rd

    echo -n "Extract: "
    time ./hstego.py extract "$stego" "$output_secret" p4ssw0rd

    if ! cmp -s "$input_secret" "$output_secret"; then
        echo "ERROR: $title" >&2
        if [[ "$input_secret" == *.txt && "$output_secret" == *.txt ]]; then
            diff "$input_secret" "$output_secret" || true
        fi
        exit 1
    fi

    echo "OK: $title"
}

make_near_capacity_secret() {
    local cover="$1"
    local output_secret="$2"

    python3 - "$cover" "$output_secret" <<'PY'
import random
import sys
import zlib
from pathlib import Path

import imageio.v2 as imageio
import hstegolib

cover = sys.argv[1]
output_secret = Path(sys.argv[2])

if hstegolib.is_ext(cover, hstegolib.SPATIAL_EXT):
    image = imageio.imread(cover)
    capacity = hstegolib.spatial_capacity(image)
elif hstegolib.is_ext(cover, "jpg"):
    jpg = hstegolib.jpeg_load(cover)
    capacity = hstegolib.jpg_capacity(jpg)
else:
    raise SystemExit(f"Unsupported cover: {cover}")

# Keep a tiny margin so STC does not operate exactly at the declared limit.
target_ciphertext_len = max(1, capacity - 8)


def data_for(size):
    rng = random.Random(0x48535445474f)
    return bytes(rng.randrange(0, 256) for _ in range(size))


def payload_len(data):
    # Cipher payload length is compressed data plus salt, nonce and tag.
    return len(zlib.compress(data, level=9)) + 3 * hstegolib.AES.block_size

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
        f"Generated payload is not close enough to capacity: slack={slack}, capacity={capacity}")

output_secret.write_bytes(payload)
print(
    f"Near-capacity secret: plaintext={len(payload)} bytes, "
    f"ciphertext={ciphertext_len} bytes, capacity={capacity} bytes")
PY
}

run_case \
    "text + grayscale png" \
    testing/input-secret.txt \
    testing/cover.png \
    "$TMPDIR/stego.png" \
    "$TMPDIR/output-secret.txt"

run_case \
    "bin + grayscale png" \
    testing/input-secret.png \
    testing/cover.png \
    "$TMPDIR/stego-bin.png" \
    "$TMPDIR/output-secret.png"

run_case \
    "text + color png" \
    testing/input-secret.txt \
    testing/cover_color.png \
    "$TMPDIR/stego-color.png" \
    "$TMPDIR/output-secret-color.txt"

make_near_capacity_secret \
    testing/cover_color.png \
    "$TMPDIR/input-near-capacity-color.bin"

run_case \
    "near-capacity bin + color png" \
    "$TMPDIR/input-near-capacity-color.bin" \
    testing/cover_color.png \
    "$TMPDIR/stego-near-capacity-color.png" \
    "$TMPDIR/output-near-capacity-color.bin"

run_case \
    "text + grayscale jpg" \
    testing/input-secret-small.txt \
    testing/cover.jpg \
    "$TMPDIR/stego.jpg" \
    "$TMPDIR/output-secret-small.txt"

run_case \
    "text + color jpg" \
    testing/input-secret-small2.txt \
    testing/cover_color.jpg \
    "$TMPDIR/stego-color.jpg" \
    "$TMPDIR/output-secret-small2.txt"

make_near_capacity_secret \
    testing/cover_color.jpg \
    "$TMPDIR/input-near-capacity-color-jpg.bin"

run_case \
    "near-capacity bin + color jpg" \
    "$TMPDIR/input-near-capacity-color-jpg.bin" \
    testing/cover_color.jpg \
    "$TMPDIR/stego-near-capacity-color.jpg" \
    "$TMPDIR/output-near-capacity-color-jpg.bin"
