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
    local capacity

    capacity=$(./hstego.py capacity "$cover" --raw)
    testing/make_near_capacity_secret.py "$capacity" "$output_secret"
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
