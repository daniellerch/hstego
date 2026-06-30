#!/bin/bash
set -euo pipefail

# Run from the repository root:
#   ./hstego-snap/prepare.sh
#
# Requirements:
#   sudo snap install snapcraft --classic
#   pyinstaller hstego-linux.spec

mkdir -p hstego-snap/snap/local/bin
cp dist/hstego-0.6-linux.x86_64 hstego-snap/snap/local/bin/hstego
chmod 755 hstego-snap/snap/local/bin/hstego

cd hstego-snap
snapcraft pack

# Install locally for testing:
#   sudo snap install hstego_0.6_amd64.snap --dangerous
