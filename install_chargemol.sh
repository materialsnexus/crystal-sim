#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${1:-${ROOT_DIR}/chargemol}"
FC="${FC:-gfortran}"
CHARGEMOL_URL="${CHARGEMOL_URL:-https://github.com/berquist/chargemol/archive/refs/heads/master.tar.gz}"

if [[ "$DEST_DIR" == --help || "$DEST_DIR" == -h ]]; then
  cat <<'EOF'
Usage: ./install_chargemol.sh [INSTALL_DIRECTORY]
Default: ./chargemol beside this script. Existing directories are not overwritten.

Requires curl, tar, cmake, make, and gfortran.
Ubuntu/Debian: sudo apt install build-essential gfortran cmake curl

Examples:
  ./install_chargemol.sh
  ./install_chargemol.sh "$HOME/chargemol"
  FC=/path/to/gfortran ./install_chargemol.sh

Override CHARGEMOL_URL to use a different source archive.
After installation, set [programs].chargemol_dir in crystal.toml to the printed
path and restart your notebook kernel. No shell setup or run helpers are needed.
EOF
  exit 0
fi

if (($# > 1)) || [[ -z "$DEST_DIR" || "$DEST_DIR" == -* ]]; then
  echo 'Usage: ./install_chargemol.sh [INSTALL_DIRECTORY]' >&2
  exit 1
fi
if [[ -e "$DEST_DIR" || -L "$DEST_DIR" ]]; then
  echo "Destination already exists: $DEST_DIR. Choose another path." >&2
  exit 1
fi
for tool in curl tar cmake make install "$FC"; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 1; }
done

mkdir -p "$DEST_DIR"
DEST_DIR="$(cd -- "$DEST_DIR" && pwd)"
curl -fL "$CHARGEMOL_URL" -o "$DEST_DIR/source.tar.gz"
mkdir "$DEST_DIR/src"
tar -xzf "$DEST_DIR/source.tar.gz" -C "$DEST_DIR/src" --strip-components=1
test -d "$DEST_DIR/src/atomic_densities"
cmake -S "$DEST_DIR/src" -B "$DEST_DIR/build" -G 'Unix Makefiles' \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER="$FC" -DENABLE_OPENMP=ON
cmake --build "$DEST_DIR/build" --target chargemol.x --parallel 2
mkdir "$DEST_DIR/bin"
install -m 0755 "$DEST_DIR/build/chargemol.x" "$DEST_DIR/bin/chargemol"
cp -R "$DEST_DIR/src/atomic_densities" "$DEST_DIR/atomic_densities"
printf '\nInstalled Chargemol. Set under [programs] in crystal.toml:\n'
printf 'chargemol_dir = "%s"\n' "$DEST_DIR"
