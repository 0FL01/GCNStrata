#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
docker run --rm --name strata-build --entrypoint /bin/sh -v "$PWD:/src" -w /src sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646 -c '
set -eu
cmake -S . -B build-906 -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/src/third_party/llama.cpp -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=ON
cmake --build build-906 --target strata -j4
'
