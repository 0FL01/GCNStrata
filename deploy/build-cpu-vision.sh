#!/bin/sh
set -eu
cd /home/radneon/strata
docker run --rm --name strata-vision-cpu-build --entrypoint /bin/sh -v "$PWD:/src" sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646 -c 'set -eu
cmake -S /src/tools/vision -B /src/build-vision-cpu -DLLAMA_DIR=/src/third_party/llama.cpp -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DSTRATA_PORTABLE=ON -DSTRATA_VISION_CUDA=OFF -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_SYCL=OFF -DGGML_METAL=OFF -DGGML_RPC=OFF -DGGML_OPENCL=OFF
cmake --build /src/build-vision-cpu --target strata-vision -j4
ldd /src/build-vision-cpu/bin/strata-vision
'
