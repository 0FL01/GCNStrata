#!/usr/bin/env bash
set -euo pipefail
cd /work
BIN=/opt/rocm/lib/llvm/bin
CXX=$BIN/clang++
COMMON=(-O3 -DNDEBUG -std=c++20 --offload-arch=gfx906 -Wall -Wextra -DSTRATA_HIP_GFX906=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -I/input/include/strata/platform/hip_compat -I/input/include)
(cd /input; sha256sum -c INPUTS.sha256)
"$CXX" --version > compiler.txt
cp /input/qsa_select.base.cu /input/qsa_select.fit17.cu .
"$CXX" -x hip "${COMMON[@]}" --cuda-device-only -E -dD qsa_select.fit17.cu -o candidate.device.ii
python3 - <<'PY'
import pathlib,re,json
s=pathlib.Path('candidate.device.ii').read_text()
assert '/input/include/strata/platform/hip_compat/strata_hip.h' in s
assert '/strata/hip_compat/intrinsics.hpp' not in s
b=re.findall(r'^#define __ballot_sync[^\n]*',s,re.M);assert b and 'strata_ballot32' in b[-1]
assert re.search(r'b\s*>>\s*\(\s*__lane_id\s*\(\s*\)\s*&\s*32u',s)
pathlib.Path('preprocessor-proof.json').write_text(json.dumps({'platform_shim':True,'half_wave_ballot':True,'ballot_macro':b[-1]},indent=2)+'\n')
PY
for v in base fit17; do
 "$CXX" -x hip "${COMMON[@]}" -Rpass-analysis=kernel-resource-usage -save-temps=obj -c "qsa_select.$v.cu" -o "qsa_select.$v.o" > "$v.compile.stdout" 2> "$v.resources.log"
 "$BIN/llvm-nm" --defined-only --demangle "qsa_select.$v.o" > "$v.defined.txt"
done
for t in public-parity capture transition; do
 case "$t" in
 public-parity) src=/input/src/kernels/qsa_topk_parity.cpp;;
 capture) src=/input/external-tests/qsa_topk_fixture_replay.cpp;;
 transition) src=/input/external-tests/qsa_reg17_transition.cpp;;
 esac
 "$CXX" -x hip "${COMMON[@]}" -c "$src" -o "$t.o" > "$t.compile.stdout" 2> "$t.compile.stderr"
 for v in base fit17; do
  "$CXX" --hip-link --offload-arch=gfx906 "qsa_select.$v.o" "$t.o" -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -Wl,-Map="$t.$v.map" -o "$t.$v" > "$t.$v.link.stdout" 2> "$t.$v.link.stderr"
  "$BIN/llvm-nm" --defined-only --demangle "$t.$v" > "$t.$v.defined.txt"
  test "$(grep -Ec ' [TW] strata::kernels::qsa_block_topk\(' "$t.$v.defined.txt")" -eq 1
  test "$(grep -Ec ' [TW] strata::kernels::qsa_block_topk_ref\(' "$t.$v.defined.txt")" -eq 1
  if grep -q 'libstrata_kernels' "$t.$v.map"; then exit 3; fi
 done
done
# save-temps retains standalone gfx906 device images and assembly for BOTH selector objects.
find . -maxdepth 1 -type f -name '*gfx906*' -print | sort > device-files.txt
test "$(grep -c 'qsa_select.base.*\.s$' device-files.txt)" -ge 1
test "$(grep -c 'qsa_select.fit17.*\.s$' device-files.txt)" -ge 1
python3 - <<'PY'
import pathlib,json,hashlib
p=pathlib.Path('.')
names=[x for x in p.iterdir() if x.is_file() and x.name!='build.json']
r={'passed':True,'gpu_executed':False,'support_archive_used':False,'input_manifest_sha256':hashlib.sha256(pathlib.Path('/input/INPUTS.sha256').read_bytes()).hexdigest(),'files':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(names)}}
pathlib.Path('build.json').write_text(json.dumps(r,indent=2)+'\n')
PY
tar -czf isa-resources.tar.gz --files-from=device-files.txt base.resources.log fit17.resources.log compiler.txt preprocessor-proof.json
