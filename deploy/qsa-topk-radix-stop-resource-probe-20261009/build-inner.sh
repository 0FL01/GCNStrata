#!/usr/bin/env bash
# Resource compilation and CPU checks only. Never executes the selector or a GPU image.
set -euo pipefail
cd /work
BIN=/opt/rocm/lib/llvm/bin
CXX=$BIN/clang++
COMMON=(-O3 -DNDEBUG -std=c++20 --offload-arch=gfx906 -Wall -Wextra -DSTRATA_HIP_GFX906=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -I/input/include/strata/platform/hip_compat -I/input/include)
(cd /input; sha256sum -c INPUTS.sha256)
test ! -e /dev/kfd
test ! -e /dev/dri
"$CXX" --version > compiler.txt
grep -Eq 'clang version 23\.' compiler.txt
printf '%q ' "$CXX" -x hip "${COMMON[@]}" > compiler-flags.txt
printf '\n' >> compiler-flags.txt
python3 /input/test-package.py > package-checks.json
python3 /input/test-source-cpu.py --cxx "$CXX" --output /work/source-check > source-test.stdout 2> source-test.stderr
python3 /input/test-dispatch.py --cxx "$CXX" --output /work/dispatch-check > dispatch-test.stdout 2> dispatch-test.stderr
python3 /input/test-resource-gate.py > gate-test.stdout 2> gate-test.stderr
cp /input/qsa_select.fit17.cu /input/qsa_select.radix-stop.cu .
for v in fit17 radix-stop; do
 "$CXX" -x hip "${COMMON[@]}" --cuda-device-only -E -dD "qsa_select.$v.cu" -o "$v.device.ii"
 "$CXX" -x hip "${COMMON[@]}" -Rpass-analysis=kernel-resource-usage -save-temps=obj -c "qsa_select.$v.cu" -o "qsa_select.$v.o" > "$v.compile.stdout" 2> "$v.resources.log"
 "$BIN/llvm-nm" --defined-only --demangle "qsa_select.$v.o" > "$v.defined.txt"
 DEV="qsa_select.$v-hip-amdgcn-amd-amdhsa-gfx906.out"
 ASM="qsa_select.$v-hip-amdgcn-amd-amdhsa-gfx906.s"
 test -s "$DEV"
 test -s "$ASM"
 "$BIN/llvm-readelf" --file-header --notes "$DEV" > "$v.device.notes"
 "$BIN/llvm-objdump" --disassemble --mcpu=gfx906 "$DEV" > "$v.device.disasm"
 "$BIN/llvm-nm" --defined-only --print-size "$DEV" > "$v.device.symbols"
done
python3 - <<'PY'
import pathlib,re,json
proof={}
for variant in ['fit17','radix-stop']:
 s=pathlib.Path(variant+'.device.ii').read_text()
 assert '/input/include/strata/platform/hip_compat/strata_hip.h' in s
 assert '/strata/hip_compat/intrinsics.hpp' not in s
 b=re.findall(r'^#define __ballot_sync[^\n]*',s,re.M);assert b and 'strata_ballot32' in b[-1]
 assert re.search(r'b\s*>>\s*\(\s*__lane_id\s*\(\s*\)\s*&\s*32u',s)
 assert re.search(r'^#define __gfx906__ 1$',s,re.M)
 assert re.search(r'^#define __HIP_DEVICE_COMPILE__ 1$',s,re.M)
 has17='block_topk_reg_kernel<17, true, 3>' in s
 has66='block_topk_reg_kernel<TK_PER_MAX, true, 4>' in s
 assert has17==has66==(variant=='radix-stop')
 proof[variant]={'platform_shim':True,'half_wave_ballot':True,'gfx906':True,'device_compile':True,'contains_radix_stop_reg17':has17,'contains_radix_stop_wide66':has66,'ballot_macro':b[-1]}
pathlib.Path('preprocessor-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
PY
set +e
python3 /input/check-resources.py /work > resource-gate.stdout 2> resource-gate.stderr
gate_rc=$?
set -e
find . -maxdepth 1 -type f -name '*gfx906*' -print | sort > device-files.txt
find . -maxdepth 1 -type f -name '*.isa.txt' -print | sort >> device-files.txt
python3 - "$gate_rc" <<'PY'
import pathlib,json,hashlib,sys
p=pathlib.Path('.')
files=[x for x in p.rglob('*') if x.is_file() and x.name not in ['build.receipt.json','build.stdout','build.stderr']]
r={'compiled':True,'resource_isa_gate_passed':int(sys.argv[1])==0,'gpu_executed':False,'gpu_runtime_loaded':False,'full_engine_built':False,'support_archive_used':False,'manual_isa_review_required_before_any_gpu':True,'performance_claim':False,'input_manifest_sha256':hashlib.sha256(pathlib.Path('/input/INPUTS.sha256').read_bytes()).hexdigest(),'files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(files)}}
pathlib.Path('build.receipt.json').write_text(json.dumps(r,indent=2)+'\n')
PY
tar -czf resource-isa-receipts.tar.gz --files-from=device-files.txt fit17.resources.log radix-stop.resources.log fit17.device.notes radix-stop.device.notes fit17.device.disasm radix-stop.device.disasm fit17.device.symbols radix-stop.device.symbols compiler.txt compiler-flags.txt preprocessor-proof.json source-check/source-proof.json dispatch-check/dispatch-proof.json gate-test.stdout gate-test.stderr resource-gate.json resource-gate.stdout resource-gate.stderr build.receipt.json
if (( gate_rc != 0 )); then echo 'STOP: resource/ISA gate failed. Preserve all outputs; do not relax the gate. No GPU qualification is authorized by this script.' >&2; fi
exit "$gate_rc"
