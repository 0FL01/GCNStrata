#!/usr/bin/env python3
"""Check frozen package identities and compile-only wrapper invariants without GPUs."""
import hashlib,json,pathlib,re,subprocess
root=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 'qsa_select.fit17.cu':'5fc5d78c6155fa0987dee751a875ab714516c88209b0938db17699061cc7de1a',
 'qsa_select.radix-stop.cu':'587c02f9a76916605be595bbf8cdf4ce72ac04edbade4c80686772e72903064c',
 'qsa_select.radix-stop.cu.patch':'694531ccfcfe3da6017540af121bb774bdf0d0896b0112a6e4dcd67a0491d38b',
 'provenance/cpu-source-proof.json':'1df9a9cdf2133d90880027b9f5f7352549a7834e5daa2284185e4a858c4f683c',
 'provenance/cpu-dispatch-proof.json':'ce9023f7c3b69f362eaa11db717cc2509b86ab3d7a63fab5113054843b1eef53',
}
for name,digest in expected.items():assert sha(root/name)==digest,name
host=(root/'build-host.sh').read_text()
inner=(root/'build-inner.sh').read_text()
old=(root/'provenance/wide51-build-inner.sh').read_text()
assert (root/'build-host.sh').read_bytes()==(root/'provenance/wide51-build-host.sh').read_bytes()
common=lambda s:re.search(r'^COMMON=\(.*\)$',s,re.M).group(0)
assert common(inner)==common(old)
assert 'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646' in host
assert '--network none' in host
command=next(x for x in host.splitlines() if x.startswith('docker run '))
assert '--device' not in command and '--privileged' not in command
assert '"$INPUT:/input:ro"' in command and '"$WORK:/work"' in command
assert 'exec 9>/tmp/strata-direct-ff34-ce.lock' in host and 'flock -n 9' in host
assert 'test ! -e /dev/kfd' in inner and 'test ! -e /dev/dri' in inner
assert 'for v in fit17 radix-stop; do' in inner
assert '-Rpass-analysis=kernel-resource-usage -save-temps=obj -c' in inner
assert 'exit "$gate_rc"' in inner
for name in ['build-host.sh','build-inner.sh']:
 subprocess.run(['bash','-n',str(root/name)],check=True)
headers={}
for line in (root/'provenance/wide51-INPUTS.sha256').read_text().splitlines():
 digest,name=line.split(None,1);name=name.lstrip('*')
 if name.startswith('./'):name=name[2:]
 if name.startswith('include/'):
  assert sha(root/name)==digest,name
  headers[name]=digest
assert len(headers)==6
print(json.dumps({'passed':True,'gpu_executed':False,'candidate_changed':False,'pinned_inputs':expected,'headers':headers,'host_wrapper_byte_identical':True,'compiler_common_flags_byte_identical':True,'no_gpu_mounts':True,'network_none':True,'engineering_lock_unchanged':True},indent=2))
