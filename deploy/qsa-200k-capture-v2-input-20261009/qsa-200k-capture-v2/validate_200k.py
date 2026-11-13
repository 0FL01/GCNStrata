#!/usr/bin/env python3
"""Experiment gate over the unchanged QSDUMP01 parser. No GPU or model text export."""
import argparse, json, pathlib, struct
from analyze_qsa_dump import read_dump, analyze_file
BASELINE_SHA='c18316d678a669c309cb95e2f5de9af03fa5573256fd0e093aafc4b41000d049'
FIXTURE_SHA='736636e97ede94cc37e32447eefde4217a38bd17c81a7349345be98060e766c4'
STAGES={0:(0,27,23),1:(27,48,47)}

def need(ok,why):
    if not ok: raise ValueError(why)

def read_200k(path):
    path=pathlib.Path(path)
    need(not path.is_symlink() and path.is_file(),'regular fixture required')
    need(160<=path.stat().st_size<=2*1024*1024,'bounded 200K fixture size')
    with path.open('rb') as f:
        head=f.read(160)
    need(len(head)==160 and head[:8]==b'QSDUMP01','QSDUMP01 header required')
    x=struct.unpack('<19q',head[8:])
    need(x[0]==1 and x[1]==0x01020304,'format/endian')
    need(x[2] in STAGES and tuple(x[3:6])==STAGES[x[2]],'exact captured device/stage/layer')
    need(2<=x[7]<=6 and x[8]==6,'native 2..6 rows required')
    need(200000<=x[6] and x[6]+x[7]-1<=200263,'actual 200K native window required; old64K rejected')
    need(tuple(x[9:])==(51202,2051,4,4,2048,128,4,204808,8204,16),'exact native capacity/shape/strides')
    h,rows,sha,nbytes=read_dump(path)
    for row in rows:
        pos,nkv,nbid,width=row['step']
        need(200000<=pos<=200263 and nkv==pos+1,'bounded actual step')
        need(17408<nbid+1<=51202<=1024*51,'real wide-window/capacity eligibility')
        ids=row['ids']
        need(len(ids)==2051 and all(0<=v<nkv for v in ids) and all(a<b for a,b in zip(ids,ids[1:])), 'ascending unique valid IDs')
    return h,rows,sha,nbytes

def validate_pair(paths,manifest=None):
    need(len(paths)==2 and pathlib.Path(paths[0]).resolve()!=pathlib.Path(paths[1]).resolve(),'exactly two distinct stage fixtures')
    if manifest is not None:
        need(manifest.get('baseline_sha256')==BASELINE_SHA and manifest.get('fixture_sha256')==FIXTURE_SHA,'200K baseline/fixture provenance')
        need(manifest.get('prompt_tokens')==200000 and manifest.get('requested_outputs')==256,'native request provenance')
    parsed=[read_200k(p) for p in paths]
    need({x[0]['device'] for x in parsed}=={0,1},'one fixture per device')
    a,b=[x[0] for x in parsed]
    keys=['version','endian','pos0','T','max_t','max_blocks','cap','step_count','idx_block','idx_top_k','idx_dim','idx_heads','score_stride','ids_stride','step_stride']
    need(all(a[k]==b[k] for k in keys),'paired stage metadata differ')
    need([r['step'] for r in parsed[0][1]]==[r['step'] for r in parsed[1][1]],'paired stage steps differ')
    analyzed=[analyze_file(p) for p in paths] # Existing CPU exact-ID sort/radix check, unchanged.
    for x in analyzed: x['file']=pathlib.Path(x['file']).name
    return {'passed':True,'same_window':True,'files':analyzed,'gpu_executed':False,'timing_claim':False,
            'sample_scope':'two last-QSA-stage layers, not all model layers',
            'provenance_checked':manifest is not None,'old64k_accepted':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs=2);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
    result=validate_pair(a.files);text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
if __name__=='__main__':main()
