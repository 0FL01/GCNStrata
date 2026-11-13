#!/usr/bin/env python3
"""Read QSDUMP01 without FP conversion; check IDs and CPU-only weighted radix depths."""
import argparse, collections, hashlib, json, pathlib, struct

FIELDS = ('version endian device layer_begin layer_end layer pos0 T max_t max_blocks cap step_count '
          'idx_block idx_top_k idx_dim idx_heads score_stride ids_stride step_stride').split()

def read_dump(path):
    data = pathlib.Path(path).read_bytes()
    off = 0
    def read(fmt):
        nonlocal off
        n = struct.calcsize(fmt)
        if off + n > len(data):
            raise ValueError(f'truncated payload at byte {off}')
        v = struct.unpack_from(fmt, data, off); off += n
        return v
    if read('<8s')[0] != b'QSDUMP01':
        raise ValueError('bad magic')
    h = dict(zip(FIELDS, read('<19q')))
    if h['version'] != 1 or h['endian'] != 0x01020304:
        raise ValueError('unsupported version/endian')
    if not (2 <= h['T'] <= h['max_t'] <= 8 and 1 <= h['max_blocks'] <= 1000000 and
            0 < h['cap'] <= 1000000 and h['step_count'] == 4 and
            h['idx_block'] == 4 and h['idx_top_k'] == 2048 and
            h['idx_dim'] == 128 and h['idx_heads'] == 4 and h['pos0'] >= 65536 and
            0 <= h['layer_begin'] <= h['layer'] < h['layer_end'] <= 1024 and h['device'] >= 0):
        raise ValueError('unsupported or inconsistent capture geometry')
    if (h['score_stride'], h['ids_stride'], h['step_stride']) != (4*h['max_blocks'],4*h['cap'],16):
        raise ValueError('invalid source byte strides')
    steps = [read('<4i') for _ in range(h['T'])]
    rows=[]
    for t,(pos,nkv,nbid,width) in enumerate(steps):
        if (pos != h['pos0']+t or nkv != pos+1 or nbid != nkv//4 or
                not 0 <= nbid < h['max_blocks'] or width != min(nkv,2051) or width > h['cap']):
            raise ValueError(f'inconsistent steps in row {t}')
        active,w = read('<2q')
        if active != nbid+1 or w != width:
            raise ValueError(f'inconsistent packed lengths in row {t}')
        bits=read('<'+str(active)+'I')
        ids=read('<'+str(w)+'i')
        rows.append({'step':(pos,nkv,nbid,width),'bits':bits,'ids':ids})
    if off != len(data):
        raise ValueError(f'{len(data)-off} trailing bytes')
    return h,rows,hashlib.sha256(data).hexdigest(),len(data)

def order_key(bits):
    if (bits & 0x7fffffff) > 0x7f800000:
        return 0
    if (bits & 0x7fffffff) == 0:
        bits=0  # s + 0.0f normalizes both signed zeros
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits | 0x80000000

def emit(keys,weights,threshold,mask,width):
    # Ascending cells, selecting all keys above threshold and the lowest equal cells.
    gt=sum(w for k,w in zip(keys,weights) if (k & mask)>threshold)
    need=width-gt
    if need < 0:
        raise AssertionError('more cells strictly above boundary than width')
    out=[]
    for b,(k,w) in enumerate(zip(keys,weights)):
        if (k & mask)>threshold:
            out.extend(range(4*b,4*b+w))
        elif (k & mask)==threshold:
            take=min(w,need); out.extend(range(4*b,4*b+take)); need-=take
    if len(out)!=width or need!=0:
        raise AssertionError('incorrect threshold cardinality')
    return out

def analyze_row(row):
    pos,nkv,nbid,width=row['step']; keys=list(map(order_key,row['bits']))
    weights=[4]*nbid+[nkv-4*nbid]
    selected=[]
    for b in sorted(range(len(keys)),key=lambda b:(-keys[b],b)):
        take=min(weights[b],width-len(selected))
        selected.extend(range(4*b,4*b+take))
        if len(selected)==width: break
    selected.sort()
    captured=list(row['ids'])
    if selected!=captured:
        j=next((j for j,(a,b) in enumerate(zip(selected,captured)) if a!=b),0)
        raise AssertionError(f'CPU sort != captured IDs at pos {pos}, index {j}')
    prefix=above=0; first=None; passes=[]
    for p,shift in enumerate((24,16,8,0),1):
        hi_mask=(0xffffffff << (shift+8)) & 0xffffffff if shift!=24 else 0
        hist=[0]*256
        for k,w in zip(keys,weights):
            if w and (k & hi_mask)==(prefix & hi_mask): hist[(k>>shift)&255]+=w
        cum=above; digit=255
        while digit>0 and cum+hist[digit]<width:
            cum+=hist[digit]; digit-=1
        prefix |= digit<<shift; above=cum
        bucket=hist[digit]
        if not above < width <= above+bucket:
            raise AssertionError('weighted radix invariant failed')
        mask=(0xffffffff<<shift)&0xffffffff
        singleton=0<bucket<=4
        if singleton:
            if emit(keys,weights,prefix,mask,width)!=captured:
                raise AssertionError('singleton-prefix selection differs from captured IDs')
            if first is None: first=p
        passes.append({'pass':p,'shift':shift,'digit':digit,'above_cells':above,
                       'boundary_cells':bucket,'singleton':singleton})
    if emit(keys,weights,prefix,0xffffffff,width)!=captured:
        raise AssertionError('full radix != captured IDs')
    return {'pos':pos,'active_blocks':nbid+1,'tail_cells':nkv%4,'width':width,
            'ids_match':True,'first_singleton_pass':first,
            'removable_passes':4-first if first is not None else 0,'passes':passes}

def analyze_file(path):
    h,rows,sha,nbytes=read_dump(path)
    results=[analyze_row(row) for row in rows]
    counts=collections.Counter(str(r['first_singleton_pass']) if r['first_singleton_pass'] is not None else 'none'
                               for r in results)
    return {'file':str(path),'sha256':sha,'bytes':nbytes,'header':h,'all_ids_match':True,
            'first_singleton_pass_counts':dict(counts),'rows':results,
            'mean_removable_passes':sum(r['removable_passes'] for r in results)/len(results),
            'scope':'one last-QSA-layer fixture; pass counts are not time fractions or whole-model savings'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+');p.add_argument('--output')
    a=p.parse_args()
    result={'files':[analyze_file(f) for f in a.files],'gpu_executed':False,
            'timing_claim':False,'sample_scope':'Only captured last QSA layer of each supplied stage, not all 12 layers'}
    if len(result['files'])==2:
        hh=[r['header'] for r in result['files']]
        result['same_window']=all(hh[0][k]==hh[1][k] for k in ['T','pos0','max_blocks','cap'])
        if not result['same_window']: raise ValueError('stage fixtures are from different windows/geometries')
    s=json.dumps(result,indent=2)+'\n'
    if a.output: pathlib.Path(a.output).write_text(s)
    print(s,end='')
if __name__=='__main__': main()
