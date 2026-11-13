"""Read-only controller check of independently sealed raw baseline archives."""
import hashlib,json,re,tarfile
from pathlib import Path
import parse_counts
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def verify():
 proof=json.loads((HERE/'baseline-proof.json').read_text());actual=[]
 for entry in proof['receipts']:
  archive=HERE.parent/entry['archive'];assert sha(archive.read_bytes())==entry['archive_sha256']
  with tarfile.open(archive) as a:
   tag='hc-counts' if 'counts' in archive.name else 'hc-cost'
   def raw(suffix):
    matches=[m for m in a.getmembers() if m.name.endswith('/'+suffix)];assert len(matches)==1;return a.extractfile(matches[0]).read()
   stderr=raw('private/'+tag+'.stderr');stdout=raw('private/'+tag+'.stdout');assert sha(stderr)==entry['raw_stderr_sha256'] and sha(stdout)==entry['raw_stdout_sha256']
   assert json.loads(raw('private/'+tag+'.output-ids.json'))==[int(x.split()[1]) for x in stdout.splitlines() if x.startswith(b'T ')]==proof['output_ids']==entry['output_ids']
   routes=[list(map(int,x)) for x in re.findall(rb'^strata hc-f16 route: attempts=(\d+) taken=(\d+)$',stderr,re.M)];assert routes==entry['routes']==proof['routes']
   rows=[json.loads(x,object_pairs_hook=parse_counts.unique) for x in stderr.splitlines() if x.startswith(b'{"type":"hc_observer_')]
   for row in rows:row.pop('guard_observations',None)
   parsed=parse_counts.parse('\n'.join(json.dumps(r,separators=(',',':')) for r in rows))
   sums=[r for r in rows if r['type']=='hc_observer_summary'];dev={(s['context'],s['generation']):s['device'] for s in sums};weights=[r for r in rows if r['type']=='hc_observer_weight'];assert len(weights)==192
   assert [sum(dev[w['context'],w['generation']]==d for w in weights) for d in (0,1)]==[108,84]
   keys=[]
   for w in weights:
    hist=[sum(x) for x in zip(w['mask_hist_4095'],w['mask_hist_4096'])];masks={i//8 for i,v in enumerate(hist) if v};assert len(masks)==1;mask=next(iter(masks))
    if dev[w['context'],w['generation']]==0 and mask:keys.append(dict(ordinal=w['tensor_ordinal']-1,N=w['N'],K=w['K'],mask=mask,hits=w['W_bad_repeats_same_visible_generation']))
   keys.sort(key=lambda x:x['ordinal']);assert keys==entry['keys']==proof['keys'];assert len(keys)==11 and all(k['mask']==4 and k['hits']==2 for k in keys)
   term=json.loads(raw('private/'+tag+'.termination.json'));assert term['native_completed'] and term['native_exit']==0 and term['reader_finished'] and term['reader_error'] is None and term['native_cleanup_error'] is None
   summary=json.loads(raw('summary.json'));assert summary['completed'] and summary['cleanup']['passed'] and summary['cleanup']['api_left_stopped'];assert summary['raw_stderr_sha256']==sha(stderr)
   actual.append(dict(archive=archive.name,passed=True,primary_weights=108,peer_weights=84,output_ids=proof['output_ids'],routes=routes,negative_keys=keys))
 return actual
if __name__=='__main__':print(json.dumps(verify(),indent=2))
