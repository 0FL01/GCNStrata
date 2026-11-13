from pathlib import Path
import subprocess as s, json, time, hashlib, re, urllib.request
R=Path('/home/radneon/strata')
O=R/'logs/perf-campaign-20261008/iq4xs-flat-dequant/pr1525'
D=O/'micro'
D.mkdir(exist_ok=False)
assert json.loads((O/'build.json').read_text())['exit_code']==0
image='sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646'
summary={'runs':[]}
def run(tag,mode,gpu,case):
    target=D/(tag+'.bin')
    cmd=['docker','run','--rm','--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--name','strata-iq4xs-flat-micro','--entrypoint','/src/logs/perf-campaign-20261008/iq4xs-flat-dequant/pr1525/gfx906_iq4xs_dequant_probe']
    for k,v in {'HIP_VISIBLE_DEVICES':gpu,'STRATA_GFX906_IQ4XS_FLAT':mode,'STRATA_DQ_IQ4XS_TRACE':1,'STRATA_IQ4XS_PROBE_CASE':case,'STRATA_DEQUANT_DUMP':'/src/'+str(target.relative_to(R))}.items():cmd+=['-e',str(k)+'='+str(v)]
    cmd+=['-v',str(R)+':/src',image,'--device','0','--shape','qkv','--warmups','5','--reps','20']
    p=s.run(cmd,stdout=s.PIPE,stderr=s.STDOUT,text=True,timeout=180)
    (D/(tag+'.log')).write_text(p.stdout)
    assert p.returncode==0,p.stdout[-2000:]
    assert 'route='+('flat' if mode else 'h16x8-eligible') in p.stdout,p.stdout[:2000]
    if mode==0:assert 'actual=h16x8 type=23' in p.stdout,p.stdout[:2000]
    b=target.read_bytes()
    ref=R/('logs/perf-campaign-20261008/iq4xs-flat-dequant/micro/'+('A1.bin' if case==0 else 'gpu0_case1_A.bin'))
    assert b==ref.read_bytes(),'Mismatch against original group32 dump'
    m=re.search(r'median_ms=([0-9.]+)',p.stdout)
    assert m,p.stdout[-3000:]
    row={'tag':tag,'gpu':gpu,'case':case,'mode':mode,'median_ms':float(m.group(1)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    summary['runs'].append(row)
    print(json.dumps(row),flush=True)
    return b,row['median_ms']
try:
    s.run(['docker','stop','strata-api'],check=True,timeout=120)
    assert not s.check_output(['docker','ps','-q'],text=True).strip()
    power=s.check_output(['sh','-c','cat /sys/class/drm/card*/device/power_dpm_force_performance_level'],text=True)
    summary['power']=power
    assert all(x=='high' for x in power.split()),power
    out=[run(tag,mode,0,0) for tag,mode in [('A1',0),('B1',1),('B2',1),('A2',0)]]
    assert all(x[0]==out[0][0] for x in out),'ABBA byte mismatch'
    ratio=(out[0][1]+out[3][1])/(out[1][1]+out[2][1])
    summary['speedup']=ratio
    summary['component_comparison']='PR1525 h16x8 versus existing flat; original group32 fullbyte reference'
    if True:
        for gpu,case in [(0,1),(1,0),(1,1)]:
            a=run('gpu%d_case%d_A'%(gpu,case),0,gpu,case)
            b=run('gpu%d_case%d_B'%(gpu,case),1,gpu,case)
            assert a[0]==b[0],'GPU/case byte mismatch'
    summary['passed']=True
except Exception as e:
    summary['error']=repr(e)
    raise
finally:
    s.run(['docker','start','strata-api'],timeout=120,check=True)
    summary['api_restored']=False
    for i in range(120):
        try:
            h=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5).read())
            if h.get('loaded') and h.get('images') and h.get('api_key'):
                summary['api_restored']=True
                summary['health']=h
                break
        except Exception:pass
        time.sleep(5)
    (D/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    assert summary['api_restored'],'API did not restore'
