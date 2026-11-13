# Isolated copy: optional FIT17 trace requirement removed; private streaming stdout and failure receipts added.
from native_support import *
import sys

def run_arm(tag,r,s,ids,ds,out):
    n=len(ids);new=s['requested_outputs']
    need(not call(['docker','ps','-q']).strip(),'unexpected active container')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    need(available>=64*1024**3,'insufficient RAM')
    need(sha(r['binary'])==r['binary_sha256'],'binary changed')
    initial=cooldown(ds,out)
    name='strata-primary-cache-abba-'+tag.lower();binary='/src/'+str(Path(r['binary']).relative_to(R))
    cmd=['docker','run','--rm','-i','--name',name,'--network','none','--device','/dev/kfd','--device','/dev/dri','--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint',binary]
    for k,v in sorted(r['flags'].items()):cmd+=['-e',k+'='+v]
    cmd+=['-v',str(R)+':/src:ro','-v','/home/radneon/models-nvme:/models:ro','-v','/home/radneon/models:/sata:ro',IMAGE,'--serve']+r['native_args']
    m={'tag':tag,'release_binary_sha256':r['binary_sha256'],'release_artifacts':r['release_artifacts'],'flags':r['flags'],'native_args':r['native_args'],'fixture_sha256':s['fixture_sha256'],'prompt_tokens':n,'requested_outputs':new,'sampling':{'temperature':0,'top_p':1,'top_k':1,'seed':12345},'started_utc':stamp(),'gpu_before':initial,'fresh_process':True}
    write(out/(tag+'.manifest.json'),m);print(json.dumps({'event':'start','tag':tag}),flush=True)
    private=out/'private';private.mkdir(mode=0o700,exist_ok=True);need(private.stat().st_mode & 0o777==0o700,'private log directory mode')
    stderr_path=private/(tag+'.stderr');m['private_stderr_file']='private/'+tag+'.stderr'
    stdout_path=private/(tag+'.stdout');m['private_stdout_file']='private/'+tag+'.stdout'
    p=None;tokens=[];info={};samples=[];started=time.monotonic();ready=None;request_start=None;done=None;q=queue.Queue()
    t=None;raw=None;reader_error=None;stdout_lines=0;parsed_lines=0;completed=False
    try:
        raw=os.fdopen(os.open(stdout_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'w',buffering=1)
        # Full stderr is private diagnostic evidence, never a public report/log upload.
        fd=os.open(stderr_path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        with os.fdopen(fd,mode='w+t') as err:
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
            def reader():
                nonlocal reader_error,stdout_lines
                try:
                    for line in p.stdout:
                        raw.write(line);raw.flush()  # Retain the full line before it can be parsed or rejected.
                        stdout_lines+=1;q.put(line)
                except BaseException as e:
                    reader_error={'error_type':type(e).__name__,'error':str(e)}
                    q.put(('reader_error',reader_error))
                finally:
                    try:raw.flush();os.fsync(raw.fileno());raw.close()
                    except BaseException as e:
                        reader_error={'error_type':type(e).__name__,'error':str(e)}
                    q.put(None)
            t=threading.Thread(target=reader,daemon=True);t.start();last=0
            while done is None:
                now=time.monotonic();need(now-started<600,'native arm timeout');need(not (out/'cancel').exists(),'cancellation marker')
                if now-last>=5:samples.append({'elapsed_s':now-started,'gpus':gpu(ds)});last=now
                try:line=q.get(timeout=5)
                except queue.Empty:continue
                need(line is not None,'engine ended before DONE')
                need(not isinstance(line,tuple),'stdout reader failed: '+str(reader_error))
                parsed_lines+=1;need(not line.startswith('ERR '),'native error')
                if line.startswith('INFO '):info=dict(x.split('=',1) for x in line.split()[1:] if '=' in x)
                elif line.startswith('READY '):
                    need(ready is None and line.split()[1]=='204800','READY protocol')
                    for k,v in {'expert_slots':'19078','expert_slots_primary':'9900','kv':'int8','context':'204800','lookup':'3','mtp_max':'4','spec':'6','kv_resident':'32768','pcie_frac':'0.00'}.items():need(info.get(k)==v,'engine configuration '+k)
                    err.flush();need('locked in RAM' in os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace'),'PLE RAM lock missing')
                    ready=time.monotonic();m['load_wall_s']=ready-started;m['engine_info']=info;request_start=time.monotonic()
                    p.stdin.write('GEN '+str(new)+' temperature=0 top_p=1 top_k=1 seed=12345 '+','.join(map(str,ids))+'\n');p.stdin.flush()
                elif line.startswith('T '):tokens.append(int(line.split()[1]));need(len(tokens)<=new,'too many outputs')
                elif line.startswith('DONE '):
                    # Persist raw protocol and complete received IDs BEFORE parsing or validation.
                    m['private_native_forensics']=capture_native_done(private,tag,line,tokens)
                    write(out/(tag+'.manifest.json'),m)
                    need(request_start is not None,'DONE before READY');done=line.split();m['request_wall_s']=time.monotonic()-request_start
            p.stdin.write('QUIT\n');p.stdin.flush();p.stdin.close()
            shutdown_started=time.monotonic()
            while p.poll() is None:
                need(time.monotonic()-shutdown_started<60,'native shutdown timeout')
                need(not (out/'cancel').exists(),'cancellation marker')
                samples.append({'elapsed_s':time.monotonic()-started,'phase':'shutdown','gpus':gpu(ds)})
                try:p.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
            code=p.returncode;t.join(timeout=5);need(code==0,'native exit')
            need(not t.is_alive() and reader_error is None,'stdout capture incomplete before admission')
            err.flush();stderr=os.pread(err.fileno(),os.fstat(err.fileno()).st_size,0).decode(errors='replace');topology=fit17_topology(stderr);m['fit17_host_topology']=topology;need('strata attn-reduce12: selected=1 ' not in stderr,'unexpected performance trace')
            suffix=suffix_totals(stderr);m['suffix_drafts']=suffix
            if r['flags'].get('STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE')=='1':need('attn-query-swizzle: active=1 ' in stderr,'candidate QSA route absent')
        need(len(done)>=15 and int(done[1])==len(tokens)==new and int(done[2])==n and done[5]=='length','DONE output coverage')
        need(int(done[8])==0 and int(done[14])==n,'prompt reuse/read')
        need(all(0<=x<248320 for x in tokens),'output ID domain')
        pm,dm=map(float,done[3:5]);need(math.isfinite(pm) and math.isfinite(dm) and pm>0 and dm>0,'durations')
        extended=native_counter_fields(done)
        result={'tag':tag,'passed':True,'prompt_tokens':n,'actual_outputs':new,'prompt_ms':pm,'decode_ms':dm,'accepted':int(done[6]),'offered':int(done[7]),'cache_hits':int(done[9]),'lookups':int(done[10]),'prompt_reused':int(done[8]),'prompt_read':int(done[14]),'output_ids':tokens,**extended,'engine_info':info,'suffix_drafts':suffix,'fit17_host_topology':topology,'timing_scope':'fixed ABBA native phase timing; adaptive counters reported separately','request_wall_s':m['request_wall_s'],'load_wall_s':m['load_wall_s']}
        write(out/(tag+'.result.json'),result);completed=True;return result
    finally:
        failure=sys.exc_info()[1]
        cleanup_error=None
        try:
            if p is not None and p.poll() is None:
                try:call(['docker','stop','--time','30',name],60)
                except (subprocess.SubprocessError,OSError):
                    try:call(['docker','kill',name],30)
                    except (subprocess.SubprocessError,OSError):pass
                try:p.wait(timeout=60)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        except BaseException as e:cleanup_error={'error_type':type(e).__name__,'error':str(e)}
        if t is not None:t.join(timeout=5)
        reader_finished=t is None or not t.is_alive()
        if t is None and raw is not None:raw.flush();os.fsync(raw.fileno());raw.close()
        status={'native_completed':completed,'done_seen_by_parser':done is not None,
                'error_type':type(failure).__name__ if failure is not None else None,
                'error':str(failure) if failure is not None else None,
                'native_exit':p.returncode if p else None,'native_cleanup_error':cleanup_error,'reader_finished':reader_finished,
                'reader_error':reader_error,'received_stdout_lines':stdout_lines,'parsed_stdout_lines':parsed_lines,
                'parsed_token_count':len(tokens),'finished_utc':stamp()}
        forensic={}
        for label,suffix,value in [('partial_output_ids','partial-output-ids.json',tokens),('termination','termination.json',status)]:
            path=private/(tag+'.'+suffix)
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'w') as stream:
                json.dump(value,stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            forensic[label]={'relative_path':'private/'+path.name,'sha256':sha(path)}
        if stdout_path.exists():
            forensic['raw_stdout']={'relative_path':'private/'+stdout_path.name,'sha256':sha(stdout_path),'complete':reader_finished and reader_error is None}
        m['private_termination_forensics']=forensic
        if stderr_path.exists():m['private_stderr_sha256']=sha(stderr_path)
        m.update(finished_utc=stamp(),elapsed_s=time.monotonic()-started,native_exit=p.returncode if p else None)
        write(out/(tag+'.manifest.json'),m);write(out/(tag+'.gpu-samples.json'),samples)
        need(reader_finished and reader_error is None,'stdout evidence capture incomplete')
        need(cleanup_error is None,'native process cleanup failed: '+str(cleanup_error))
