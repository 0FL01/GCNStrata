#!/usr/bin/env python3
"""Root-run permission-only container check; never loads HIP or native model code."""
import json, os, pathlib, subprocess, tempfile

# Run inside the exact pinned native image, with the same identity, devices and binds.
# No model contents are read or emitted; files are opened and immediately closed.
PROBE = r'''
import glob, grp, json, os, pathlib, stat, sys
spec=json.loads(sys.argv[1])
def need(ok,why):
 if not ok: raise RuntimeError(why)
need(os.geteuid()==spec['uid'] and os.getegid()==spec['gid'],'container UID/GID differ')
expected=set(spec['groups'])|{grp.getgrnam('video').gr_gid}
actual=set(os.getgroups())
need(expected <= actual and actual <= expected|{spec['gid']},'container supplementary groups differ')
for p in spec['binaries']:
 need(os.path.isfile(p) and os.access(p,os.R_OK|os.X_OK),'binary read/execute access: '+p)
count=0;seen=set()
def check_file(p):
 global count
 count+=1;need(count<=100000,'bounded permission walk exceeded')
 fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC)
 try:need(stat.S_ISREG(os.fstat(fd).st_mode),'nonregular model input: '+p)
 finally:os.close(fd)
for p in spec['files']:check_file(p)
for root in spec['directories']:
 need(os.path.isdir(root),'model directory missing: '+root)
 for folder,dirs,files in os.walk(root,followlinks=True,onerror=lambda e: (_ for _ in ()).throw(e)):
  st=os.stat(folder);key=(st.st_dev,st.st_ino)
  if key in seen:dirs[:]=[];continue
  seen.add(key);need(len(seen)<=100000,'bounded directory walk exceeded')
  need(os.access(folder,os.R_OK|os.X_OK),'model directory access: '+folder)
  for name in files:check_file(os.path.join(folder,name))
devices=['/dev/kfd']+sorted(glob.glob('/dev/dri/renderD*'))
need(len(devices)>=3,'expected kfd and two render nodes')
for p in devices:
 need(stat.S_ISCHR(os.stat(p).st_mode) and os.access(p,os.R_OK|os.W_OK),'GPU device permissions: '+p)
parent='/captures';st=os.stat(parent)
need(st.st_uid==spec['uid'] and stat.S_IMODE(st.st_mode)==0o700,'private capture directory owner/mode')
p=parent+'/permission-probe';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
try:need(stat.S_IMODE(os.fstat(fd).st_mode)==0o600,'private output file mode')
finally:os.close(fd);os.unlink(p)
print(json.dumps({'passed':True,'uid':os.geteuid(),'gid':os.getegid(),'groups':sorted(actual),'files_opened':count,'directories_checked':len(seen),'gpu_nodes_permissions_checked':devices,'gpu_nodes_opened':False,'hip_initialized':False,'model_contents_read':False,'capture_create_unlink':True}))
'''

def permission_spec(native_args,binaries):
    args=list(native_args)
    def value(flag):return args[args.index(flag)+1]
    return {'uid':os.geteuid(),'gid':os.getegid(),'groups':sorted(set(os.getgroups())),
            'binaries':binaries,
            'files':[value(f) for f in ['--native','--ple-gguf','--expert-profile']],
            'directories':[value(f) for f in ['--pack','--mtp']]}

def command(image,R,private_dir,flags,spec):
    cmd=['docker','run','--rm','--name','strata-qsa-200k-access-preflight','--network','none',
         '--user',str(spec['uid'])+':'+str(spec['gid']), '--device','/dev/kfd','--device','/dev/dri',
         '--group-add','video','--ipc','host','--ulimit','memlock=-1:-1','--entrypoint','python3']
    for gid in spec['groups']:cmd+=['--group-add',str(gid)]
    for k,v in sorted(flags.items()):cmd+=['-e',k+'='+v]
    cmd+=['-v',str(R)+':/src:ro','-v','/home/radneon/models-nvme:/models:ro',
          '-v','/home/radneon/models:/sata:ro','-v',str(private_dir)+':/captures:rw',
          image,'-c',PROBE,json.dumps(spec,separators=(',',':'))]
    return cmd

def run(image,R,campaign,native_args,binaries,flags):
    spec=permission_spec(native_args,binaries)
    # Separate owned temporary directory; failed preflight does not consume the one native pair.
    with tempfile.TemporaryDirectory(prefix='qsa-200k-access-',dir=campaign) as private:
        cmd=command(image,R,private,flags,spec)
        try:
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
            if result.returncode!=0:
                raise RuntimeError('permission-only container preflight failed: '+result.stderr[-3000:])
            data=json.loads(result.stdout)
            if data.get('passed') is not True:raise RuntimeError('permission preflight did not pass')
            return data
        finally:
            # A timed-out Docker CLI can leave its container alive. Stop only this owned name.
            name='strata-qsa-200k-access-preflight'
            active=subprocess.check_output(['docker','ps','-q','--filter','name=^/'+name+'$'],timeout=30).strip()
            if active:
                try:subprocess.run(['docker','stop','--time','10',name],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
                except (OSError,subprocess.SubprocessError):subprocess.run(['docker','kill',name],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
            if subprocess.check_output(['docker','ps','-q','--filter','name=^/'+name+'$'],timeout=30).strip():
                raise RuntimeError('permission preflight container remains; stop before native work')
