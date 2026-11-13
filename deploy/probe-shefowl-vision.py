from pathlib import Path
import json,subprocess,struct,zlib,binascii,time,array,math,hashlib
R=Path('/home/radneon/strata');O=R/'logs/shefowl-production-20261006'
W=H=1024;white=b'\xff\xff\xff';red=b'\xff\x00\x00';blue=b'\x00\x00\xff'
rows=[]
for y in range(H):
 row=white*W if y<128 or y>=896 else white*64+red*416+white*64+blue*416+white*64
 rows.append(b'\0'+row)
def chunk(tag,data):return struct.pack('>I',len(data))+tag+data+struct.pack('>I',binascii.crc32(tag+data)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b'')
(O/'red-blue.png').write_bytes(png)
def vram():
 return {str(d.resolve()):int((d/'mem_info_vram_used').read_text()) for d in Path('/sys/class/drm').glob('card[0-9]*/device') if (d/'mem_info_vram_used').exists()}
before=vram();start=time.monotonic()
cmd=['docker','run','--rm','-i','--name','strata-vision-cpu-probe','--entrypoint','/src/build-vision-cpu/bin/strata-vision','-v',str(R)+':/src','-v','/home/radneon/models:/models:ro','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','--mmproj','/models/shefowl-qwen38-hybrid/mmproj-Qwen3.8-Flash-Next-f16.gguf','--model','/models/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf','--threads','16','--max-tokens','1024']
with (O/'probe.stdout').open('x') as out,(O/'probe.stderr').open('x') as err:
 p=subprocess.run(cmd,input='ENC /src/logs/shefowl-production-20261006/red-blue.png /src/logs/shefowl-production-20261006/probe.sve\nQUIT\n',text=True,stdout=out,stderr=err,timeout=1200)
after=vram();stdout=(O/'probe.stdout').read_text();assert p.returncode==0 and 'READY 2560' in stdout and '\nOK ' in stdout,stdout[-1000:]
raw=(O/'probe.sve').read_bytes();magic,n,nx,ny,dim=struct.unpack('<5i',raw[:20]);assert magic==0x31455653 and nx*ny==n and dim==2560 and n<=1024 and len(raw)==20+n*dim*4
values=array.array('f');values.frombytes(raw[20:]);assert all(math.isfinite(x) for x in values)
e={'native_exit':0,'passed':True,'elapsed_s':time.monotonic()-start,'stdout':stdout,'image_sha256':hashlib.sha256(png).hexdigest(),'tokens':n,'grid':[nx,ny],'embedding_dim':dim,'all_finite':True,'vram_before':before,'vram_after':after,'vram_unchanged':before==after,'gpu_devices_exposed_to_probe':False,'argv':cmd}
(O/'probe-result.json').write_text(json.dumps(e,indent=2));print(json.dumps(e),flush=True)
