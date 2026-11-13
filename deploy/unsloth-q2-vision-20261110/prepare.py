from pathlib import Path
import hashlib, json, struct, subprocess, sys, zlib
import numpy as np
ROOT = Path('/home/radneon/strata')
OUT = Path(__file__).resolve().parent
MODEL = '/home/radneon/models/qwen38-flash-next-ud-q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf'
CPATH = '/src/deploy/unsloth-q2-vision-20261110'
def chunk(tag, data):
    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
pixels = b''.join(b'\0' + b''.join(bytes((220,20,20) if 56 <= x < 168 and 56 <= y < 168 else (20,40,220)) for x in range(224)) for y in range(224))
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB',224,224,8,2,0,0,0)) + chunk(b'IDAT',zlib.compress(pixels)) + chunk(b'IEND',b'')
(OUT / 'image.png').write_bytes(png)
cmd = ['docker','run','--rm','-i','--name','unsloth-q2-cpu-vision','--network','none',
       '-v',str(ROOT)+':/src','-v','/home/radneon/models:/sata:ro',
       '--entrypoint','/src/build-vision-cpu/bin/strata-vision',
       'sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646',
       '--mmproj','/sata/shefowl-qwen38-hybrid/mmproj-Qwen3.8-Flash-Next-f16.gguf',
       '--model','/sata/qwen38-flash-next-ud-q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf',
       '--threads','16','--max-tokens','1024']
try:
    p = subprocess.run(cmd,input=f'ENC {CPATH}/image.png {CPATH}/image.embd\nQUIT\n',text=True,capture_output=True,timeout=300)
finally:
    subprocess.run(['docker','rm','-f','unsloth-q2-cpu-vision'],capture_output=True)
(OUT/'encoder.stdout').write_text(p.stdout)
(OUT/'encoder.stderr').write_text(p.stderr)
assert p.returncode == 0 and p.stdout.startswith('READY 2560') and 'OK ' in p.stdout and 'ERR ' not in p.stdout, p.stdout+p.stderr[-2000:]
blob=(OUT/'image.embd').read_bytes()
magic,n,nx,ny,width=struct.unpack_from('<5i',blob)
assert magic==0x31455653 and width==2560 and n==nx*ny and len(blob)==20+n*width*4
assert np.isfinite(np.frombuffer(blob,dtype='<f4',offset=20)).all()
text='<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the colors and shape in this image briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
code="import sys,json; from pathlib import Path; sys.path.insert(0,'/src/tools'); from strata_tokenizer import Tokenizer; tk=Tokenizer.from_gguf(Path('/sata/qwen38-flash-next-ud-q2_0/Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf')); text=json.load(sys.stdin); print(json.dumps({'ids':tk.encode(text,parse_special=True),'pad':tk.encode('<|image_pad|>',parse_special=True)[0]}))"
tok=subprocess.run(['docker','run','--rm','-i','--network','none','-v',str(ROOT)+':/src:ro','-v','/home/radneon/models:/sata:ro','--entrypoint','python3','sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646','-c',code],input=json.dumps(text),text=True,capture_output=True,check=True,timeout=60)
encoded=json.loads(tok.stdout);ids=encoded['ids'];pad=encoded['pad']
assert ids.count(pad)==1
i=ids.index(pad);ids[i:i+1]=[pad]*n
(OUT/'smoke.json').write_text(json.dumps({'ids':ids,'embeddings':CPATH+'/image.embd','max_new':64})+'\n')
result={'passed':True,'cpu_only':True,'devices_exposed':False,'image_tokens':n,'width':width,
        'image_sha256':hashlib.sha256(png).hexdigest(),'command':cmd,'model':MODEL,
        'encoder_stdout':p.stdout,'engine_smoke':'pending'}
(OUT/'encoder-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
