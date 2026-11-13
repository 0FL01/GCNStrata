from pathlib import Path
import sys,json,hashlib
R=Path('/src');O=R/'logs/quant-comparison-64k-20261006';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'tools'));from strata_tokenizer import Tokenizer;import gguf_reader as G
coder=Path('/models/qwen38-flash-next-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf');legacy=Path('/models/qwen38-keep1-Q4_0.gguf')
a=G.GGUFFile(coder).metadata;b=G.GGUFFile(legacy).metadata
checks={key:a.get(key)==b.get(key) for key in ['tokenizer.ggml.tokens','tokenizer.ggml.merges','tokenizer.ggml.token_type','tokenizer.ggml.pre']}
assert all(checks.values()),checks
tk=Tokenizer.from_gguf(coder)
prefix='<|im_start|>user\nReview this synthetic Python module for correctness, edge cases and maintainability. After the source, give a detailed technical review with concrete examples.\n```python\n'
suffix='\n```\nProvide the detailed review now, including examples and suggested tests.\n<|im_end|>\n<|im_start|>assistant\n'
body=[]
for i in range(4500):
 body.append(f'def transform_{i:05d}(values, limit={i%97+3}):\n    """Normalize batch {i} while preserving order."""\n    result = []\n    for index, value in enumerate(values):\n        if value is None:\n            continue\n        adjusted = (value * {i%19+1} + index) % limit\n        if adjusted >= {i%7}:\n            result.append((index, adjusted))\n    return result\n\n')
p=tk.encode(prefix,parse_special=True);s=tk.encode(suffix,parse_special=True);tokens=tk.encode(''.join(body),parse_special=False)
N=65536;middle=tokens[:N-len(p)-len(s)];ids=p+middle+s
assert len(ids)==N and all(0<=x<len(tk.tokens) for x in ids)
text=tk.decode(ids);assert tk.encode(text,parse_special=True)==ids,'fixture decode/reencode differs'
(O/'prompt-65536.ids.json').write_text(json.dumps(ids)+'\n');(O/'prompt-65536.txt').write_text(text)
small=p+tokens[:4096-len(p)-len(s)]+s;assert len(small)==4096;(O/'prompt-4096.ids.json').write_text(json.dumps(small)+'\n')
m={'prompt_tokens':N,'generation_tokens':512,'context_capacity':131072,'vocabulary_size':len(tk.tokens),'tokenizer_equal_checks':checks,'prompt_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'source':'deterministic synthetic Python review fixture,4500 uniquely numbered functions then token-prefix cut with complete chat suffix','sampling':{'temperature':1,'top_p':.95,'top_k':20,'seed':12345},'eos_stop':'disabled with out-of-vocabulary sentinel2147483647','quality_claim':False}
(O/'fixture-manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
