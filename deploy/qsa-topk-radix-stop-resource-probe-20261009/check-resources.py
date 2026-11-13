#!/usr/bin/env python3
"""Pinned clang23 resource gate, adapted from the validated WIDE51 gate.

CPU-only. Uses the same readelf/remarks/ELF/objdump formats as the prior probe.
No device image is executed; passing still requires manual ISA review.
"""
import hashlib,json,pathlib,re,struct,sys,traceback

PREFIX='_ZN6strata7kernels12_GLOBAL__N_121block_topk_reg_kernel'
SUFFIX='EEEvPKfPKillPi'
OLD17=PREFIX+'ILi17ELb1ELi1'+SUFFIX
OLD66=PREFIX+'ILi66ELb1ELi2'+SUFFIX
NEW17=PREFIX+'ILi17ELb1ELi3'+SUFFIX
NEW66=PREFIX+'ILi66ELb1ELi4'+SUFFIX
ADDITIONS={NEW17,NEW66}
CROSS={'.private_segment_fixed_size':'ScratchSize [bytes/lane]', '.vgpr_count':'VGPRs',
       '.sgpr_count':'TotalSGPRs', '.vgpr_spill_count':'VGPRs Spill',
       '.sgpr_spill_count':'SGPRs Spill', '.group_segment_fixed_size':'LDS Size [bytes/block]'}

def metadata(path):
    text=path.read_text()
    assert re.search(r'^amdhsa.target:\s+amdgcn-amd-amdhsa--gfx906$',text,re.M),'Missing gfx906 metadata target'
    assert re.search(r'^\.\.\.$',text,re.M),'Truncated metadata document'
    records={}
    for match in re.finditer(r'^  - \.args:.*?(?=^  - \.args:|^amdhsa\.|^\.\.\.|\Z)',text,re.M|re.S):
        block=match.group(0).rstrip()
        pairs=re.findall(r'^    \.([a-z0-9_]+):\s+([^\n]+)$',block,re.M)
        fields=dict(pairs)
        assert len(fields)==len(pairs),'Duplicate metadata field'
        name=fields.pop('name',None)
        assert name and name not in records,'Missing or duplicate metadata kernel'
        fields={k:(int(v) if re.fullmatch(r'\d+',v) else v) for k,v in fields.items()}
        assert fields.get('symbol')==name+'.kd','Descriptor symbol mismatch'
        records[name]={'fields':fields,'complete_block':block}
    assert records,'No kernel metadata'
    return records

def remarks(path):
    result={};current=None
    for line in path.read_text().splitlines():
        if '[-Rpass-analysis=kernel-resource-usage]' not in line:continue
        m=re.search(r':\s+([^:\n]+): (.*?) \[-Rpass-analysis=kernel-resource-usage\]$',line)
        assert m,'Malformed resource remark'
        key,value=m[1].strip(),m[2]
        if key=='Function Name':
            assert value not in result,'Duplicate function resource remark'
            current=value;result[current]={};continue
        assert current and key not in result[current],'Missing function or duplicate resource field'
        result[current][key]=int(value) if value.isdigit() else value
    required=set(CROSS.values())|{'Dynamic Stack','Occupancy [waves/SIMD]'}
    assert result,'No resource remarks'
    for name,fields in result.items():
        assert required<=fields.keys(),'Missing required resource remark: '+name
        for key in required-{'Dynamic Stack'}:
            assert type(fields[key]) is int and fields[key]>=0,'Invalid resource number: '+name
    return result

def elf_functions(path):
    # The validated original probe's actual-function-byte parser, with bounded
    # structural checks so missing/truncated evidence fails rather than passing.
    b=path.read_bytes()
    assert len(b)>=64 and b[:6]==b'\x7fELF\x02\x01','ELF64 little-endian required'
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',b)
    assert h[2]==224,'AMDGPU ELF machine required'
    assert h[11]==64 and h[12]>0 and h[6]+h[11]*h[12]<=len(b),'Device ELF section table is truncated or invalid'
    sections=[struct.unpack_from('<IIQQQQIIQQ',b,h[6]+i*h[11]) for i in range(h[12])]
    funcs={}
    for sec in sections:
        if sec[1]!=2:continue
        assert sec[9]==24 and sec[5]%24==0 and sec[4]+sec[5]<=len(b),'Invalid ELF symbol table'
        assert 0<=sec[6]<len(sections),'Missing ELF symbol string section'
        strings=sections[sec[6]]
        assert strings[4]+strings[5]<=len(b),'Truncated ELF symbol strings'
        names=b[strings[4]:strings[4]+strings[5]]
        for off in range(sec[4],sec[4]+sec[5],sec[9]):
            n,info,other,idx,val,size=struct.unpack_from('<IBBHQQ',b,off)
            if info&15!=2:continue
            assert size>0 and 0<idx<len(sections),'Invalid or unsized ELF function'
            assert n<len(names) and b'\0' in names[n:],'Invalid ELF function name'
            name=names[n:names.index(b'\0',n)].decode()
            assert name not in funcs,'Duplicate ELF function'
            target=sections[idx]
            assert target[2]&4 and val>=target[3] and val-target[3]+size<=target[5],'Function outside executable section'
            begin=target[4]+val-target[3]
            assert begin>=0 and begin+size<=len(b),'Device ELF function is truncated'
            code=b[begin:begin+size]
            funcs[name]={'sha256':hashlib.sha256(code).hexdigest(),'bytes':size}
    assert funcs,'No sized STT_FUNC symbols in actual device ELF'
    return funcs

def disassembly(path):
    result={};name=None
    for line in path.read_text().splitlines():
        m=re.match(r'^[0-9a-fA-F]+ <([^>]+)>:$',line)
        if m:
            name=m[1];assert name not in result,'Duplicate disassembly function'
            result[name]=[];continue
        if name and re.search(r'// [0-9A-Fa-f]+:',line):
            result[name].append(line.split('//')[0].strip())
    return result

def main(root):
    failures=[]
    report={'passed':False,'gate_passed':False,'gpu_executed':False,'correctness_claim':False,
            'performance_claim':False,'manual_isa_review_required_before_any_gpu':True}
    def require(test,message):
        if not test:failures.append(message)
    try:
        a=metadata(root/'fit17.device.notes');b=metadata(root/'radix-stop.device.notes')
        ar=remarks(root/'fit17.resources.log');br=remarks(root/'radix-stop.resources.log')
        ae=elf_functions(root/'qsa_select.fit17-hip-amdgcn-amd-amdhsa-gfx906.out')
        be=elf_functions(root/'qsa_select.radix-stop-hip-amdgcn-amd-amdhsa-gfx906.out')
        ad=disassembly(root/'fit17.device.disasm');bd=disassembly(root/'radix-stop.device.disasm')
        require(len(a)==len(ae)==len(ar)==13,'Baseline must contain exactly 13 kernels/functions/remarks')
        require(len(b)==len(be)==len(br)==15,'Candidate must contain exactly 15 kernels/functions/remarks')
        require(set(a)==set(ae)==set(ar),'Baseline metadata/ELF/resource kernel sets differ')
        require(set(b)==set(be)==set(br),'Candidate metadata/ELF/resource kernel sets differ')
        require(set(a)<=set(b) and set(b)-set(a)==ADDITIONS,'Metadata additions are not exactly 17/3 and 66/4')
        require(set(ae)<=set(be) and set(be)-set(ae)==ADDITIONS,'Function additions are not exactly 17/3 and 66/4')
        require(set(ad)==set(a) and set(bd)==set(b),'Missing or extra disassembly kernel')
        common={}
        for name in sorted(a):
            same_meta=a[name]==b.get(name)
            same_code=ae.get(name)==be.get(name)
            same_remarks=ar.get(name)==br.get(name)
            require(same_meta,'Unchanged-kernel complete metadata changed: '+name)
            require(same_code,'Unchanged-kernel ELF machine code changed: '+name)
            require(same_remarks,'Unchanged-kernel compiler resource remarks changed: '+name)
            common[name]={'metadata_equal':same_meta,'machine_code_equal':same_code,'remarks_equal':same_remarks,'baseline_code':ae.get(name),'candidate_code':be.get(name)}
        report['unchanged_kernels']=common;report['comparisons']=[]
        for old_name,new_name,isa_file in [(OLD17,NEW17,'radix-stop.reg17.guard3.isa.txt'),(OLD66,NEW66,'radix-stop.reg66.guard4.isa.txt')]:
            assert old_name in a and new_name in b,'Missing comparison specialization'
            old=a[old_name]['fields'];new=b[new_name]['fields']
            for descriptor,remark in CROSS.items():
                key=descriptor[1:]
                assert type(old.get(key)) is int and type(new.get(key)) is int,'Missing required resource field: '+key
                require(new[key]==br[new_name][remark],'Candidate ELF metadata/remarks mismatch: '+key)
                require(old[key]==ar[old_name][remark],'Baseline ELF metadata/remarks mismatch: '+key)
                if key!='group_segment_fixed_size':require(new[key]<=old[key],'Resource regression '+key+': '+new_name)
            for key,value in [('wavefront_size',64),('max_flat_workgroup_size',1024),('group_segment_fixed_size',32908),('uses_dynamic_stack','false')]:
                require(new.get(key)==old.get(key)==value,'Required layout/resource changed '+key+': '+new_name)
            require(br[new_name]['Dynamic Stack']==ar[old_name]['Dynamic Stack']=='False','Dynamic stack remark changed')
            require(br[new_name]['Occupancy [waves/SIMD]']>=ar[old_name]['Occupancy [waves/SIMD]'],'Occupancy regression: '+new_name)
            require(bool(bd.get(new_name)),'No candidate disassembly: '+new_name)
            require(any(x.startswith('s_barrier') for x in bd.get(new_name,[])),'No candidate barrier: '+new_name)
            require(any(x.startswith('s_endpgm') for x in bd.get(new_name,[])),'No candidate endpgm: '+new_name)
            text=(root/'radix-stop.device.disasm').read_text()
            m=re.search(r'^[0-9a-fA-F]+ <'+re.escape(new_name)+r'>:\n',text,re.M);assert m,'No candidate ISA section'
            nxt=re.search(r'^[0-9a-fA-F]+ <[^>]+>:\n',text[m.end():],re.M)
            end=m.end()+nxt.start() if nxt else len(text)
            (root/isa_file).write_text(text[m.start():end])
            report['comparisons'].append({'baseline':old_name,'candidate':new_name,'baseline_resources':old,'candidate_resources':new,'baseline_remarks':ar[old_name],'candidate_remarks':br[new_name],'isa_file':isa_file,'manual_isa_review_passed':None})
        report['automatic_gate_scope']='Pinned clang23 formats; exactly 13 unchanged complete metadata/remarks and actual ELF function bytes; exactly two added specializations; no scratch/spill/SGPR/VGPR/occupancy regression; fixed LDS/wave/thread layout.'
        report['manual_review_scope']='Inspect both new ISA bodies for guards, uniform stop, single writer, barrier ordering and exact key recovery. Barrier presence is not a correctness proof.'
    except Exception as exc:
        failures.append(type(exc).__name__+': '+str(exc));report['exception_traceback']=traceback.format_exc()
    report['failures']=failures;report['passed']=report['gate_passed']=not failures
    (root/'resource-gate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['passed'] else 14

if __name__=='__main__':sys.exit(main(pathlib.Path(sys.argv[1])))
