#!/usr/bin/env python3
"""Conservative CPU resource / unchanged-kernel ISA gate; never runs a device image."""
import hashlib,json,pathlib,re,struct,sys,traceback
root=pathlib.Path(sys.argv[1])
failures=[]
report={"passed":False,"gpu_executed":False,"performance_claim":False,"manual_isa_review_required_before_any_gpu":True}
def require(test,message):
    if not test:failures.append(message)
def metadata(path):
    records={}
    for block in re.split(r"\n  - \.args:",path.read_text())[1:]:
        fields=dict(re.findall(r"^    \.([a-z0-9_]+):\s+(\S+)",block,re.M))
        name=fields.get("name")
        if name:
            records[name]={k:(int(v) if re.fullmatch(r"\d+",v) else v) for k,v in fields.items() if k!="name"}
    return records
def remarks(path):
    result={};current=None
    for line in path.read_text().splitlines():
        m=re.search(r"Function Name: (\S+)",line)
        if m:current=m[1];result[current]={};continue
        m=re.search(r":\s+([^:\n]+): (.*?) \[-Rpass-analysis=kernel-resource-usage\]$",line)
        if m and current:
            value=m[2];result[current][m[1].strip()]=int(value) if value.isdigit() else value
    return result
def elf_functions(path):
    b=path.read_bytes()
    assert b[:6]==b"\x7fELF\x02\x01","ELF64 little-endian required"
    h=struct.unpack_from("<16sHHIQQQIHHHHHH",b)
    assert h[2]==224,"AMDGPU ELF machine required"
    assert h[6]+h[11]*h[12]<=len(b),"Device ELF is truncated: section table extends beyond file"
    sections=[struct.unpack_from("<IIQQQQIIQQ",b,h[6]+i*h[11]) for i in range(h[12])]
    funcs={}
    for sec in sections:
        if sec[1]!=2:continue
        strings=sections[sec[6]]
        names=b[strings[4]:strings[4]+strings[5]]
        for off in range(sec[4],sec[4]+sec[5],sec[9]):
            n,info,other,idx,val,size=struct.unpack_from("<IBBHQQ",b,off)
            if info&15!=2 or size==0 or idx>=len(sections):continue
            name=names[n:names.index(b"\0",n)].decode()
            target=sections[idx]
            begin=target[4]+val-target[3]
            assert begin>=0 and begin+size<=len(b),"Device ELF function extends beyond file"
            code=b[begin:begin+size]
            assert len(code)==size
            funcs[name]={"sha256":hashlib.sha256(code).hexdigest(),"bytes":size}
    assert funcs,"No sized STT_FUNC symbols in actual device ELF"
    return funcs
def disassembly(path):
    result={};name=None
    for line in path.read_text().splitlines():
        m=re.match(r"^[0-9a-fA-F]+ <([^>]+)>:$",line)
        if m:name=m[1];result[name]=[];continue
        if name and re.search(r"// [0-9A-Fa-f]+:",line):
            result[name].append(line.split("//")[0].strip())
    return result
try:
    a=metadata(root/"fit17.device.notes")
    b=metadata(root/"wide51.device.notes")
    ar=remarks(root/"fit17.resources.log")
    br=remarks(root/"wide51.resources.log")
    ae=elf_functions(root/"qsa_select.fit17-hip-amdgcn-amd-amdhsa-gfx906.out")
    be=elf_functions(root/"qsa_select.wide51-hip-amdgcn-amd-amdhsa-gfx906.out")
    ad=disassembly(root/"fit17.device.disasm")
    bd=disassembly(root/"wide51.device.disasm")
    s51="_ZN6strata7kernels12_GLOBAL__N_121block_topk_reg_kernelILi51ELb1ELi2EEEvPKfPKillPi"
    s66=s51.replace("ILi51","ILi66")
    require(set(a)<=set(b),"Baseline kernel missing from candidate metadata")
    require(set(b)-set(a)=={s51},"Candidate kernel additions are not exactly guarded PER51")
    require(set(a)<=set(ae) and set(b)<=set(be),"Actual device ELF is missing metadata-listed functions")
    common={}
    fields=["group_segment_fixed_size","private_segment_fixed_size","sgpr_count","sgpr_spill_count","vgpr_count","vgpr_spill_count","wavefront_size","max_flat_workgroup_size","uses_dynamic_stack"]
    for name in sorted(a):
        same_meta={k:a[name].get(k) for k in fields}=={k:b.get(name,{}).get(k) for k in fields}
        same_code=ae.get(name)==be.get(name)
        same_remarks=ar.get(name)==br.get(name)
        require(same_meta,"Unchanged-kernel resource descriptor changed: "+name)
        require(same_code,"Unchanged-kernel ELF machine code changed: "+name)
        require(same_remarks,"Unchanged-kernel compiler resource remarks changed: "+name)
        common[name]={"metadata_equal":same_meta,"machine_code_equal":same_code,"remarks_equal":same_remarks,"baseline_code":ae.get(name),"candidate_code":be.get(name)}
    require(s51 in b and s66 in a,"Missing wide51 or baseline guarded wide66")
    old=a[s66];new=b[s51]
    require(new.get("wavefront_size")==64,"PER51 is not wave64")
    require(new.get("max_flat_workgroup_size")==1024,"PER51 launch bound changed")
    require(new.get("group_segment_fixed_size")==old.get("group_segment_fixed_size")==32908,"PER51 LDS layout changed")
    require(new.get("private_segment_fixed_size",10**9)<old["private_segment_fixed_size"],"PER51 has no scratch reduction")
    require(new.get("vgpr_spill_count",10**9)<old["vgpr_spill_count"],"PER51 has no VGPR spill reduction")
    require(new.get("sgpr_spill_count",10**9)<=old["sgpr_spill_count"],"PER51 increases SGPR spills")
    require(new.get("vgpr_count",10**9)<=old["vgpr_count"],"PER51 increases VGPR allocation")
    require(new.get("sgpr_count",10**9)<=old["sgpr_count"],"PER51 increases SGPR allocation")
    require(new.get("uses_dynamic_stack")=="false","PER51 unexpectedly uses a dynamic stack")
    require(br[s51]["Occupancy [waves/SIMD]"]>=ar[s66]["Occupancy [waves/SIMD]"],"PER51 compiler occupancy regresses")
    cross={"private_segment_fixed_size":"ScratchSize [bytes/lane]","vgpr_count":"VGPRs","sgpr_count":"TotalSGPRs","vgpr_spill_count":"VGPRs Spill","sgpr_spill_count":"SGPRs Spill","group_segment_fixed_size":"LDS Size [bytes/block]"}
    for key,remark in cross.items():
        require(new.get(key)==br[s51].get(remark),"PER51 ELF/remarks mismatch: "+key)
        require(old.get(key)==ar[s66].get(remark),"PER66 ELF/remarks mismatch: "+key)
    def isa_summary(instructions):
        return {"instruction_lines":len(instructions),"barriers":sum(x.startswith("s_barrier") for x in instructions),"conditional_branches":sum(x.startswith("s_cbranch") for x in instructions),"buffer_or_scratch_load_stores":sum(bool(re.match(r"(buffer|scratch)_(load|store)",x)) for x in instructions)}
    require(s51 in bd and len(bd[s51])>0,"No PER51 disassembly")
    require(any(x.startswith("s_barrier") for x in bd[s51]),"PER51 has no expected barriers")
    require(any(x.startswith("s_endpgm") for x in bd[s51]),"PER51 has no terminal instruction")
    # Save the complete new function separately so root can inspect its guard/control flow.
    text=(root/"wide51.device.disasm").read_text()
    m=re.search(r"^[0-9a-fA-F]+ <"+re.escape(s51)+r">:\n",text,re.M)
    assert m
    nxt=re.search(r"^[0-9a-fA-F]+ <[^>]+>:\n",text[m.end():],re.M)
    end=m.end()+nxt.start() if nxt else len(text)
    (root/"wide51.guard2.isa.txt").write_text(text[m.start():end])
    report.update({"unchanged_kernels":common,"baseline_wide66":old,"candidate_wide51":new,"baseline_wide66_remarks":ar[s66],"candidate_wide51_remarks":br[s51],"baseline_wide66_code":ae[s66],"candidate_wide51_code":be[s51],"baseline_wide66_isa_summary":isa_summary(ad[s66]),"candidate_wide51_isa_summary":isa_summary(bd[s51]),"automatic_gate_scope":"Actual device ELF symbols, all existing kernels' exact machine bytes and resources, one new guarded PER51, wave64/1024/LDS layout and lower spill/scratch without register or occupancy regression.","manual_review_scope":"Inspect wide51.guard2.isa.txt and source proof for preserved pre-barrier short/window guard exits and body control flow. No GPU work follows this script.","qualification_limits":"No real 200K score/steps capture exists yet. The existing 64K capture exercises reg17 and is not wide51 performance proof."})
except Exception as exc:
    failures.append(type(exc).__name__+": "+str(exc))
    report["exception_traceback"]=traceback.format_exc()
report["failures"]=failures
report["passed"]=not failures
(root/"resource-gate.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"failures":failures,"baseline_wide66":report.get("baseline_wide66"),"candidate_wide51":report.get("candidate_wide51"),"manual_isa_review_required_before_any_gpu":True,"performance_claim":False},indent=2))
sys.exit(0 if report["passed"] else 14)

