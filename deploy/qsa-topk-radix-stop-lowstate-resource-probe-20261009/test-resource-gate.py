#!/usr/bin/env python3
"""CPU-only synthetic ELF64 AMDGPU fixtures; no compiler, YAML package, or GPU.

Run: python3 test-resource-gate.py
Each negative case executes the real CLI and requires exit14 and a false JSON
report. The positive fixture contains real ELF sections, STT_FUNC symbols,
64-byte kernel descriptors, full metadata/remarks, and byte-matched disassembly.
"""
import copy
import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("resource_gate", HERE / "check-resources.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)
VARIANTS = ("fit17", "radix-stop")
ELFNAME = "qsa_select.{}-hip-amdgcn-amd-amdhsa-gfx906.out"


def metadata(name):
    return {".args": [
        {".actual_access": "read_only", ".address_space": "global", ".offset": 0,
         ".size": 8, ".value_kind": "global_buffer"},
        {".offset": 8, ".size": 8, ".value_kind": "by_value"},
        {".actual_access": "write_only", ".address_space": "global", ".offset": 16,
         ".size": 8, ".value_kind": "global_buffer"}],
        ".gfx1250_revision": "B0", ".group_segment_fixed_size": 32908,
        ".kernarg_segment_align": 8, ".kernarg_segment_size": 24,
        ".language": "OpenCL C", ".language_version": [2, 0],
        ".max_flat_workgroup_size": 1024, ".name": name,
        ".private_segment_fixed_size": 1048, ".sgpr_count": 104,
        ".sgpr_spill_count": 527, ".symbol": name + ".kd",
        ".uniform_work_group_size": 1, ".uses_dynamic_stack": False,
        ".vgpr_count": 64, ".vgpr_spill_count": 467, ".wavefront_size": 64}


def yaml_lines(value, indent=0):
    prefix = " " * indent
    if type(value) is dict:
        result = []
        for key, val in value.items():
            if type(val) in (list, dict) and val:
                result.append(prefix + key + ":")
                result.extend(yaml_lines(val, indent + 2))
            else:
                result.append(prefix + key + ": " + scalar(val))
        return result
    if type(value) is list:
        result = []
        for val in value:
            if type(val) is dict and val:
                lines = yaml_lines(val, indent + 2)
                result.append(prefix + "- " + lines[0][indent + 2:])
                result.extend(lines[1:])
            else:
                result.append(prefix + "- " + scalar(val))
        return result
    raise AssertionError("bad synthetic YAML")


def scalar(value):
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return value
    return json.dumps(value)


def make_elf(kernels, code_overrides=None, extra_functions=None, omit_functions=(),
             descriptor_overrides=None, omit_descriptors=()):
    code_overrides = code_overrides or {}
    descriptor_overrides = descriptor_overrides or {}
    names = [name for name in kernels if name not in omit_functions] + list(extra_functions or [])
    default_code = struct.pack("<IIII", 0xBF800000, 0xBF8A0000, 0xBF800000, 0xBF810000)
    codes = {name: code_overrides.get(name, default_code) for name in names}
    addresses, text = {}, bytearray()
    for name in names:
        addresses[name] = 0x1000 + len(text)
        text.extend(codes[name])
    descriptors, rod = {}, bytearray()
    for name, kernel in kernels.items():
        if name in omit_descriptors:
            continue
        address = 0x2000 + len(rod)
        descriptors[name] = address
        blob = bytearray(64)
        struct.pack_into("<III", blob, 0, kernel.get(".group_segment_fixed_size", 32908),
                         kernel.get(".private_segment_fixed_size", 1048), kernel.get(".kernarg_segment_size", 24))
        struct.pack_into("<q", blob, 16, addresses.get(name, 0x1000) - address)
        v = max(0, (kernel.get(".vgpr_count", 64) + 3) // 4 - 1)
        s = max(0, (kernel.get(".sgpr_count", 104) + 7) // 8 - 1)
        struct.pack_into("<IIH", blob, 48, v | (s << 6), 0,
                         (int(kernel.get(".uses_dynamic_stack", False)) << 11)
                         | (int(kernel.get(".wavefront_size", 64) == 32) << 10))
        for offset, (fmt, value) in descriptor_overrides.get(name, {}).items():
            struct.pack_into(fmt, blob, offset, value)
        rod.extend(blob)
    strings = bytearray(b"\0")
    symtab = bytearray(24)
    for name in names:
        index = len(strings)
        strings.extend(name.encode() + b"\0")
        symtab.extend(struct.pack("<IBBHQQ", index, 0x12, 0, 1, addresses[name], len(codes[name])))
    for name, address in descriptors.items():
        index = len(strings)
        strings.extend((name + ".kd").encode() + b"\0")
        symtab.extend(struct.pack("<IBBHQQ", index, 0x11, 0, 2, address, 64))
    shstr = b"\0.text\0.rodata\0.symtab\0.strtab\0.shstrtab\0"
    image = bytearray(256)
    sections = [(0,) * 10]
    for label, content, typ, flags, address, link, align, entsize in (
        (b".text", text, 1, 6, 0x1000, 0, 256, 0),
        (b".rodata", rod, 1, 2, 0x2000, 0, 64, 0),
        (b".symtab", symtab, 2, 0, 0, 4, 8, 24),
        (b".strtab", strings, 3, 0, 0, 0, 1, 0),
        (b".shstrtab", shstr, 3, 0, 0, 0, 1, 0)):
        image.extend(b"\0" * ((-len(image)) % align))
        offset = len(image)
        image.extend(content)
        sections.append((shstr.index(label), typ, flags, address, offset, len(content), link, 0, align, entsize))
    image.extend(b"\0" * ((-len(image)) % 8))
    shoff = len(image)
    for section in sections:
        image.extend(struct.pack("<IIQQQQIIQQ", *section))
    ident = b"\x7fELF\x02\x01\x01\x40\x03" + b"\0" * 7
    struct.pack_into("<16sHHIQQQIHHHHHH", image, 0, ident, 3, 224, 1, 0, 0,
                     shoff, 0x2f, 64, 56, 0, 64, len(sections), 5)
    disasm = ["synthetic.out: file format elf64-amdgpu", "", "Disassembly of section .text:", ""]
    for name in names:
        disasm.append("%016x <%s>:" % (addresses[name], name))
        for offset in range(0, len(codes[name]), 4):
            word = struct.unpack_from("<I", codes[name], offset)[0]
            mnemonic = {0xBF8A0000: "s_barrier", 0xBF810000: "s_endpgm"}.get(word, "s_nop 0")
            disasm.append("    %-30s // %012X: %08X" % (mnemonic, addresses[name] + offset, word))
        disasm.append("")
    return bytes(image), "\n".join(disasm) + "\n"


class Fixture:
    def __init__(self):
        old_names = ["_Z19fixture_old_kernel%02dv" % i for i in range(11)] + [GATE.OLD17, GATE.OLD66]
        base = {name: metadata(name) for name in old_names}
        self.meta = {"fit17": base, "radix-stop": copy.deepcopy(base)}
        for name in (GATE.NEW17, GATE.NEW66):
            self.meta["radix-stop"][name] = metadata(name)
        self.remark_changes = {variant: {} for variant in VARIANTS}
        self.elf_options = {variant: {} for variant in VARIANTS}
        self.post = []

    def write(self, root):
        for variant in VARIANTS:
            kernels = self.meta[variant]
            doc = {"amdhsa.kernels": list(kernels.values()), "amdhsa.target": "amdgcn-amd-amdhsa--gfx906", "amdhsa.version": [1, 2]}
            (root / (variant + ".device.notes")).write_text(
                "Displaying notes found in: .note\n  AMDGPU Metadata:\n        ---\n" +
                "\n".join(yaml_lines(doc)) + "\n...\n")
            remarks = ["synthetic.cu:1:2: warning: irrelevant compiler warning"]
            for name, kernel in kernels.items():
                fields = {value: kernel.get(key, 0) for key, value in GATE.CROSS.items()}
                fields["Dynamic Stack"] = kernel.get(".uses_dynamic_stack", False)
                fields["Occupancy [waves/SIMD]"] = 4
                fields.update(self.remark_changes[variant].get(name, {}))
                marker = " [-Rpass-analysis=kernel-resource-usage]"
                prefix = "remark: qsa_select." + variant + ".cu:123:0: "
                remarks.append(prefix + "Function Name: " + name + marker)
                for key, value in fields.items():
                    if value is None:
                        continue
                    remarks.append(prefix + "    " + key + ": " + str(value) + marker)
            (root / (variant + ".resources.log")).write_text("\n".join(remarks) + "\n")
            image, disasm = make_elf(kernels, **self.elf_options[variant])
            (root / ELFNAME.format(variant)).write_bytes(image)
            (root / (variant + ".device.disasm")).write_text(disasm)
        for action in self.post:
            action(root)


def edit_text(filename, action):
    def edit(root):
        path = root / filename
        path.write_text(action(path.read_text()))
    return edit


def edit_binary(filename, action):
    def edit(root):
        path = root / filename
        data = bytearray(path.read_bytes())
        result = action(data)
        path.write_bytes(data if result is None else result)
    return edit


class ResourceGateTests(unittest.TestCase):
    checks=0
    def invoke(self,fixture,passed=False):
        with tempfile.TemporaryDirectory(prefix='radix-resource-gate-test-') as folder:
            root=pathlib.Path(folder);fixture.write(root)
            r=subprocess.run([sys.executable,str(HERE/'check-resources.py'),str(root)],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0 if passed else 14,r.stdout+r.stderr)
            report=json.loads((root/'resource-gate.json').read_text())
            self.assertEqual(report['passed'],passed);self.assertEqual(report['gate_passed'],passed)
            self.assertFalse(report['gpu_executed']);self.assertFalse(report['performance_claim']);self.assertFalse(report['correctness_claim'])
            self.assertTrue(report['manual_isa_review_required_before_any_gpu'])
            self.assertEqual(json.loads(r.stdout),report)
            if passed:
                self.assertEqual(len(report['unchanged_kernels']),13)
                self.assertEqual(len(report['comparisons']),2)
                for comparison in report['comparisons']:
                    self.assertIn(comparison['candidate'],(root/comparison['isa_file']).read_text())
            else:self.assertTrue(report['failures'])
            ResourceGateTests.checks+=1
    def test_valid_equal_and_improved(self):
        self.invoke(Fixture(),True)
        f=Fixture()
        for name in GATE.ADDITIONS:
            for key in GATE.CROSS:
                if key!='.group_segment_fixed_size':f.meta['radix-stop'][name][key]-=1
            f.remark_changes['radix-stop'][name]={'Occupancy [waves/SIMD]':5}
        self.invoke(f,True)
    def test_each_resource_regression(self):
        for name in GATE.ADDITIONS:
            for key in GATE.CROSS:
                f=Fixture();f.meta['radix-stop'][name][key]+=1;self.invoke(f)
    def test_each_layout_and_occupancy_regression(self):
        for name in GATE.ADDITIONS:
            for key,value in [('.wavefront_size',32),('.max_flat_workgroup_size',512),('.uses_dynamic_stack',True)]:
                f=Fixture();f.meta['radix-stop'][name][key]=value;self.invoke(f)
            f=Fixture();f.remark_changes['radix-stop'][name]={'Occupancy [waves/SIMD]':3};self.invoke(f)
    def test_each_missing_required_metadata_or_remark(self):
        for name in [GATE.OLD17,GATE.OLD66,GATE.NEW17,GATE.NEW66]:
            variant='fit17' if name in (GATE.OLD17,GATE.OLD66) else 'radix-stop'
            for key in GATE.CROSS:
                f=Fixture();del f.meta[variant][name][key];self.invoke(f)
            for remark in list(GATE.CROSS.values())+['Occupancy [waves/SIMD]','Dynamic Stack']:
                f=Fixture();f.remark_changes[variant][name]={remark:None};self.invoke(f)
    def test_crosscheck_disagreement(self):
        for name in GATE.ADDITIONS:
            for remark in GATE.CROSS.values():
                f=Fixture();f.remark_changes['radix-stop'][name]={remark:123};self.invoke(f)
    def test_every_old_machine_body(self):
        for name in Fixture().meta['fit17']:
            f=Fixture();f.elf_options['radix-stop']['code_overrides']={name:struct.pack('<IIII',0xBF800001,0xBF8A0000,0xBF800000,0xBF810000)};self.invoke(f)
    def test_old_metadata_arguments_and_other_fields(self):
        f=Fixture();f.meta['radix-stop'][GATE.OLD17]['.args'][0]['.actual_access']='read_write';self.invoke(f)
        f=Fixture();f.meta['radix-stop'][GATE.OLD66]['.gfx1250_revision']='B1';self.invoke(f)
        f=Fixture();f.meta['radix-stop'][GATE.OLD17]['.extra_field']='changed';self.invoke(f)
    def test_old_remarks(self):
        f=Fixture();f.remark_changes['radix-stop'][GATE.OLD17]={'Unexpected field':1};self.invoke(f)
        f=Fixture();f.remark_changes['radix-stop'][GATE.OLD66]={'Occupancy [waves/SIMD]':5};self.invoke(f)
    def test_kernel_sets(self):
        f=Fixture();del f.meta['radix-stop'][GATE.NEW17];self.invoke(f)
        f=Fixture();f.meta['radix-stop']['_Zextra']=metadata('_Zextra');self.invoke(f)
        f=Fixture();f.elf_options['radix-stop']['extra_functions']=['_Zextra'];self.invoke(f)
        f=Fixture();f.elf_options['radix-stop']['omit_functions']=[GATE.NEW66];self.invoke(f)
    def test_missing_inputs(self):
        for variant in VARIANTS:
            for filename in [variant+'.device.notes',variant+'.resources.log',variant+'.device.disasm',ELFNAME.format(variant)]:
                f=Fixture();f.post.append(lambda root,name=filename:(root/name).unlink());self.invoke(f)
    def test_truncated_and_invalid_elf(self):
        for variant in VARIANTS:
            for length in [0,24,64,-10]:
                f=Fixture();f.post.append(edit_binary(ELFNAME.format(variant),lambda b,n=length:b[:n]));self.invoke(f)
            f=Fixture();f.post.append(edit_binary(ELFNAME.format(variant),lambda b:struct.pack_into('<H',b,18,62)));self.invoke(f)
    def test_malformed_metadata_and_remarks(self):
        f=Fixture();f.post.append(edit_text('radix-stop.device.notes',lambda s:s.replace('...','')));self.invoke(f)
        f=Fixture();f.post.append(edit_text('radix-stop.device.notes',lambda s:s.replace('gfx906','gfx908')));self.invoke(f)
        f=Fixture();f.post.append(edit_text('radix-stop.device.notes',lambda s:s.replace('.sgpr_count: 104','.sgpr_count: 104\n    .sgpr_count: 104',1)));self.invoke(f)
        f=Fixture();f.post.append(edit_text('radix-stop.resources.log',lambda s:s.replace('TotalSGPRs: 104','TotalSGPRs: unknown',1)));self.invoke(f)
    def test_no_barrier_or_endpgm(self):
        for word in [0xBF8A0000,0xBF810000]:
            f=Fixture();f.elf_options['radix-stop']['code_overrides']={GATE.NEW17:struct.pack('<IIII',*[0xBF800000 if x==word else x for x in [0xBF800000,0xBF8A0000,0xBF800000,0xBF810000]])};self.invoke(f)
    def test_qualified_pinned_parser(self):
        p=HERE/'provenance'
        if not p.exists():return
        self.assertEqual(len(GATE.metadata(p/'qualified-fit17.device.notes')),13)
        self.assertEqual(len(GATE.remarks(p/'qualified-fit17.resources.log')),13)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ResourceGateTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed':result.wasSuccessful(),'test_methods':result.testsRun,'cli_fixture_checks':ResourceGateTests.checks,'gpu_executed':False,'actual_compiled_images_tested':False},indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
