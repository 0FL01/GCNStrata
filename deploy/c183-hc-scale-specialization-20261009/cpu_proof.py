#!/usr/bin/env python3
"""Exhaustive host arithmetic proof; GPU NaN/zero compilation behavior remains a separate gate."""
import hashlib
import json
import math
from pathlib import Path
import struct

def f32_bits(x):
    try: return struct.unpack('<I',struct.pack('<f',x))[0]
    except OverflowError:return 0xff800000 if math.copysign(1.0,x)<0 else 0x7f800000

def f32(x):return struct.unpack('<f',struct.pack('<I',f32_bits(x)))[0]
def from_f32_bits(u):return struct.unpack('<f',struct.pack('<I',u))[0]

def half_bits(x):
    try:return struct.unpack('<H',struct.pack('<e',x))[0]
    except OverflowError:return 0xfc00 if math.copysign(1.0,x)<0 else 0x7c00

def from_half_bits(u):return struct.unpack('<e',struct.pack('<H',u))[0]

def prove():
    rows=[]
    for shift in [12,9]:
        scale=float(2**shift);inverse=2.0**(-shift)
        counts={'bf16_patterns_enumerated':0,'non_nan_output_and_guard_equivalent':0,
                'nan_input_patterns_gpu_required':0,'half_patterns_enumerated':0,
                'finite_half_inverse_bits_equal':0,'infinite_half_inverse_bits_equal':0,
                'nan_half_patterns_gpu_required':0,'negative_zero_forward_model_checks':0}
        digest=hashlib.sha256()
        for b in range(65536):
            counts['bf16_patterns_enumerated']+=1
            original=b<<16
            if b&0x7f80==0x7f80 and b&0x7f:
                counts['nan_input_patterns_gpu_required']+=1
                digest.update(struct.pack('<HB',b,255));continue
            value=from_f32_bits(original)
            converted=f32(value*scale)
            h=half_bits(converted);half_value=from_half_bits(h)
            back_div=f32_bits(half_value/scale);back_mul=f32_bits(half_value*inverse)
            assert back_div==back_mul,(shift,b,h,back_div,back_mul)
            flags_div=(int(not math.isfinite(value)),
                       int(not math.isfinite(converted) or abs(converted)>65504.0),
                       int(original!=back_div))
            flags_mul=flags_div[:2]+(int(original!=back_mul),)
            assert flags_div==flags_mul
            counts['non_nan_output_and_guard_equivalent']+=1
            if b==0x8000:
                assert h==0x8000 and back_div==0x80000000
                counts['negative_zero_forward_model_checks']+=1
            digest.update(struct.pack('<HHIBBB',b,h,back_div,*flags_div))
        # Every possible post-conversion half payload, independently of the forward converter.
        for h in range(65536):
            counts['half_patterns_enumerated']+=1
            if h&0x7c00==0x7c00 and h&0x3ff:
                counts['nan_half_patterns_gpu_required']+=1;continue
            value=from_half_bits(h)
            assert f32_bits(value/scale)==f32_bits(value*inverse),(shift,h)
            if math.isfinite(value):counts['finite_half_inverse_bits_equal']+=1
            else:counts['infinite_half_inverse_bits_equal']+=1
        assert counts['bf16_patterns_enumerated']==65536
        assert counts['non_nan_output_and_guard_equivalent']==65282
        assert counts['nan_input_patterns_gpu_required']==254
        assert counts['finite_half_inverse_bits_equal']==63488
        assert counts['infinite_half_inverse_bits_equal']==2
        assert counts['nan_half_patterns_gpu_required']==2046
        rows.append({'shift':shift,'scale':int(scale),'inverse_exact_hex':inverse.hex(),
                     'counts':counts,'non_nan_cpu_records_sha256':digest.hexdigest()})
    return {'scope':'CPU model and finite-value mathematical proof; no GPU execution',
            'cpu_non_nan_equivalence_passed':True,'all_gpu_bit_equivalence_established':False,
            'candidate_source_retains_division_expression':True,
            'finite_proof':'For any finite half h, h/2^k with k=9 or12 is exactly representable in FP32; nonzero magnitude ranges from2^-36 to127.9375. Multiplication by the exact inverse and division therefore have the same correctly rounded finite result. Signed zeros retain sign in this arithmetic model.',
            'limitations':['Host half/NaN conversion behavior is not evidence of AMD GPU NaN payload behavior.',
                           'Constant folding may alter forward conversion ISA, including signed-zero behavior. Every GPU output half bit and all guard words must match actual c183.',
                           'NaN inputs are enumerated and explicitly deferred to GPU comparison, not silently treated as CPU-proven.',
                           'No model correctness, runtime speed, or whole-model gain is established.'],
            'scales':rows}

if __name__=='__main__':
    result=prove();Path(__file__).with_name('cpu-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
