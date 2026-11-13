#!/usr/bin/env python3
import pathlib,struct,tempfile,unittest
from validate_200k import read_200k,validate_pair

def fixture(device=0,pos=200000,T=4):
 stage=(0,27,23) if device==0 else (27,48,47)
 h=[1,0x01020304,device,*stage,pos,T,6,51202,2051,4,4,2048,128,4,204808,8204,16]
 steps=[(pos+t,pos+t+1,(pos+t+1)//4,2051) for t in range(T)]
 b=bytearray(b'QSDUMP01'+struct.pack('<19q',*h))
 for step in steps:b+=struct.pack('<4i',*step)
 for p,n,nb,w in steps:
  b+=struct.pack('<2q',nb+1,w)+struct.pack('<I',0x3e800000)*(nb+1)+struct.pack('<2051i',*range(2051))
 return bytes(b)
class ValidateTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def write(self,name,b):p=self.p/name;p.write_bytes(b);return p
 def test_actual_rows_and_window(self):
  for pos,T in [(200000,2),(200000,4),(200258,6)]:
   a=self.write('d0',fixture(0,pos,T));b=self.write('d1',fixture(1,pos,T));r=validate_pair([a,b]);self.assertTrue(r['passed']);self.assertEqual(r['files'][0]['header']['T'],T)
 def test_reject_old64k_and_limits(self):
  for pos,T in [(65536,4),(199999,4),(200259,6),(200000,7)]:
   with self.assertRaises(ValueError):read_200k(self.write('bad',fixture(0,pos,T)))
 def test_reject_malformed_header_and_payload(self):
  base=fixture()
  for index,value in [(2,2),(5,24),(8,8),(9,51201),(10,2052),(16,204804),(18,20)]:
   b=bytearray(base);struct.pack_into('<q',b,8+index*8,value)
   with self.assertRaises(ValueError):read_200k(self.write('bad',b))
  for bad in [base[:-1],base+b'x',b'BADDUMP!'+base[8:]]:
   with self.assertRaises(ValueError):read_200k(self.write('bad',bad))
 def test_paired_metadata_and_duplicate_rejection(self):
  a=self.write('d0',fixture());b=self.write('d1',fixture(1,200001,4))
  with self.assertRaises(ValueError):validate_pair([a,b])
  with self.assertRaises(ValueError):validate_pair([a,a])
  b.write_bytes(fixture(0))
  with self.assertRaises(ValueError):validate_pair([a,b])
 def test_fake_200k_header_rejects_old_steps(self):
  b=bytearray(fixture(0,65536,4));struct.pack_into('<q',b,8+6*8,200000)
  with self.assertRaises(ValueError):read_200k(self.write('fake',b))
 def test_id_exactness_not_just_shape(self):
  b=bytearray(fixture());off=160+4*16+16+50001*4;struct.pack_into('<i',b,off,1)
  with self.assertRaises(ValueError):read_200k(self.write('ids',b))
if __name__=='__main__':unittest.main()
