import unittest
import tls_verify as T
def symbols(bind='WEAK',typ='TLS',section='12',count=1):
 return "Symbol table '.symtab' contains 10 entries:\n"+('  1: 0000000000000000 8 '+typ+' '+bind+' DEFAULT '+section+' '+T.SYMBOL+'\n')*count
class Tests(unittest.TestCase):
 def test_defined_coalescible_tls(self):
  for bind in ('WEAK','UNIQUE'):self.assertEqual(T.check_symbol(symbols(bind),'fixture')['bind'],bind)
 def test_noncoalesced_or_missing_rejected(self):
  for value in [symbols('LOCAL'),symbols('GLOBAL'),symbols(typ='OBJECT'),symbols(section='UND'),symbols(count=0),symbols(count=2)]:
   with self.assertRaises(RuntimeError):T.check_symbol(value,'fixture')
 def test_dynsym_not_double_counted(self):
  text=symbols()+symbols().replace("'.symtab'","'.dynsym'")
  self.assertEqual(len(T.rows(text)),1)
 def test_objects_need_comdat_and_tls_reference(self):
  groups='COMDAT group section [ 1] `.group` ['+T.SYMBOL+'] contains 1 sections:\n';reloc='000000000 00000000 R_X86_64_TPOFF32 '+T.SYMBOL+' + 0'
  T.check_object(symbols(),groups,reloc,'fixture')
  for g,r in [('',reloc),(groups,''),(groups,reloc.replace('TPOFF32','PC32'))]:
   with self.assertRaises(RuntimeError):T.check_object(symbols(),g,r,'fixture')
if __name__=='__main__':unittest.main()
