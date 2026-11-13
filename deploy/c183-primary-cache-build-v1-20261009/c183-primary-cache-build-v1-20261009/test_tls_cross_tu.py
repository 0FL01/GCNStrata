"""Ordinary g++ CPU fixture only; never compiles project/HIP or runs model/GPU."""
import os,subprocess,tempfile,unittest
from pathlib import Path
import tls_verify as T
class TestCrossTU(unittest.TestCase):
 def test_real_two_tu_coalescence_and_thread_isolation(self):
  header=Path(__file__).resolve().parent/'patched/include'
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   (root/'a.cpp').write_text('#include "strata/prefill/hc_negative_cache.hpp"\nextern "C" void set_current(strata::prefill::hc_negative::Request* p) { strata::prefill::hc_negative::current=p; }\n')
   (root/'b.cpp').write_text('#include "strata/prefill/hc_negative_cache.hpp"\nextern "C" strata::prefill::hc_negative::Request* get_current() { return strata::prefill::hc_negative::current; }\n')
   (root/'main.cpp').write_text('#include "strata/prefill/hc_negative_cache.hpp"\n#include <thread>\nextern "C" void set_current(strata::prefill::hc_negative::Request*);\nextern "C" strata::prefill::hc_negative::Request* get_current();\nint main(){ using namespace strata::prefill::hc_negative; std::atomic_flag gate=ATOMIC_FLAG_INIT; Request r(gate,{},false); if(get_current())return 1;set_current(&r);if(get_current()!=&r||current!=&r)return 2;int bad=0;std::thread t([&]{if(get_current())bad=1;set_current(&r);if(get_current()!=&r)bad=2;set_current(nullptr);});t.join();if(bad||get_current()!=&r)return 3;set_current(nullptr);return get_current()!=nullptr;}\n')
   for n in ('a','b','main'):subprocess.run(['g++','-O3','-std=c++20','-I'+str(header),'-c',str(root/(n+'.cpp')),'-o',str(root/(n+'.o'))],check=True,capture_output=True,timeout=30)
   subprocess.run(['g++','-pthread',*[str(root/(n+'.o')) for n in ('a','b','main')],'-o',str(root/'fixture')],check=True,capture_output=True,timeout=30)
   subprocess.run([str(root/'fixture')],check=True,capture_output=True,timeout=10)
   def run(label,argv):
    result=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=10);(root/(label+'.stdout')).write_text(result.stdout)
   result=T.verify(root/'fixture',[root/'a.o',root/'b.o'],root,run)
   self.assertTrue(result['passed']);self.assertEqual(len(result['objects']),2)
if __name__=='__main__':unittest.main()
