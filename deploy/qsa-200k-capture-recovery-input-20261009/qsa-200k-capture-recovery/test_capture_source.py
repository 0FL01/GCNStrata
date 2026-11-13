#!/usr/bin/env python3
import pathlib,re,unittest,hashlib
P=pathlib.Path(__file__).resolve().parent
class SourceTests(unittest.TestCase):
 def test_one_file_no_abi(self):
  patch=(P/'qsa-200k-capture.patch').read_text();s=(P/'verify.capture.cpp').read_text();base=(P/'verify.base.cpp').read_text()
  self.assertEqual(re.findall(r'^\+\+\+ (.+)$',patch,re.M),['b/src/core/verify.cpp'])
  self.assertNotIn('Verifier::dump_qsa_selection_once',s)
  self.assertNotIn('qsa_selection_dumped_',s)
  self.assertEqual(s[s.index('struct Bump {'):s.index('struct Bump {')+400],base[base.index('struct Bump {'):base.index('struct Bump {')+400])
  self.assertIn('static std::atomic<unsigned char> dump_state[2]{};',s)
  self.assertIn('compare_exchange_strong(expected_state, 1)',s)
  self.assertIn('dump_state[device_].store(2)',s)
  self.assertNotIn('dump_state[device_].store(0)',s)
 def test_safe_location_and_bounded_diagnostic(self):
  s=(P/'verify.capture.cpp').read_text();run=s[s.index('bool Verifier::run('):s.index('bool Verifier::run(',s.index('bool Verifier::run(')+1) if s.count('bool Verifier::run(')>1 else len(s)]
  start=run.index('auto dump_qsa_selection_once =');end=run.index('if (!dump_qsa_selection_once()) return false;');b=run[start:end]
  self.assertLess(run.index('const cudaError_t se = cudaStreamSynchronize(cs_)'),start)
  self.assertLess(run.index('dump_copy_status = cudaStreamSynchronize(copy_)'),start)
  self.assertLess(end,run.index('next_->run(T,'))
  self.assertLess(b.index('cudaStreamIsCapturing'),b.index('cudaMemcpy('))
  self.assertIn('status != cudaStreamCaptureStatusNone',b)
  self.assertIn('pos0 < 200000',b);self.assertIn('pos0 + T - 1 > 200263',b)
  self.assertIn('device_ < 0 || device_ > 1',b)
  for forbidden in ['cudaMalloc','cudaStreamSynchronize','cudaStreamBeginCapture','<<<','cudaMemcpyHostToDevice','cudaMemcpyDeviceToDevice']:
   self.assertNotIn(forbidden,b)
  self.assertEqual(b.count('cudaMemcpyDeviceToHost'),3)
 def test_private_creation_owned_cleanup_and_format(self):
  s=(P/'verify.capture.cpp').read_text()
  self.assertIn('O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600',s)
  self.assertIn('(directory.st_mode & 0777) != 0700',s)
  self.assertIn('now.st_dev == identity.st_dev && now.st_ino == identity.st_ino',s)
  self.assertIn('owned.complete = true;',s)
  self.assertIn('::openat(dir_fd, leaf.c_str()',s)
  self.assertIn('::fstatat(directory, leaf.c_str()',s)
  self.assertIn('::unlinkat(directory, leaf.c_str(), 0)',s)
  self.assertNotIn('std::remove(path.c_str())',s)
  self.assertIn("{'Q', 'S', 'D', 'U', 'M', 'P', '0', '1'}",s)
 def test_original_consumer_parser_bytes(self):
  self.assertEqual(hashlib.sha256((P/'qsa_topk_fixture_replay.cpp').read_bytes()).hexdigest(),'0eac4ceb11f783e14977f35afdbe0bf6c39fbddaa0fe8c39675e4a0f346a667e')
  self.assertEqual(hashlib.sha256((P/'analyze_qsa_dump.py').read_bytes()).hexdigest(),'53ad4c4d185d761aa35656e463d34b07a9c7bc91deeafb0e36e2b8bc3ea3daaa')
if __name__=='__main__':unittest.main()
