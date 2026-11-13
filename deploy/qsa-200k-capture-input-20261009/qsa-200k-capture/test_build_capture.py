"""CPU-only pure validation tests. Never starts a tool, compiler, container or GPU."""
import copy
import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import pathlib
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location('capture_build', HERE / 'build_capture.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BuildSpecTests(unittest.TestCase):
    def setUp(self):
        # All tests run with process creation prohibited, including accidental calls.
        self.guards = [mock.patch.object(build.subprocess, name, side_effect=AssertionError('No subprocess allowed'))
                       for name in ('run', 'Popen', 'check_output')]
        for guard in self.guards:
            guard.start(); self.addCleanup(guard.stop)
        self.package = dict(schema_version=1, patch_sha256='a' * 64, patched_verify_sha256='b' * 64)

    def receipt(self):
        return dict(build.pinned_fields(), schema_version=1, purpose=build.PURPOSE,
                    patch_sha256=self.package['patch_sha256'], patched_verify_sha256=self.package['patched_verify_sha256'],
                    gpu_devices=[], network='none', exit_code=0, api_touched=False, promoted=False,
                    compiler_confirmed_stopped=True, sources_restored=True, sources_invalidated=True, build_artifacts_restored=True,
                    patch_path='/local/bundle/qsa-200k-capture.patch',
                    select_object_unchanged=True, kernel_archive_unchanged=True, kernel_rebuild_detected=False,
                    select_object_path=str(build.O / 'qsa_select.cu.o'),
                    kernel_archive_path=str(build.O / 'libstrata_kernels.a'), binary_path=str(build.O / 'strata'),
                    compiled_sources=dict(build.SOURCES, **{build.VERIFY: self.package['patched_verify_sha256']}),
                    binary_sha256='c' * 64, package_sha256='d' * 64, restore_failed_files=[])

    def test_default_main_validates_only(self):
        with tempfile.TemporaryDirectory() as temp:
            package = pathlib.Path(temp) / 'package.json'
            package.write_text(json.dumps(self.package))
            output = io.StringIO()
            with mock.patch.object(sys, 'argv', ['build_capture.py', str(package)]), \
                 mock.patch.object(build, 'execute', side_effect=AssertionError('Execution forbidden')), \
                 contextlib.redirect_stdout(output):
                build.main()
            self.assertEqual(json.loads(output.getvalue()),
                             dict(spec_valid=True, executed=False, remote_state_checked=False))

    def test_spec_is_pure_and_pins_identity(self):
        self.assertTrue(build.validate_spec(self.package))
        for key, value in build.pinned_fields().items():
            package = dict(self.package, **{key: 'wrong'})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_spec(package)

    def test_bad_hashes_and_noop_patch_rejected(self):
        for key in ('patch_sha256', 'patched_verify_sha256'):
            for value in (None, '', 'z' * 64, 'A' * 64, 'a' * 63):
                with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                    build.validate_spec(dict(self.package, **{key: value}))
        with self.assertRaises(RuntimeError):
            build.validate_spec(dict(self.package, patched_verify_sha256=build.SOURCES[build.VERIFY]))
        with self.assertRaises(RuntimeError):
            build.validate_spec(dict(self.package, schema_version=True))

    def test_exact_seven_sources(self):
        self.assertEqual(len(build.SOURCES), 7)
        self.assertEqual(build.SOURCES[build.SELECT], '8d836acd211b7e493cd50bd680d128489f3f25f94ea26725afbab2402ef899bc')
        self.assertEqual(build.SOURCES[build.VERIFY], 'e0562925ea41f430193df5c326c7ddec743a59b1e43bf5d2545040475053c292')

    def test_receipt_success_and_binary_package_binding(self):
        receipt = self.receipt()
        self.assertTrue(build.validate_receipt(receipt, self.package, 'c' * 64, 'd' * 64))
        for binary, package in (('e' * 64, 'd' * 64), ('c' * 64, 'e' * 64)):
            with self.assertRaises(RuntimeError):
                build.validate_receipt(receipt, self.package, binary, package)

    def test_incomplete_receipts_fail_closed(self):
        required = ('exit_code', 'compiler_confirmed_stopped', 'sources_restored', 'build_artifacts_restored',
                    'select_object_unchanged', 'kernel_archive_unchanged', 'api_touched', 'promoted', 'sources_invalidated', 'patch_path',
                    'kernel_rebuild_detected', 'base_archive_snapshot_json_sha256')
        for key in required:
            receipt = self.receipt(); del receipt[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_receipt(receipt, self.package)
        for key in ('sources_restored', 'build_artifacts_restored', 'compiler_confirmed_stopped',
                    'select_object_unchanged', 'kernel_archive_unchanged'):
            receipt = self.receipt(); receipt[key] = False
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_receipt(receipt, self.package)

    def test_recompile_is_allowed_only_with_identical_used_bytes(self):
        receipt = self.receipt(); receipt['kernel_rebuild_detected'] = True
        self.assertTrue(build.validate_receipt(receipt, self.package))
        receipt['kernel_archive_sha256'] = 'e' * 64
        with self.assertRaises(RuntimeError):
            build.validate_receipt(receipt, self.package)

    def test_noop_binary_source_drift_and_errors_rejected(self):
        for key, value in (('binary_sha256', build.BASE_SHA), ('error_type', 'RuntimeError'),
                           ('restore_failed_files', [build.VERIFY]), ('network', 'host'),
                           ('gpu_devices', ['/dev/kfd']), ('exit_code', False)):
            receipt = self.receipt(); receipt[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                build.validate_receipt(receipt, self.package)
        receipt = copy.deepcopy(self.receipt()); receipt['compiled_sources'][build.SELECT] = 'e' * 64
        with self.assertRaises(RuntimeError):
            build.validate_receipt(receipt, self.package)

    def test_patch_is_only_one_text_source(self):
        patch = '--- a/src/core/verify.cpp\n+++ b/src/core/verify.cpp\n@@ -1 +1 @@\n-old\n+new\n'
        build.validate_patch_text(patch)
        build.validate_patch_text('diff --git a/src/core/verify.cpp b/src/core/verify.cpp\n' + patch)
        for bad in (patch + patch, patch.replace('verify.cpp', 'verify.hpp'),
                    patch + 'new file mode 100644\n', patch + 'GIT binary patch\n'):
            with self.assertRaises(RuntimeError):
                build.validate_patch_text(bad)

    def test_kernel_actions_recorded_without_false_link_positive(self):
        for line in ('[20%] Building HIP object CMakeFiles/strata_kernels.dir/src/kernels/cuda/qsa_select.cu.o',
                     '[50%] Linking HIP static library libstrata_kernels.a',
                     '/usr/bin/ar qc libstrata_kernels.a CMakeFiles/strata_kernels.dir/a.o',
                     '/opt/rocm/bin/hipcc -c /src/src/kernels/cuda/qsa_select.cu',
                     '/opt/rocm/llvm/bin/clang++ -c /src/src/kernels/cuda/qsa_select.cu'):
            with self.subTest(line=line):
                self.assertTrue(build.kernel_action(line))
        for line in ('/usr/bin/c++ -o strata main.o libstrata_kernels.a',
                     '[90%] Building CXX object CMakeFiles/strata.dir/src/core/verify.cpp.o'):
            self.assertFalse(build.kernel_action(line))

    def test_command_pinned_gpu_free_network_free_and_shared_locks(self):
        command = build.build_command()
        self.assertIn(build.IMAGE, command)
        self.assertEqual(command[command.index('--network') + 1], 'none')
        self.assertNotIn('--device', command)
        self.assertNotIn('--gpus', command)
        self.assertEqual(command[-1], 'cmake --build build-906 --target strata -j4 --verbose')
        self.assertEqual(len(build.LOCKS), 2)
        self.assertEqual(build.BASE_ARCHIVE, build.BASE / 'libstrata_kernels.a')
        self.assertNotEqual(build.BASE_ARCHIVE, build.W / build.ARCHIVE_REL)


if __name__ == '__main__':
    unittest.main()
