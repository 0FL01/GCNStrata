#!/usr/bin/env python3
"""One exact failed-restore exception; never edits a build receipt or executes recovery."""
import copy, json, pathlib, stat
import build_capture as B
ORIGINAL_BUILD='fe93c7e6588c88069ed318f22cc589f7f0fa44ae42ca6c903b862ef369ea3387'
ORIGINAL_PACKAGE='0c9fd8de4db8cd9153474c470197af1c1e3250390290239c5697565fbc8e5614'
CANDIDATE='0548008ed5c40a7e69ac65092e0f4767d6da70af1bc6cc72c2d391537c7b0fdd'
EVIDENCE='ca909cc67f1f10a47714774e55a3b0735d9cbd7711a3dabcf2c747132241bac1'
RECOVERY='3b8fa0813a5ed5fe603dbe9e34bc1c35646150897a96da5a3122c6210db0a37b'
MTIME=1791556136746099213
TARGET=B.W/B.BINARY_REL
SNAPSHOT=B.O/'before'/B.BINARY_REL

def expected_recovery():
 return dict(schema_version=1,purpose='qsa-200k-binary-restore-recovery',original_build_sha256=ORIGINAL_BUILD,
  original_package_sha256=ORIGINAL_PACKAGE,candidate_sha256=CANDIDATE,restore_target=str(TARGET),
  snapshot_path=str(SNAPSHOT),restored_sha256=B.BASE_SHA,restored_mtime_ns=MTIME,restored_mode=0o755,
  restoration_verified=True,gpu_executed=False,native_executed=False,image=B.IMAGE,network='none',evidence_sha256=EVIDENCE)

def validate_original(receipt,original_package,diagnostic_sha):
 B.need(diagnostic_sha==CANDIDATE,'only exact already-built candidate may recover')
 B.need(receipt.get('build_artifacts_restored') is False and receipt.get('restore_failed_files')==[B.BINARY_REL],
        'only original singleton binary-restore failure is eligible')
 B.need(receipt.get('kernel_rebuild_detected') is False,'unexpected kernel rebuild in this exact recovery')
 before=receipt['before'][B.BINARY_REL]
 B.need(before['sha256']==B.BASE_SHA and before['mtime_ns']==MTIME and before['size']==21589440,'original binary snapshot identity')
 # Validate every original success condition except precisely these two restoration fields.
 # This ephemeral view is never persisted and does not relabel the original failed build.
 view=copy.deepcopy(receipt);view['build_artifacts_restored']=True;view['restore_failed_files']=[]
 B.validate_receipt(view,original_package,CANDIDATE,ORIGINAL_PACKAGE)
 return True

def validate_recovery_document(recovery):
 for k,v in expected_recovery().items():
  B.need(k in recovery and recovery[k]==v and type(recovery[k]) is type(v),'recovery field: '+k)
 return True

def validate(receipt,original_package_path,recovery_path,recovery_sha,diagnostic_sha):
 original_package_path=pathlib.Path(original_package_path);recovery_path=pathlib.Path(recovery_path)
 B.need(recovery_sha==RECOVERY,'exact root recovery receipt SHA required')
 B.need(B.sha(B.O/'build.json')==ORIGINAL_BUILD,'original failed build receipt changed')
 B.need(B.sha(original_package_path)==ORIGINAL_PACKAGE,'original v2 package changed')
 B.need(recovery_path==B.O/'recovery.json' and not recovery_path.is_symlink() and B.sha(recovery_path)==recovery_sha,'exact recovery path/hash')
 original_package=json.loads(original_package_path.read_text())
 validate_original(receipt,original_package,diagnostic_sha)
 recovery=json.loads(recovery_path.read_text());validate_recovery_document(recovery)
 evidence_path=B.O/'binary-restore-evidence.json'
 B.need(B.sha(evidence_path)==EVIDENCE,'original root recovery evidence changed')
 evidence=json.loads(evidence_path.read_text())
 B.need(evidence['passed'] is True and evidence['original_build_sha256']==ORIGINAL_BUILD and
        evidence['candidate_sha256']==CANDIDATE and evidence['backup_sha256']==B.BASE_SHA and
        evidence['no_gpu'] is True and evidence['network']=='none' and evidence['no_chmod_chown'] is True,'root restoration evidence')
 B.verify_baseline() # Re-check archived baseline and live c183 binary/kernel artifacts.
 B.need(B.sha(B.O/'strata')==CANDIDATE,'diagnostic candidate changed')
 for f,before in receipt['before'].items():
  B.need(B.sha(B.W/f)==before['sha256'],'live pre-build bytes not restored: '+f)
 for f,h in receipt['compiled_sources'].items():
  B.need(B.sha(B.O/'compiled-source'/f)==h,'compiled source snapshot changed: '+f)
 B.need(B.sha(B.O/'qsa_select.cu.o')==B.SELECT_SHA and B.sha(B.O/'libstrata_kernels.a')==B.ARCHIVE_SHA,'candidate kernel snapshots changed')
 B.need(not TARGET.is_symlink() and not SNAPSHOT.is_symlink() and B.sha(SNAPSHOT)==B.BASE_SHA,'original restore snapshot changed')
 st=TARGET.stat();backup=SNAPSHOT.stat()
 B.need(stat.S_ISREG(st.st_mode) and st.st_size==21589440 and st.st_mtime_ns==MTIME and stat.S_IMODE(st.st_mode)==0o755,'live binary metadata not restored')
 B.need(backup.st_mtime_ns==MTIME,'backup original mtime changed')
 for k,v in [('sha256',B.BASE_SHA),('mtime_ns',MTIME),('mode',0o755),('uid',st.st_uid),('gid',st.st_gid)]:
  B.need(evidence['after'][k]==v,'live binary differs from recovery evidence: '+k)
 return {'accepted':True,'original_build_sha256':ORIGINAL_BUILD,'original_build_package_sha256':ORIGINAL_PACKAGE,
         'original_build_artifacts_restored':False,'original_restore_failed_files':[B.BINARY_REL],
         'recovery_receipt_sha256':recovery_sha,'root_evidence_sha256':EVIDENCE,'live_restoration_rechecked':True,
         'recompiled':False,'native_requests_before_recovery':0}
