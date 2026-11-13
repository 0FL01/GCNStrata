#!/usr/bin/env python3
import pathlib,hashlib,difflib,json
P=pathlib.Path(__file__).resolve().parent
base=(P/'verify.base.cpp').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='e0562925ea41f430193df5c326c7ddec743a59b1e43bf5d2545040475053c292'
old=(P/'dump-method.original.txt').read_text()
body=old[old.index('    const char* prefix'):old.rfind('\n}')]
a='''    const char* prefix = std::getenv("STRATA_QSA_SELECT_DUMP");
    if (!prefix || !prefix[0] || T < 2 || pos0 < 200000) return true;
    // Diagnostic process is one serial native request on these two verified devices.
    if (device_ < 0 || device_ > 1 || T > 6 || max_t_ != 6 || pos0 + T - 1 > 200263 ||
        max_blocks_ != 51202 || cap_ != 2051 ||
        !((device_ == 0 && lb_ == 0 && le_ == 27) || (device_ == 1 && lb_ == 27 && le_ == 48))) {
        err = "QSA 200K dump: unsupported device/stage/window geometry"; return false;
    }
    // Process-local per-device states: 0 unused, 1 claimed/failed, 2 completed.
    // A failed attempt never retries in this process. No Verifier member/layout changes.
    static std::atomic<unsigned char> dump_state[2]{};
    if (dump_state[device_].load() == 2) return true;
'''
body=body.replace('    const char* prefix = std::getenv("STRATA_QSA_SELECT_DUMP");\n    if (!prefix || !prefix[0] || qsa_selection_dumped_ || T < 2 || pos0 < 65536) return true;\n',a)
anchor='    using namespace strata::kernels;\n'
insert='''    // Existing HIP compatibility API. None is required, not merely "not Active".
    for (cudaStream_t stream : {cs_, copy_, sh_cs_}) {
        if (stream == nullptr) continue;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess || status != cudaStreamCaptureStatusNone) {
            err = "QSA 200K dump: stream capture active or status unavailable"; return false;
        }
    }
    unsigned char expected_state = 0;
    if (!dump_state[device_].compare_exchange_strong(expected_state, 1)) {
        err = "QSA 200K dump: repeated or concurrent device attempt"; return false;
    }
'''
body=body.replace(anchor,insert+anchor)
body=body.replace('    const QsaShapes s = shapes_of(*g_);','''    const QsaShapes s = shapes_of(*g_);
    if (s.idx_block != 4 || s.idx_top_k != 2048 || s.idx_dim != 128 || s.idx_n_head != 4 ||
        layer != (device_ == 0 ? 23 : 47)) {
        err = "QSA 200K dump: unsupported QSA shape/layer"; return false;
    }''')
body=body.replace('    std::FILE* f = std::fopen(path.c_str(), "wbx");  // refuse to overwrite an earlier receipt\n    if (!f) { err = "QSA selection dump: cannot create " + path; return false; }', '''    // Private parent directory and exclusive 0600 creation, even with a permissive umask.
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || path[0] != '/') {
        err = "QSA 200K dump: absolute output path required"; return false;
    }
    const std::string leaf = path.substr(slash + 1);
    const int dir_fd = ::open(path.substr(0, slash).c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir_fd < 0) { err = "QSA 200K dump: private output directory open failed"; return false; }
    struct OwnedDir { int fd; ~OwnedDir() { ::close(fd); } } directory_fd{dir_fd};
    struct stat directory{};
    if (::fstat(dir_fd, &directory) != 0 || !S_ISDIR(directory.st_mode) ||
        (directory.st_mode & 0777) != 0700 || directory.st_uid != ::geteuid()) {
        err = "QSA 200K dump: output parent must be owned private directory"; return false;
    }
    // Allocate bounded staging before file creation; no later vector growth can strand a partial file.
    std::vector<float> scores((size_t) max_blocks_);
    std::vector<int32_t> ids((size_t) cap_);
    const int fd = ::openat(dir_fd, leaf.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) { err = "QSA selection dump: cannot create " + path; return false; }
    struct stat created{};
    if (::fstat(fd, &created) != 0) { ::close(fd); err = "QSA 200K dump: fstat failed; retain unverified empty file"; return false; }
    struct OwnedDump {
        int directory; const std::string& leaf; struct stat identity; std::FILE* file; int raw_fd; bool complete = false;
        ~OwnedDump() {
            if (file) std::fclose(file); else if (raw_fd >= 0) ::close(raw_fd);
            struct stat now{};
            // Runner owns this fresh directory exclusively; no concurrent same-UID mutation.
            if (!complete && ::fstatat(directory, leaf.c_str(), &now, AT_SYMLINK_NOFOLLOW) == 0 &&
                now.st_dev == identity.st_dev && now.st_ino == identity.st_ino)
                ::unlinkat(directory, leaf.c_str(), 0); // Only this newly created incomplete file.
        }
    } owned{dir_fd, leaf, created, nullptr, fd};
    if ((created.st_mode & 0777) != 0600) { err = "QSA 200K dump: file is not private"; return false; }
    std::FILE* f = ::fdopen(fd, "wb");
    if (!f) { err = "QSA selection dump: fdopen failed"; return false; }
    owned.file = f; owned.raw_fd = -1;''')
body=body.replace('    std::vector<float> scores;\n    std::vector<int32_t> ids;\n','')
body=body.replace('        scores.resize((size_t) row[0]);\n        ids.resize((size_t) row[1]);\n','')
body=body.replace('scores.size() * sizeof(float)','(size_t) row[0] * sizeof(float)')
body=body.replace('ids.size() * sizeof(int32_t)','(size_t) row[1] * sizeof(int32_t)')
body=body.replace('    if (std::fclose(f) != 0) ok = false;','    if (std::fclose(f) != 0) ok = false;\n    owned.file = nullptr;')
body=body.replace('        std::remove(path.c_str());  // only this newly-created, incomplete file\n','')
body=body.replace('    qsa_selection_dumped_ = true;  // per verifier: stage 1 still records its own layer','    owned.complete = true;\n    dump_state[device_].store(2);  // Other GPU still gets its own one-shot capture.')
block='''    // Opt-in isolated 200K diagnostic. Existing streams are drained; graph capture has ended.
    auto dump_qsa_selection_once = [&]() -> bool {
'''+''.join('    '+line+'\n' for line in body.splitlines())+'''    };
    if (!dump_qsa_selection_once()) return false;
'''
anchor="    if (prof_on_ && G == 1) collect_profile();   // the window's GPU stage stamps\n"
assert base.count(anchor)==1
patched=base.replace(anchor,anchor+block)
patched=patched.replace('#include <exception>\n','#include <exception>\n#include <fcntl.h>\n#include <sys/stat.h>\n#include <unistd.h>\n')
oldcopy='''        cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
        copy_used_ = false;'''
newcopy='''        const cudaError_t dump_copy_status = cudaStreamSynchronize(copy_); // existing completion point
        if (const char* dump = std::getenv("STRATA_QSA_SELECT_DUMP"); dump && dump[0] && dump_copy_status != cudaSuccess) {
            err = "QSA 200K dump: copy stream completion failed"; return false;
        }
        copy_used_ = false;'''
assert patched.count(oldcopy)==1
patched=patched.replace(oldcopy,newcopy)
(P/'verify.capture.cpp').write_text(patched)
patch=''.join(difflib.unified_diff(base.splitlines(True),patched.splitlines(True),fromfile='a/src/core/verify.cpp',tofile='b/src/core/verify.cpp'))
(P/'qsa-200k-capture.patch').write_text(patch)
print(json.dumps({k:hashlib.sha256((P/v).read_bytes()).hexdigest() for k,v in [('patch_sha256','qsa-200k-capture.patch'),('patched_verify_sha256','verify.capture.cpp')]}))
