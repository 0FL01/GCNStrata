#!/usr/bin/env python3
"""Host-only regression for watchdog progress during serial graph captures.

Run: python3 tests/test_verify_capture_progress.py
Requires Python 3 and a C++17 compiler (CXX, default c++). Compiles the actual
Verifier::capture method with the real progress header and a fake HIP runtime;
no GPU, SDK, model, or watchdog timeout is needed.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / 'src/core/verify.cpp').read_text()
start = src.index('bool Verifier::capture(int T, std::string& err)')
pos = src.index('{', start) + 1
depth = 1
while depth:
    depth += (src[pos] == '{') - (src[pos] == '}')
    pos += 1
method = src[start:pos]

preamble = r'''
#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "strata/core/progress.hpp"
using namespace strata::core;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
using cudaError_t = int;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
using hipGraphNode_t = void*;
using hipGraphNodeType = int;
struct hipKernelNodeParams { void* func = nullptr; };
constexpr int cudaSuccess = 0, hipSuccess = 0, cudaStreamCaptureModeThreadLocal = 1;
constexpr int hipGraphNodeTypeKernel = 0, hipGraphNodeTypeMemcpy = 1, hipGraphNodeTypeMemset = 2;
enum Failure { none, begin, record, end, instantiate, upload, sync, upload_and_sync };
static Failure failure = none;
static uint64_t beats_before;
static int expected_t, graph_handle, exec_handle;
static std::vector<std::string> calls;

// Every fake operation observes the real counter: no early heartbeat.
static void step(const char* name) {
    CHECK(progress().beats.load() == beats_before);
    calls.emplace_back(name);
}
int cudaMemGetInfo(size_t* free_b, size_t* total_b) { step("memory"); *free_b = *total_b = 0; return 0; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int e) { return e ? "fake failure" : "success"; }
int cudaStreamBeginCapture(void*, int) { step("begin"); return failure == begin; }
int cudaStreamEndCapture(void*, cudaGraph_t* graph) { step("end"); *graph = &graph_handle; return failure == end; }
int cudaGraphDestroy(cudaGraph_t) { step("destroy"); return 0; }
int cudaGraphUpload(cudaGraphExec_t, void*) { step("upload"); return failure == upload || failure == upload_and_sync; }
int cudaStreamSynchronize(void*) { step("sync"); return failure == sync || failure == upload_and_sync; }

// Optional graph-node diagnostics compile but are disabled for this test.
int hipGraphGetNodes(cudaGraph_t, hipGraphNode_t*, size_t*) { std::abort(); }
int hipGraphNodeGetType(hipGraphNode_t, hipGraphNodeType*) { std::abort(); }
int hipGraphKernelNodeGetParams(hipGraphNode_t, hipKernelNodeParams*) { std::abort(); }
const char* hipKernelNameRefByPtr(void*, void*) { std::abort(); }

struct Verifier {
    bool ar_off_ = false;
    cudaGraphExec_t exec_[4] = {}, exec_nr_[4] = {};
    void* cs_ = nullptr;
    bool capture(int T, std::string& err);
    bool record_window(int T, void*, std::string& err) {
        step("record"); CHECK(T == expected_t);
        if (failure == record) { err = "record failure"; return false; }
        return true;
    }
    bool instantiate_evicting(cudaGraphExec_t& ex, cudaGraph_t, const std::vector<int>&,
                              const char*, std::string& err) {
        step("instantiate");
        if (failure == instantiate) { err = "instantiate failure"; return false; }
        ex = &exec_handle; return true;
    }
};
'''

tests = r'''
int main() {
    unsetenv("STRATA_VERIFY_NODES");
    progress().beats.store(17);  // Existing request progress must be incremented, not reset.
    const std::vector<std::string> completed = {
        "memory", "begin", "record", "end", "instantiate", "destroy", "upload", "sync"
    };
    Verifier serial;
    for (bool doorbell : {false, true}) {
        serial.ar_off_ = doorbell;
        for (int T : {1, 2, 3}) {
            expected_t = T; beats_before = progress().beats.load(); calls.clear();
            std::string err = "untouched";
            CHECK(serial.capture(T, err));
            CHECK(calls == completed);
            CHECK(progress().beats.load() == beats_before + 1);
            CHECK(err == "untouched");

            // Cached graphs change neither the heartbeat nor the stage diagnostic.
            progress_at("cached sentinel", 91, 92);
            progress().since_ms.store(-123);
            beats_before = progress().beats.load(); calls.clear();
            CHECK(serial.capture(T, err));
            CHECK(calls.empty());
            CHECK(progress().beats.load() == beats_before);
            CHECK(std::string(progress().where.load()) == "cached sentinel");
            CHECK(progress().detail.load() == 91 && progress().chunk.load() == 92);
            CHECK(progress().since_ms.load() == -123);
        }
    }
    std::puts("PASS: six serial new captures beat once each; cached hits preserve progress");

    for (bool doorbell : {false, true}) {
        for (Failure f : {begin, record, end, instantiate, upload, sync, upload_and_sync}) {
            Verifier v; v.ar_off_ = doorbell; failure = f;
            expected_t = 2; beats_before = progress().beats.load(); calls.clear();
            std::string err = "untouched";
            // Upload/sync errors are currently logged, but capture still returns true.
            CHECK(v.capture(expected_t, err) == (f >= upload));
            CHECK(progress().beats.load() == beats_before);
            if (f == begin) CHECK(err == "verify: begin capture failed");
            if (f == record) CHECK(err == "record failure");
            if (f == end) CHECK(err == "verify: end capture: fake failure");
            if (f == instantiate) CHECK(err == "instantiate failure");
            if (f >= upload) {
                CHECK(calls == completed);  // Sync still runs after an upload failure.
                CHECK(err == "untouched");
                CHECK((doorbell ? v.exec_nr_[2] : v.exec_[2]) != nullptr);
            }
        }
    }
    std::puts("PASS: begin/record/end/instantiate/upload/sync failures never beat; error behavior preserved");
}
'''

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)
    source = path / 'test.cpp'
    binary = path / 'test'
    source.write_text(preamble + '\n' + method + '\n' + tests)
    subprocess.run([
        *shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O2',
        '-DSTRATA_USE_HIP', '-I', str(ROOT / 'include'), str(source), '-o', str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
print('PASS: extracted Verifier::capture with real progress counters (host-only HIP build)')
