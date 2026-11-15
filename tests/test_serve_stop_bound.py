#!/usr/bin/env python3
"""Host-only stop-bound consumer regression; no GPU, SDK, or model required.

Run: python3 tests/test_serve_stop_bound.py (CXX, default c++, must support C++17).
Compiles narrow expressions/loops from the actual serial serve path against a
recording verifier/policy. Also binds pipeline commits and rollback replay.
This checks host accounting, not GPU ordering or continuation equivalence.

Negative control: copy this test and include/strata/core/serve_stop_bound.hpp
into a temporary source tree; populate src/program/generate.cpp with
`git show c723df72:src/program/generate.cpp`, then run the copied test. The old
callsite compiles but fails the committed-input-rows vs emitted-outputs check.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
gen = (ROOT / "src/program/generate.cpp").read_text()


def between(text, start, end):
    pos = text.index(start)
    return text[pos:text.index(end, pos)]


def balanced(text, marker, opening, closing):
    start = text.index(marker)
    pos = text.index(opening, start) + 1
    depth = 1
    while depth:
        depth += (text[pos] == opening) - (text[pos] == closing)
        pos += 1
    return text[start:pos]


def line(text, marker):
    lines = [s.strip() for s in text.splitlines() if marker in s and not s.lstrip().startswith("//")]
    assert len(lines) == 1, f"expected one source binding for {marker!r}, got {len(lines)}"
    return lines[0]


serial = between(gen, "int a = merged ? T - 1 : 0;", "const double decode_ms =")
accept = serial[:serial.index("if (from_sfx)")]
commit = balanced(serial, "ver.commit(", "(", ")")
consume = line(serial, "consumed.push_back(window[(size_t) i]);")
stats = balanced(serial, "if (!merged) {", "{", "}")
emit = balanced(serial, "for (int i = merged ?", "{", "}")
observe = between(serial, "if (timed_round && !eos && chain_n == 0)", "if (eos) { finish")
pending = line(serial, "x = outv[") + "\n" + line(serial, "p += ")

preamble = r'''
#include "strata/core/serve_stop_bound.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CHECK(x, why) do { if (!(x)) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
namespace strata { namespace core { void progress_beat() {} }}
struct Verifier {
    int rows = -1;
    bool commit(int n, std::string&, const void*) { rows = n; return true; }
};
struct Policy {
    int accepted = -1, calls = 0, chain_calls = 0;
    void observe(bool, int, int a, int, double) { accepted = a; ++calls; }
    void observe_chain(int, int, int a, int, double) { accepted = a; ++chain_calls; }
};
struct Suffix { void append(int32_t) {} };
void extra_sources_append(const int32_t*, int) {}
static void check_serial(bool merged, int chain_n) {
    const int T = merged ? 3 : 2, T_mtp = T - chain_n;
    // Two accepted input rows / one accepted DRAFT, with only one output left.
    // In merged mode, an EOS in an unexposed prompt row must not cut the commit.
    const std::vector<int32_t> window = merged ? std::vector<int32_t>{10, 20, 30}
                                               : std::vector<int32_t>{10, 20};
    const std::vector<int32_t> outv = merged ? std::vector<int32_t>{-1, 99, 40}
                                             : std::vector<int32_t>{20, 30};
    struct Options { std::vector<int64_t> eos_ids{-1}; int lookup_chain = 0; } o;
    const int64_t max_new = 1;
    int64_t produced_n = 0, p = 100;
    int32_t x = window[0];
    std::vector<int32_t> consumed;
    Verifier ver;
    Policy policy;
    Suffix sfx;
    std::string err;
    const bool defer_primary_commit = false, from_sfx = false, sfx_on = false;
    const bool timed_round = !merged;
    const int sfx_match = 0, cm = 0, primary_context = 0;
    int draft_offered = 0, draft_accepted = 0;
    const auto round0 = Clock::now();
'''
checks = r'''
    CHECK(produced_n == 1, "native output budget must emit exactly one output");
    CHECK(a == T - 1, "raw verdict must remain accepted drafts, not bounded rows minus one");
    CHECK(ver.rows == (merged ? T : produced_n),
          "committed INPUT rows must equal emitted outputs (except merged prompt rows)");
    CHECK(consumed.size() == static_cast<size_t>(ver.rows), "consumed prefix must match committed rows");
    CHECK(consumed == (merged ? window : std::vector<int32_t>{10}), "only the input prefix enters cached history");
    CHECK(x == (merged ? 40 : 20), "last emitted output must stay pending");
    CHECK(p == 100 + ver.rows, "next position must advance by committed input rows");
    CHECK(draft_accepted == (merged ? 0 : 1), "acceptance statistics must keep the raw draft verdict");
    CHECK(policy.accepted == (merged ? -1 : 1), "policy must receive one accepted draft despite a one-row budget cut");
    CHECK(policy.calls == (!merged && chain_n == 0), "ordinary verdict must reach policy.observe");
    CHECK(policy.chain_calls == (!merged && chain_n != 0), "chained verdict must reach policy.observe_chain");
}
int main() {
    check_serial(false, 0);
    check_serial(false, 1);
    check_serial(true, 0);
    std::puts("PASS: host actual-source serial budget, raw policy verdict, and merged prompt bypass");
}
'''
program = "\n".join([preamble, accept, commit + ";", consume, stats,
                     "bool eos = false;", emit, observe, pending, checks])
with tempfile.TemporaryDirectory(prefix="strata-stop-bound-") as tmp:
    p = Path(tmp)
    (p / "test.cpp").write_text(program)
    subprocess.run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++17", "-O2",
                    "-I", str(ROOT / "include"), str(p / "test.cpp"), "-o", str(p / "test")], check=True)
    subprocess.run([str(p / "test")], check=True)

# Source bindings for pipeline stages the small host fixture does not execute.
pipeline = between(gen, "// ---- A's verdict", "// --adapt-async: the verdict's tick")
assert "const int kept_rows = strata::core::serve_stop_bound(a + 1, max_new - produced_n, outp.data()," in pipeline, \
    "pipeline must bound its raw accepted input rows by the remaining output budget"
assert "V1(A).pl_commit_async(kept_rows, err)" in pipeline, "last stage must commit kept_rows"
assert "VF(st, A).pl_commit_async(kept_rows, err)" in pipeline, "front-stage final commit must use kept_rows"
assert "undo_keep = kept_rows;" in pipeline, "rollback replay must retain the same kept_rows"
assert "i < kept_rows" in line(pipeline, "consumed.push_back(A.tok[i]);"), "pipeline consumed prefix must be bounded"
assert "i < kept_rows" in line(pipeline, "!nf && i"), "pipeline output loop must use kept_rows"
assert "x = outp[(size_t) kept_rows - 1];" in pipeline, "pipeline must leave its last output pending"
assert "p = A.p + kept_rows;" in pipeline, "pipeline position must advance by kept_rows"
assert "draft_accepted += a;" in pipeline, "pipeline acceptance statistics must retain raw drafts"
assert "policy.observe(A.sfx, A.T, a," in pipeline, "pipeline policy must retain raw drafts"
assert "const bool would = a == A.T - 1" in pipeline, "gate calibration must use the raw full-acceptance verdict"
assert "VF(st, undo_w).pl_commit_async(undo_keep, err)" in gen, "restored front stages must replay undo_keep"
print("PASS: host pipeline commit/consumed/output/replay and raw-verdict source bindings")
