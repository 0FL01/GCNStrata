#!/usr/bin/env python3
"""Create an audit-only generate.cpp from baseline or candidate; never edit input.

Adds bounded-row/early-commit/replay evidence and per-stage state fingerprints.
Not a release source. Use only with the approved, isolated GPU qualification.
"""
import argparse
from pathlib import Path


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one instrumentation anchor: {old[:90]!r}")
    return text.replace(old, new, 1)


def instrument(source):
    candidate = 'const int kept_rows = strata::core::serve_stop_bound(' in source
    keep = 'kept_rows' if candidate else 'a + 1'
    source = once(source, f'                    if (!V1(A).pl_commit_async({keep}, err))',
        f'''                    std::fprintf(stderr, "STOP_BOUND path=pipe seq=%d p=%lld T=%d accepted=%d kept=%d remaining=%lld B=%d com=%x\\n",
                                 A.seq, (long long) A.p, A.T, a + 1, {keep}, (long long) (max_new - produced_n), B.launched, A.com);
                    if (!V1(A).pl_commit_async({keep}, err))''')
    source = once(source, f'                if (!ver.commit({keep}, err, defer_primary_commit ? &primary_context : nullptr))',
        f'''                std::fprintf(stderr, "STOP_BOUND path=serial p=%lld T=%d accepted=%d kept=%d remaining=%lld merged=%d\\n",
                             (long long) p, T, a + 1, {keep}, (long long) (max_new - produced_n), merged);
                if (!ver.commit({keep}, err, defer_primary_commit ? &primary_context : nullptr))''')
    source = once(source, '                            if (!VF(st, undo_w).pl_commit_async(undo_keep, err)) return die(err);',
        '''                            if (!VF(st, undo_w).pl_commit_async(undo_keep, err)) return die(err);
                            std::fprintf(stderr, "STOP_UNDO stage=%d seq=%d keep=%d\\n", st, undo_w.seq, undo_keep);''')
    source = once(source, '                            A.com |= 1u << st;',
        '''                            A.com |= 1u << st;
                            std::fprintf(stderr, "STOP_EARLY stage=%d seq=%d rows=%d\\n", st, A.seq, A.T);''')

    start = source.index('            if (state_hash && !cancelled && o.prompt_cache > 0)')
    end = source.index('            const int64_t req_hits', start)
    block = source[start:end]
    block = once(block, '                // DEBUG:', '''                std::fprintf(stderr, "STOP_HISTORY ids=");
                for (int32_t id : live) std::fprintf(stderr, "%d,", id);
                std::fprintf(stderr, "\\n");
                auto& primary_ss = ss;
                for (int audit_stage = 0; audit_stage < n_stages; ++audit_stage) {
                auto& ss = audit_stage == 0 ? primary_ss : stages[(size_t) audit_stage - 1]->ss;
                const strata::core::OnDevice audit_device(stage_ver(audit_stage).device());
                // DEBUG:''')
    block = once(block, '                if (use_mtp) {', '''                int64_t mL = 0;
                if (use_mtp && audit_stage == 0) {
                    const strata::core::OnDevice draft_device(mtp.device());
                    if (cudaDeviceSynchronize() != cudaSuccess) {
                        std::printf("ERR synchronizing draft fingerprint\\n");
                        return 1;
                    }''')
    block = once(block, '                    const int64_t mL = std::min<int64_t>(L, ms.max_cells);',
        '''                    // The terminal window need not run the drafter: its final cell can be uncomputed.
                    // These short fixtures stay within the identity part of the draft ring.
                    mL = std::min<int64_t>(std::max<int64_t>(0, L - 1), ms.max_cells);''')
    block = once(block, '"strata serve: STATE_HASH L=%lld gdn=', '"STOP_STATE stage=%d draft_upto=%lld L=%lld gdn=')
    block = once(block, '(long long) L,\n', 'audit_stage, (long long) mL, (long long) L,\n')
    if not block.endswith('            }\n'):
        raise ValueError('unexpected state-fingerprint block end')
    block = block.removesuffix('            }\n') + '                }\n            }\n'
    return source[:start] + block + source[end:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path, help='new output file; existing files refused')
    args = parser.parse_args()
    result = instrument(args.source.read_text())
    with args.output.open('x') as output:
        output.write(result)
    print(f'Audit-only source: {args.output}')
