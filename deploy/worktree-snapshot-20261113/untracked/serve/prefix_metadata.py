"""Opt-in metadata-only solo-dispatch diagnostics; no native or sampling changes.

Set STRATA_PREFIX_META=1 or create the sibling .strata-prefix-meta-enabled
marker before adapter startup. Token IDs stay in RAM, never in emitted records. This
compares submitted prompts, not native checkpoints or prompt-plus-output state.
All public hooks fail open. The caller must serialize native solo dispatches.
"""
from array import array
from collections import deque
import json
import os
from pathlib import Path
import sys

ENV = "STRATA_PREFIX_META"
MARKER_NAME = ".strata-prefix-meta-enabled"
try:
    MARKER_ENABLED = Path(__file__).with_name(MARKER_NAME).is_file()
except OSError:
    MARKER_ENABLED = False
HISTORY_ENTRIES = 4
# Also cap an unexpectedly large advertised context. Oversize inputs are not
# retained or truncated; the next record explicitly says its previous LCP is
# unavailable. 4 x 1,048,576 uint32s is 16 MiB of token-array payload (plus small array overhead).
HARD_TOKEN_CAP = 1_048_576
MAX_NUMBER = (1 << 53) - 1
PREFIX = "[prefix-meta] "
STATE_KEY = "_prefix_meta_state_v1"
FINISHES = frozenset(("stop", "length", "eos", "cancel", "cancelled", "nonfinite"))


def _number(value):
    """Bounded integral metadata only; never stringify arbitrary objects."""
    if type(value) is int and 0 <= value <= MAX_NUMBER:
        return value
    if type(value) is str and len(value) <= 16 and value.isascii() and value.isdigit():
        parsed = int(value)
        return parsed if parsed <= MAX_NUMBER else None
    return None


def _emit(engine, record):
    # One complete line, no interpolation of request content or exceptions.
    payload = PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    stream = getattr(engine, "log", None)
    if not callable(getattr(stream, "write", None)):
        stream = sys.stderr
    stream.write(payload)
    stream.flush()


def _lcp(stored, current):
    limit = min(len(stored), len(current))
    for index in range(limit):
        # Match the existing native dispatch's int(t) conversion.
        if stored[index] != int(current[index]):
            return index
    return limit


def _flags(head, image):
    # Inspect only the known generated command head, never the prompt payload.
    # Unknown keys, paths, seeds, bias values and strings are never logged.
    fields = head.split()[2:]
    if image and fields:
        fields = fields[:-1]
    ckpt, cvec = "default", "default"
    for field in fields:
        if field in ("ckpt=0", "ckpt=1"):
            ckpt = int(field[-1])
        elif field in ("cvec=0", "cvec=1"):
            cvec = int(field[-1])
    return {"ckpt_key": ckpt, "cvec_key": cvec}


def begin(engine, ids, max_new, head, image=False):
    """Call immediately AFTER a successful solo GEN/GENI write+flush."""
    try:
        if os.environ.get(ENV, "1" if MARKER_ENABLED else "0") != "1":
            engine.__dict__.pop(STATE_KEY, None)
            return None
        state = engine.__dict__.get(STATE_KEY)
        if state is None:
            state = {"seq": 0, "history": deque(), "last_pid": None, "last_n": None}
            engine.__dict__[STATE_KEY] = state
        state["seq"] += 1
        seq = state["seq"]
        pid = _number(getattr(getattr(engine, "proc", None), "pid", None))
        prompt_n = len(ids)
        context = _number(getattr(engine, "max_context", None)) or 0
        limit = min(context, HARD_TOKEN_CAP)
        history = state["history"]
        prev_lcp = best_lcp = best_seq = best_pid = best_n = None
        # Newest wins ties. Entries are previous submitted prompt snapshots,
        # including cancelled requests; they make no claim about native state.
        for old_seq, old_pid, old_ids in reversed(history):
            lcp = _lcp(old_ids, ids)
            if old_seq == seq - 1:
                prev_lcp = lcp
            if best_lcp is None or lcp > best_lcp:
                best_lcp, best_seq, best_pid, best_n = lcp, old_seq, old_pid, len(old_ids)
        if history:
            del old_ids  # Do not keep an evicted snapshot alive via the loop variable.
        record = {"v": 1, "event": "dispatch", "seq": seq,
                  "server_pid": os.getpid(), "engine_pid": pid,
                  "prompt_tokens": prompt_n, "max_new": _number(max_new),
                  "max_context": context, "history_limit_tokens": limit,
                  "previous_seq": seq - 1 if seq > 1 else None,
                  "previous_engine_pid": state["last_pid"],
                  "previous_prompt_tokens": state["last_n"],
                  "previous_lcp_tokens": prev_lcp,
                  "best_seq": best_seq, "best_engine_pid": best_pid,
                  "best_prompt_tokens": best_n, "best_lcp_tokens": best_lcp,
                  "compared_entries": len(history), "image": bool(image),
                  "retained": False}
        record.update(_flags(head, image))
        state["last_pid"], state["last_n"] = pid, prompt_n
        # Entries are the last FOUR dispatches, not the last four retainable
        # requests; a skipped oversize request still advances the window.
        while history and (history[0][0] <= seq - HISTORY_ENTRIES or
                           len(history[0][2]) > limit):
            history.popleft()
        # All entries must obey the current context bound after a context change.
        if any(len(entry[2]) > limit for entry in history):
            state["history"] = history = deque(entry for entry in history if len(entry[2]) <= limit)
        if prompt_n <= limit and limit > 0:
            # Evicted before allocation: at most four arrays, even transiently.
            snapshot = array("I", (int(token) for token in ids))
            if snapshot.itemsize != 4:
                raise ValueError("unsupported token width")
            history.append((seq, pid, snapshot))
            record["retained"] = True
        record["retained_entries"] = len(history)
        record["retained_tokens"] = sum(len(entry[2]) for entry in history)
        _emit(engine, record)
        return seq
    except Exception:
        # Diagnostics cannot prevent generation or expose exception text.
        return None


def observe(engine, seq, line, *, cancelled=False, drained=False):
    """Observe existing RESUME/DONE lines only; never consume or change them."""
    if seq is None or not isinstance(line, str) or not line.startswith(("RESUME ", "DONE ")):
        return
    try:
        fields = line.split()
        record = {"v": 1, "seq": seq, "server_pid": os.getpid(),
                  "engine_pid": _number(getattr(getattr(engine, "proc", None), "pid", None)),
                  "cancel_requested": bool(cancelled), "drained": bool(drained)}
        if fields[0] == "RESUME":
            if len(fields) < 2 or _number(fields[1]) is None:
                return
            record.update(event="resume", reused_tokens=_number(fields[1]))
        elif fields[0] == "DONE":
            if len(fields) < 6:
                return
            record.update(event="done", generated_tokens=_number(fields[1]),
                          prompt_tokens=_number(fields[2]),
                          finish=fields[5] if fields[5] in FINISHES else "other")
            if len(fields) >= 9:
                record["reused_tokens"] = _number(fields[8])
            if len(fields) >= 15:
                record["prompt_read_tokens"] = _number(fields[14])
        else:
            return
        _emit(engine, record)
    except Exception:
        pass
