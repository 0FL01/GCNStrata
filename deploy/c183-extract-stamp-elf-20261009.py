#!/usr/bin/env python3
"""Read an exact existing HIP host object; copy its one gfx906 ELF without loading it."""
import hashlib, json, re, struct, sys
from pathlib import Path

EXPECTED_SHA = "50a3e527914aa51e6c85a6b6aa358625b4398bd1a9fb9ab06742c735194486c5"
EXPECTED_SIZE = 237736
MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"

def need(ok, message):
    if not ok:
        raise ValueError(message)

def chunk(data, start, size):
    need(start >= 0 and size >= 0 and start + size <= len(data), "out-of-bounds data")
    return data[start:start + size]

def number(data, offset, form):
    return struct.unpack(form, chunk(data, offset, struct.calcsize(form)))[0]

def hip_section(data):
    need(data[:7] == b"\x7fELF\x02\x01\x01", "expected ELF64 little-endian")
    need(number(data, 18, "<H") == 62, "expected x86-64 host object")
    off = number(data, 40, "<Q")
    entsize, count, names_index = struct.unpack("<HHH", chunk(data, 58, 6))
    need(entsize == 64 and off >= 64, "unsupported ELF section table")
    first = chunk(data, off, 64)
    if count == 0:
        count = number(first, 32, "<Q")
    if names_index == 65535:
        names_index = number(first, 40, "<I")
    need(0 < count < 65536 and names_index < count, "invalid ELF section count")
    table = chunk(data, off, count * 64)
    def section(index):
        return struct.unpack("<IIQQQQIIQQ", chunk(table, index * 64, 64))
    shstr = section(names_index)
    names = chunk(data, shstr[4], shstr[5])
    matches = []
    for i in range(count):
        sh = section(i)
        need(sh[0] < len(names), "invalid section name offset")
        end = names.find(b"\0", sh[0])
        need(end >= 0, "unterminated section name")
        if names[sh[0]:end] == b".hip_fatbin":
            need(sh[1] == 1, "unexpected .hip_fatbin section type")
            matches.append((sh[4], chunk(data, sh[4], sh[5])))
    need(len(matches) == 1, "expected exactly one .hip_fatbin")
    return matches[0]

def select_image(bundle):
    need(bundle.startswith(MAGIC), "unknown/compressed bundle; stop, do not guess")
    count = number(bundle, 24, "<Q")
    need(0 < count < 1024, "invalid bundle entry count")
    entries, cursor = [], 32
    for _ in range(count):
        offset, size, label_size = struct.unpack("<QQQ", chunk(bundle, cursor, 24))
        cursor += 24
        need(0 < label_size < 4096, "invalid target label size")
        label = chunk(bundle, cursor, label_size).decode("ascii")
        cursor += label_size
        payload = chunk(bundle, offset, size)
        entries.append({"target": label, "offset": offset, "bytes": size,
                        "sha256": hashlib.sha256(payload).hexdigest()})
    need(len({e["target"] for e in entries}) == count, "duplicate bundle target")
    for entry in entries:
        need(entry["offset"] >= cursor, "bundle payload overlaps header")
    candidates = [e for e in entries if e["target"].startswith(("hip-", "hipv4-"))
                  and re.search(r"(?:^|-)gfx906(?:$|:)", e["target"])]
    need(len(candidates) == 1, "expected exactly one gfx906 target")
    chosen = candidates[0]
    image = chunk(bundle, chosen["offset"], chosen["bytes"])
    need(image[:7] == b"\x7fELF\x02\x01\x01", "gfx906 payload is not ELF64LE")
    need(number(image, 18, "<H") == 224, "gfx906 payload is not EM_AMDGPU")
    return image, entries, chosen

def main():
    need(len(sys.argv) == 3, "usage: extract_stamp_elf.py INPUT_OBJECT NEW_OUTPUT_DIRECTORY")
    source, output = map(Path, sys.argv[1:])
    need(source.stat().st_size == EXPECTED_SIZE, "input size differs from pinned object")
    data = source.read_bytes()
    need(hashlib.sha256(data).hexdigest() == EXPECTED_SHA, "input SHA256 mismatch")
    section_offset, bundle = hip_section(data)
    image, entries, chosen = select_image(bundle)
    receipt = {"input_sha256": EXPECTED_SHA, "input_bytes": len(data),
               "section": ".hip_fatbin", "section_file_offset": section_offset,
               "section_bytes": len(bundle), "section_sha256": hashlib.sha256(bundle).hexdigest(),
               "entries": entries, "selected": chosen,
               "image_host_file_offset": section_offset + chosen["offset"],
               "device_image_sha256": hashlib.sha256(image).hexdigest(),
               "compiled": False, "gpu_executed": False, "gpu_runtime_loaded": False}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    with (output / "verify-kernels.gfx906.elf").open("xb") as f:
        f.write(image)
    with (output / "extraction.json").open("x") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, UnicodeError, struct.error) as error:
        sys.exit("STOP: " + str(error))
