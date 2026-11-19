"""Count tokens in four equivalent read/write payloads from the Etude alias patch.

Usage: python measure_payload.py path/to/patch.safetensors
Requires tiktoken==0.13.0 and its o200k_base encoding.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import tiktoken


def load_slice(path):
    # The fixture pins both the sparse-data layout and the row-copy provenance.
    blob = path.read_bytes()
    expected = "77512244acb066bdc5c6409eca33f23d785ceef566953582206fd8d48b153b2b"
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("Expected the native Etude G5-ALIAS seed 3 patch")

    header_end = 8 + int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:header_end])
    data = blob[header_end:]

    def numbers(name, fmt):
        lo, hi = header[name]["data_offsets"]
        return (n for (n,) in struct.iter_unpack(fmt, data[lo:hi]))

    indices = numbers("__iwi_indices_0", "<q")
    bfloat16 = numbers("__iwi_values_0", "<H")
    entries = dict(zip(indices, bfloat16, strict=True))
    start = 122076 * 2560
    # A BF16 number occupies the upper half of the corresponding FP32 bits.
    return [struct.unpack("<f", struct.pack("<I", entries[start + i] << 16))[0]
            for i in range(256)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    values = load_slice(parser.parse_args().patch)
    ref = {"handle": "0123456789abcdef0123456789abcdef", "shape": [256], "scalars": 256}
    tensor = "model.language_model.embed_tokens.weight"
    destination = [122076, [0, 256]]
    payloads = {
        "inline read": {**ref, "values": values},
        "handle-only read": ref,
        "literal write": {"tensor": tensor, "operation": "replace", "selection": destination,
                          "value": {"values": values}},
        "copy write": {"tensor": tensor, "operation": "copy", "selection": destination,
                       "source": {"tensor": tensor, "selection": [26365, [0, 256]]}},
    }
    encoding = tiktoken.get_encoding("o200k_base")
    for name, payload in payloads.items():
        message = json.dumps(payload, separators=(",", ":"))
        print(f"{name}: {len(encoding.encode(message)):,} tokens")


if __name__ == "__main__":
    main()
