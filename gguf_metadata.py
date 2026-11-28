"""Bounded, best-effort GGUF layer-count discovery (no tensor data read)."""
import struct


def layer_count(path):
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            raw = fh.read(4)
            if raw in (b"\x02\x00\x00\x00", b"\x03\x00\x00\x00"):
                endian = "<"
            elif raw in (b"\x00\x00\x00\x03",):
                endian = ">"
            else:
                return None

            def number(fmt):
                return struct.unpack(endian + fmt, fh.read(struct.calcsize(fmt)))[0]

            def string(read=True, cap=4096):
                size = number("Q")
                if size > cap:
                    if read:
                        raise ValueError("metadata string too long")
                    fh.seek(size, 1)
                    return None
                if read:
                    raw = fh.read(size)
                    if len(raw) != size:
                        raise ValueError("truncated metadata")
                    return raw.decode("utf-8")
                fh.seek(size, 1)
                return None

            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                     7: 1, 10: 8, 11: 8, 12: 8}

            string_budget = 10000

            def skip(kind):
                nonlocal string_budget
                if kind == 8:
                    string(False)
                elif kind == 9:
                    element, count = number("I"), number("Q")
                    if element in sizes:
                        fh.seek(count * sizes[element], 1)
                    elif element == 8 and count <= string_budget:
                        string_budget -= count
                        for _ in range(count):
                            string(False)
                    else:
                        raise ValueError("unsupported metadata array")
                elif kind in sizes:
                    fh.seek(sizes[kind], 1)
                else:
                    raise ValueError("unsupported metadata type")

            number("Q")  # tensor count
            count = number("Q")
            architecture, blocks = None, {}
            for _ in range(min(count, 4096)):
                key, kind = string(), number("I")
                if key == "general.architecture" and kind == 8:
                    architecture = string()
                elif key.endswith(".block_count") and kind == 4:
                    blocks[key] = number("I")
                else:
                    skip(kind)
                if architecture and architecture + ".block_count" in blocks:
                    value = blocks[architecture + ".block_count"]
                    return value if 0 < value <= 1024 else None
    except (OSError, ValueError, OverflowError, struct.error):
        pass
    return None
