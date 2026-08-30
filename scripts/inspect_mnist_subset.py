"""
Look at data/mnist/subset.bin.gz — the actual bytes, not a description of them.

    python scripts/inspect_mnist_subset.py           # header + first training image
    python scripts/inspect_mnist_subset.py 7         # training image 7
    python scripts/inspect_mnist_subset.py 7 test    # test image 7

Writing a reader for a binary file you have never seen is guesswork. This
prints the header with every byte accounted for, then renders one image as
characters beside the raw bytes that produced it, so the format stops being
something you take on trust.

THE WHOLE FILE, after gunzip — 1,962,512 bytes for the vendored subset:

    ┌──────────────────────────────────────────────────────────────┐
    │ 0    "MNSB"           4 bytes   magic, so a wrong file fails  │
    │ 4    nTrain           4 bytes   uint32 little-endian → 2000   │
    │ 8    nTest            4 bytes   uint32 little-endian →  500   │
    ├──────────────────────────────────────────────────────────────┤
    │ 12   train images     nTrain × 784 bytes = 1,568,000          │
    │      image 0 is bytes 12..795, image 1 is 796..1579, ...      │
    ├──────────────────────────────────────────────────────────────┤
    │      train labels     nTrain × 1 byte = 2,000                 │
    │      one byte per image, value 0-9, SAME ORDER as the images  │
    ├──────────────────────────────────────────────────────────────┤
    │      test images      nTest × 784 = 392,000                   │
    ├──────────────────────────────────────────────────────────────┤
    │      test labels      nTest × 1 = 500                         │
    └──────────────────────────────────────────────────────────────┘

Two things that trip people up:

  * Images and labels are in SEPARATE BLOCKS, not interleaved. Image 5's
    label is not next to image 5 — it is 1.5 MB further along, at the fifth
    byte of the label block. They line up by position, nothing else.

  * Each image is 784 bytes with no separator, no length prefix and no
    marker. The only reason you know where image 1 ends is that you counted
    784 from where it started. Lose count once and everything after is
    garbage that still looks like numbers.
"""
import gzip
import struct
import sys
from pathlib import Path

PIXELS = 784
RAMP = " .:-=+*#%@"


def hex_row(blob: bytes, offset: int, width: int = 16) -> str:
    """One line of a hex dump: offset, bytes as hex, then as printable text."""
    chunk = blob[offset:offset + width]
    hexed = " ".join(f"{b:02x}" for b in chunk)
    text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    return f"{offset:9,}  {hexed:<47}  |{text}|"


def main() -> None:
    path = Path(__file__).resolve().parent.parent / "data" / "mnist" / "subset.bin.gz"
    compressed = path.read_bytes()
    blob = gzip.decompress(compressed)

    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    print(f"file       {path.name}")
    print(f"on disk    {len(compressed):,} bytes, gzipped")
    print(f"unpacked   {len(blob):,} bytes")
    print()

    print("HEADER - the first 32 bytes as they physically sit on disk")
    print(f"{'offset':>9}  {'hex':<47}  ascii")
    for offset in (0, 16):
        print(hex_row(blob, offset))
    print()

    magic = blob[0:4].decode("ascii", "replace")
    n_train, n_test = struct.unpack("<II", blob[4:12])
    print(f"  bytes 0-3    {' '.join(f'{b:02x}' for b in blob[0:4])}   magic {magic!r}")
    print(f"  bytes 4-7    {' '.join(f'{b:02x}' for b in blob[4:8])}   nTrain = {n_train}")
    print(f"  bytes 8-11   {' '.join(f'{b:02x}' for b in blob[8:12])}   nTest  = {n_test}")
    print()
    print("  little-endian means the least significant byte is written FIRST:")
    print(f"    {' '.join(f'{b:02x}' for b in blob[4:8])}  =  0x{n_train:08x}  =  {n_train}")
    print()

    # where each block begins
    train_img = 12
    train_lab = train_img + n_train * PIXELS
    test_img = train_lab + n_train
    test_lab = test_img + n_test * PIXELS
    end = test_lab + n_test
    print("BLOCK LAYOUT")
    for name, start, length in [
        ("train images", train_img, n_train * PIXELS),
        ("train labels", train_lab, n_train),
        ("test images", test_img, n_test * PIXELS),
        ("test labels", test_lab, n_test),
    ]:
        print(f"  {name:<13} offset {start:>10,}   length {length:>10,}")
    print(f"  {'END':<13} offset {end:>10,}   {'matches file size' if end == len(blob) else 'MISMATCH'}")
    print()

    # one image, rendered and dumped
    base = train_img if split == "train" else test_img
    label_base = train_lab if split == "train" else test_lab
    count = n_train if split == "train" else n_test
    if not 0 <= index < count:
        raise SystemExit(f"{split} index must be 0..{count - 1}")

    start = base + index * PIXELS
    image = blob[start:start + PIXELS]
    label = blob[label_base + index]

    print(f"{split.upper()} IMAGE {index} - bytes {start:,} to {start + PIXELS - 1:,}")
    print("  space = 0 = paper, @ = 255 = ink. one character per byte.")
    print()
    for r in range(28):
        line = "".join(RAMP[min(9, image[r * 28 + c] * 10 // 256)] for c in range(28))
        print(f"    |{line}|")
    print()

    # the busiest row, as raw numbers — the same data, unrendered
    busiest = max(range(28), key=lambda r: sum(image[r * 28:(r + 1) * 28]))
    print(f"  row {busiest} of that picture, as the bytes it actually is:")
    print("    " + " ".join(f"{image[busiest * 28 + c]:02x}" for c in range(28)))
    print()
    print(f"  its label lives in the OTHER block, at byte {label_base + index:,}: value {label}")
    print(f"  so the picture above is a {label}.")


if __name__ == "__main__":
    main()
