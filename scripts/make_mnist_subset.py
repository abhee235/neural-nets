"""
Build the vendored MNIST subset used by docs/deep-dives/mnist-dense-network.md.

The full MNIST set is ~54 MB and has no business in a course repository, but
a deep-dive that says "run this and see" needs the data to be there. So we
commit a stratified subset: an equal number of examples per digit, gzipped.

    python scripts/make_mnist_subset.py <dir-with-idx-gz-files>

Reads the four standard IDX files (train/test images and labels), takes
TRAIN_PER_CLASS and TEST_PER_CLASS examples of each digit in the order they
appear, and writes a single gzipped blob to data/mnist/subset.bin.gz.

FILE FORMAT (after gunzip) — deliberately trivial, so the TypeScript loader
that reads it is ten lines and teaches nothing but array offsets:

    offset 0    4 bytes   magic "MNSB"
    offset 4    4 bytes   nTrain, uint32 little-endian
    offset 8    4 bytes   nTest,  uint32 little-endian
    offset 12   nTrain*784 bytes   train images, one byte per pixel, row-major
    ...         nTrain     bytes   train labels, one byte each, 0-9
    ...         nTest*784  bytes   test images
    ...         nTest      bytes   test labels

Pixels stay as uint8 (0-255) rather than floats: it is four times smaller,
and the loader divides by 255 anyway.
"""
import gzip
import struct
import sys
from pathlib import Path

TRAIN_PER_CLASS = 200   # 2000 training images total
TEST_PER_CLASS = 50     # 500 test images total
IMAGE_BYTES = 28 * 28


def read_idx_images(path: Path) -> tuple[bytes, int]:
    """Return (pixel bytes, count) from an IDX3 image file."""
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: expected image magic 2051, got {magic}")
        if rows * cols != IMAGE_BYTES:
            raise ValueError(f"{path}: expected 28x28, got {rows}x{cols}")
        return handle.read(count * IMAGE_BYTES), count


def read_idx_labels(path: Path) -> bytes:
    """Return label bytes from an IDX1 label file."""
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: expected label magic 2049, got {magic}")
        return handle.read(count)


def take_stratified(images: bytes, labels: bytes, per_class: int) -> tuple[bytearray, bytearray]:
    """
    Take `per_class` examples of each digit, in the order they appear.

    Stratifying matters: MNIST is close to balanced but not exactly, and a
    subset drawn off the top would inherit that skew and quietly make the
    accuracy numbers harder to reason about.
    """
    wanted = {digit: per_class for digit in range(10)}
    out_images, out_labels = bytearray(), bytearray()
    for index, label in enumerate(labels):
        if wanted.get(label, 0) == 0:
            continue
        wanted[label] -= 1
        out_images += images[index * IMAGE_BYTES:(index + 1) * IMAGE_BYTES]
        out_labels.append(label)
        if not any(wanted.values()):
            break
    missing = {d: n for d, n in wanted.items() if n}
    if missing:
        raise ValueError(f"ran out of examples for digits {missing}")
    return out_images, out_labels


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    train_images, _ = read_idx_images(source / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(source / "train-labels-idx1-ubyte.gz")
    test_images, _ = read_idx_images(source / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(source / "t10k-labels-idx1-ubyte.gz")

    tr_x, tr_y = take_stratified(train_images, train_labels, TRAIN_PER_CLASS)
    te_x, te_y = take_stratified(test_images, test_labels, TEST_PER_CLASS)

    blob = bytearray(b"MNSB")
    blob += struct.pack("<II", len(tr_y), len(te_y))
    blob += tr_x + tr_y + te_x + te_y

    destination = Path(__file__).resolve().parent.parent / "data" / "mnist"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "subset.bin.gz"
    target.write_bytes(gzip.compress(bytes(blob), 9))

    print(f"train {len(tr_y)} images, test {len(te_y)} images")
    print(f"raw {len(blob):,} bytes -> gzipped {target.stat().st_size:,} bytes")
    print(f"written to {target}")


if __name__ == "__main__":
    main()
