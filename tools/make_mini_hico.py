"""
Create a tiny HICO-DET subset for a portable smoke test.

The subset keeps the same layout as the full dataset:

    data/hico_mini/
    ├── images/train2015/...
    ├── images/test2015/...
    └── annotations/{trainval_hico.json,test_hico.json,corre_hico.npy}

Usage (from the repository root):
    python tools/make_mini_hico.py \
        --src data/hico_20160224_det \
        --dst data/hico_mini \
        --n-train 32 --n-test 8

It still requires the full HICO-DET images/annotations; it only reduces the
number of training/evaluation samples used by the smoke test.
"""

import argparse
import json
import shutil
from pathlib import Path


SPLITS = [
    ("train2015", "trainval_hico.json", "images/train2015", "n_train"),
    ("test2015", "test_hico.json", "images/test2015", "n_test"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/hico_20160224_det")
    parser.add_argument("--dst", default="data/hico_mini")
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--n-test", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(
            "source HICO-DET directory not found: %s" % src)
    (dst / "annotations").mkdir(parents=True, exist_ok=True)

    limits = {"n_train": args.n_train, "n_test": args.n_test}
    for split, anno_name, img_rel, limit_key in SPLITS:
        anno_path = src / "annotations" / anno_name
        if not anno_path.exists():
            raise FileNotFoundError("annotation file not found: %s" % anno_path)

        with open(anno_path, "r") as f:
            data = json.load(f)
        keep = [item for item in data if item.get("hoi_annotation")][:limits[limit_key]]
        if not keep:
            raise RuntimeError("no annotated images found in %s" % anno_path)

        out_img = dst / img_rel
        out_img.mkdir(parents=True, exist_ok=True)
        for item in keep:
            src_img = src / img_rel / item["file_name"]
            if not src_img.exists():
                raise FileNotFoundError("image not found: %s" % src_img)
            shutil.copy2(src_img, out_img / item["file_name"])

        out_anno = dst / "annotations" / anno_name
        with open(out_anno, "w") as f:
            json.dump(keep, f)
        print("%s: %d images -> %s" % (split, len(keep), out_img))

    corre_src = src / "annotations" / "corre_hico.npy"
    if not corre_src.exists():
        raise FileNotFoundError("correction matrix not found: %s" % corre_src)
    shutil.copy2(corre_src, dst / "annotations" / "corre_hico.npy")
    print("mini dataset ready:", dst)


if __name__ == "__main__":
    main()
