"""
Convert the HICO-DET annotation JSONs from fredzzhang/hicodet into the
PPDM-style annotation files used by this repository.

Source files (no Google Drive required):
    https://raw.githubusercontent.com/fredzzhang/hicodet/main/instances_train2015.json
    https://raw.githubusercontent.com/fredzzhang/hicodet/main/instances_test2015.json

Each source file is a dict with parallel lists:
    filenames[i]  -> image file name, e.g. HICO_train2015_00000001.jpg
    annotation[i] -> {boxes_h, boxes_o, hoi, object, verb}
where boxes are xyxy pixel coordinates, `hoi` is the 0-based HICO class index,
`object` indexes the source `objects` list and `verb` is the 0-based verb index.

The output format expected by datasets/hico.py is:
    [{"file_name": ...,
      "annotations": [{"bbox": [x, y, w, h], "category_id": ...}, ...],
      "hoi_annotation": [{"subject_id": ..., "object_id": ...,
                          "category_id": <verb 1..117>,
                          "hoi_category_id": <hoi 1..600>}, ...]}]

Usage:
    python tools/convert_hico_annotations.py \
        --train_json ~/hicodet_meta/instances_train2015.json \
        --test_json  ~/hicodet_meta/instances_test2015.json \
        --output_dir data/hico_20160224_det/annotations
"""

import argparse
import json
import os
import re
import sys

# Make the repository root importable when this file is run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same tuple as datasets/hico.py::HICODetection._valid_obj_ids.
VALID_OBJ_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)


def build_object_mapping(source_objects):
    """Map the source object index (0..79) to the project's object index."""
    from datasets.hico_text_label import hico_obj_text_label

    project = [
        re.sub(r"^a photo of (a|an) ", "", label)
        for _, label in hico_obj_text_label[:80]
    ]
    source = [name.replace("_", " ") for name in source_objects]
    if len(source) != len(project):
        raise ValueError("object list length mismatch: %d vs %d" % (len(source), len(project)))

    mapping = {}
    for src_idx, name in enumerate(source):
        hits = [idx for idx, proj_name in enumerate(project) if proj_name == name]
        if len(hits) != 1:
            raise ValueError("cannot uniquely map object %r: %r" % (name, hits))
        mapping[src_idx] = hits[0]
    return mapping


def convert_split(source_path, obj_mapping):
    with open(source_path, "r") as f:
        data = json.load(f)

    filenames = data["filenames"]
    annotations = data["annotation"]
    if len(filenames) != len(annotations):
        raise ValueError("filenames/annotation length mismatch in %s" % source_path)

    person_cat = VALID_OBJ_IDS[0]
    output = []

    for filename, ann in zip(filenames, annotations):
        boxes = []
        box_index = {}

        def add_box(box_xyxy, category_id):
            x1, y1, x2, y2 = [float(v) for v in box_xyxy]
            if x2 <= x1 or y2 <= y1:
                return None
            key = (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3), int(category_id))
            if key not in box_index:
                box_index[key] = len(boxes)
                boxes.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": int(category_id),
                })
            return box_index[key]

        hois = []
        boxes_h = ann.get("boxes_h", [])
        boxes_o = ann.get("boxes_o", [])
        hoi_ids = ann.get("hoi", [])
        objects = ann.get("object", [])
        verbs = ann.get("verb", [])
        n = len(boxes_h)
        if not (len(boxes_o) == len(hoi_ids) == len(objects) == len(verbs) == n):
            raise ValueError("inconsistent annotation lengths for %s" % filename)

        for box_h, box_o, hoi_id, object_idx, verb_idx in zip(
            boxes_h, boxes_o, hoi_ids, objects, verbs
        ):
            subject_id = add_box(box_h, person_cat)
            object_id = add_box(box_o, VALID_OBJ_IDS[obj_mapping[int(object_idx)]])
            if subject_id is None or object_id is None:
                continue
            hois.append({
                "subject_id": subject_id,
                "object_id": object_id,
                "category_id": int(verb_idx) + 1,      # 1..117
                "hoi_category_id": int(hoi_id) + 1,     # 1..600
            })

        output.append({
            "file_name": filename,
            "annotations": boxes,
            "hoi_annotation": hois,
        })

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument(
        "--output_dir", default="data/hico_20160224_det/annotations"
    )
    args = parser.parse_args()

    with open(args.train_json, "r") as f:
        train_source = json.load(f)
    obj_mapping = build_object_mapping(train_source["objects"])

    os.makedirs(args.output_dir, exist_ok=True)
    for source, name in [
        (args.train_json, "trainval_hico.json"),
        (args.test_json, "test_hico.json"),
    ]:
        converted = convert_split(source, obj_mapping)
        out_path = os.path.join(args.output_dir, name)
        with open(out_path, "w") as f:
            json.dump(converted, f)
        n_boxes = sum(len(item["annotations"]) for item in converted)
        n_hois = sum(len(item["hoi_annotation"]) for item in converted)
        print("%s: %d images, %d boxes, %d HOI instances -> %s" % (
            name, len(converted), n_boxes, n_hois, out_path))


if __name__ == "__main__":
    main()
