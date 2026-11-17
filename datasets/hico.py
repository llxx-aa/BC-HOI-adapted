"""
HICO detection dataset.
"""
import os
import random
from pathlib import Path

import torchvision.transforms
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
# import clip

import datasets.transforms as T
from .hico_text_label import hico_text_label, hico_unseen_index
# from .captions_template import template_one,template_two,template_three,template_four

from lavis.processors.blip_processors import BlipImageEvalProcessor




"""
HICO-DET 数据集的加载与标签构造。

每个样本返回两类信息：
1. 检测器输入：原图经过 DETR 的 transforms，得到 samples；
2. 目标标签 target：
   - boxes/labels：COCO 风格的物体框和类别；
   - sub_boxes/obj_boxes：每个 HOI 实例的人框、物框；
   - obj_labels：每个 HOI 实例的物体类别；
   - verb_labels：多热向量，表示这个实例涉及哪些动词；
   - hoi_labels：多热向量，表示这个实例属于哪些 HOI 类别；
   - blip_inputs：BLIP-2 的 224×224 输入；
   - llava_captions：GPT-4o/QwenVL 生成的描述，供 LSG 使用。
"""


class HICODetection(torch.utils.data.Dataset):
    """
    HICO-DET 数据集。

    注意：一个"HOI 实例"是 (人框, 物框) 这一对，
    同一对可以对应多个动词/HOI 类别，所以 verb_labels 和 hoi_labels 都是多热向量。
    """

    def __init__(self, img_set, img_folder, anno_file, clip_feats_folder, transforms, num_queries, args):
        self.img_set = img_set
        self.img_folder = img_folder
        self.clip_feates_folder = clip_feats_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self.unseen_index = hico_unseen_index.get(args.zero_shot_type, [])
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))

        self.text_label_dict = hico_text_label
        self.text_label_ids = list(self.text_label_dict.keys())
        if img_set == 'train' and len(self.unseen_index) != 0 and args.del_unseen:
            tmp = []
            for idx, k in enumerate(self.text_label_ids):
                if idx in self.unseen_index:
                    continue
                else:
                    tmp.append(k)
            self.text_label_ids = tmp

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                new_img_anno = []
                skip_pair = []
                for hoi in img_anno['hoi_annotation']:
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        skip_pair.append((hoi['subject_id'], hoi['object_id']))
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        new_img_anno = []
                        break
                    if (hoi['subject_id'], hoi['object_id']) not in skip_pair:
                        new_img_anno.append(hoi)
                if len(new_img_anno) > 0:
                    self.ids.append(idx)
                    img_anno['hoi_annotation'] = new_img_anno
        else:
            self.ids = list(range(len(self.annotations)))
        print("{} contains {} images".format(img_set, len(self.ids)))

        # Only the 224x224 eval transform is needed here. Previously this
        # loaded the whole BLIP-2 feature extractor (network + ~1 GB of
        # weights) merely to obtain the processor. Instantiate it directly.
        self.vis_processors = BlipImageEvalProcessor(image_size=224)

        hoi_text_label = hico_text_label
        unseen_index = hico_unseen_index
        if args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        self.hoi_string = []
        for key ,value in hoi_text_label_del.items():
            if ' and ' in value:
                value = value.replace(' and '," irrelated to ")
            self.hoi_string.append(value)

        # Only the GPT-4o captions are consumed by the current training path
        # (they become target['llava_captions'] for the LSG loss). The other
        # caption files belong to earlier ablations and were previously loaded
        # unconditionally, which made the dataset unusable unless every one of
        # them existed. Keep only the file we actually use, and make it optional.
        self.gpt4o_cap = {}
        gpt4o_caption_path = "data/captions_gpt4o_new.json"
        if os.path.exists(gpt4o_caption_path):
            with open(gpt4o_caption_path, 'r') as f:
                self.gpt4o_cap = json.load(f)
        else:
            print("[HICODetection] WARNING: {} not found; LSG captions will be empty.".format(
                gpt4o_caption_path))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        读取一张图，并构造检测器 + BLIP-2 + LSG 所需的全部标签。
        """
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img_0, target_0 = self._transforms[0](img, target)
                img, target = self._transforms[1](img_0, target_0)
            # BLIP-2 需要单独的 224×224 输入；它和检测器的输入分辨率不同。
            blip_inputs = self.vis_processors(img_0)
            target['blip_inputs'] = blip_inputs
            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            hoi_labels = []
            # 遍历 HICO 的 HOI 标注，把"人-物对"整理成实例级别。
            for hoi in img_anno['hoi_annotation']:
                # print('hoi: ', hoi)
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                verb_obj_pair = (self._valid_verb_ids.index(hoi['category_id']),
                                 target['labels'][kept_box_indices.index(hoi['object_id'])])
                if verb_obj_pair not in self.text_label_ids:
                    continue

                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    # 同一对 (人, 物) 已经出现过：只追加新的动词/HOI 标签，
                    # 不重复添加框。这就是"一人多动作"的多标签处理。
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_labels[sub_obj_pairs.index(sub_obj_pair)][self.text_label_ids.index(verb_obj_pair)] = 1
                else:
                    # 新的 (人, 物) 对：新建一个实例，记录人框、物框、物体类别。
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    hoi_label = [0] * len(self.text_label_ids)
                    hoi_label[self.text_label_ids.index(verb_obj_pair)] = 1
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    hoi_labels.append(hoi_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
            # caption_final_list = []
            # use_bounding_box = False
            # for i in range(len(hoi_labels)):
            #     hoi_label = hoi_labels[i]
            #     sub_box = sub_boxes[i].tolist()
            #     sub_box = [float('{:.4f}'.format(i)) for i in sub_box]
            #     obj_box = obj_boxes[i].tolist()
            #     obj_box = [float('{:.4f}'.format(i)) for i in obj_box]
            #     caption_list = []
            #     for j in range(len(hoi_label)):
            #         if hoi_label[j] == 1:
            #             caption_list.append(self.hoi_string[j][11:])
            #
            #     if len(caption_list) > 1:
            #         caption = ""
            #         for k in range(len(caption_list)):
            #             if k != len(caption_list) - 1:
            #                 caption += (caption_list[k][9:] + ' or ')
            #             else:
            #                 caption += caption_list[k][9:]
            #     else:
            #         caption = caption_list[0][9:]
            #     if use_bounding_box:
            #         caption = caption + ' {subject:'+str(sub_box) + ',object:'+str(obj_box) + '} '
            #     if 'irrelated to' in caption:
            #         continue
            #     else:
            #         caption_final_list.append('('+caption+')')
            # random.shuffle(caption_final_list)
            # if len(caption_final_list) > 4 :
            #     caption_final_list = caption_final_list[:4]
            # num = int(random.random() * 10)
            # if len(caption_final_list) == 0 :
            #     caption = self.Qwen_CoT_cap[img_anno['file_name']]['q1']
            # elif len(caption_final_list) == 1:
            #     caption = template_one[num]
            # elif len(caption_final_list) == 2:
            #     caption = template_two[num]
            # elif len(caption_final_list) == 3:
            #     caption = template_three[num]
            # elif len(caption_final_list) == 4:
            #     caption = template_four[num]
            # for caption_final in caption_final_list:
            #     caption = caption.replace("<doing something>", caption_final, 1)
            # caption_list = []
            # for i in range(len(hoi_labels)):
            #     hoi_label = hoi_labels[i]
            #     for j in range(len(hoi_label)):
            #         if hoi_label[j] == 1:
            #             caption_list.append(self.hoi_string[j])
            # target['llava_captions'] = 'Caption of interaction: ' + caption
            rand_num = random.random()
            # llava_captions = self.Qwen_CoT_cap[img_anno['file_name']]['q1']
            llava_captions = self.gpt4o_cap.get(img_anno['file_name'], {}).get('caption', '')
            # print(len(llava_captions))
            # llava_captions_string_list = llava_captions.split(" ")
            # randn_mask = torch.rand([len(llava_captions_string_list)])
            # randn_mask = torch.where(randn_mask > 0.25 , 1 , 0)
            # llava_captions_mask = ''
            # for k in range(len(llava_captions_string_list)):
            #     if k != len(llava_captions_string_list) -1:
            #         if randn_mask[k] == 0 :
            #             llava_captions_mask = llava_captions_mask + '_'+' '
            #         else:
            #             llava_captions_mask = llava_captions_mask + llava_captions_string_list[k]+' '
            #     else:
            #         if randn_mask[k] == 0 :
            #             llava_captions_mask = llava_captions_mask + '_'
            #         else:
            #             llava_captions_mask = llava_captions_mask + llava_captions_string_list[k]
            # print(llava_captions,llava_captions_string_list,llava_captions_mask)
            target['llava_captions'] = llava_captions

            # NOTE: the old code stored unused string lists
            # (target['sentence_Qwcap_CoT_q1']/..._mask) here. engine.py moves
            # every non-filename/llava_captions target to the GPU with
            # `.to(device)`, which crashes on lists of strings. These fields
            # are not used by the model, so they are no longer created.

            # print(caption)
            target['filename'] = img_anno['file_name']
            #
            # num = random.random()
            # if num > 0.5:
            # Qwcap_CoT_q1 = self.Qwen_CoT_cap[img_anno['file_name']]['q1']
            # target['captions'] = Qwcap_CoT_q1
            # if num > 0.7:
            #     Qwcap_CoT_q5 = self.Qwen_CoT_cap[img_anno['file_name']]['q5']
            #     target['captions'] = Qwcap_CoT_q5
            # if num > 0.8:
            #     Qwcap_CoT_q6 = self.Qwen_CoT_cap[img_anno['file_name']]['q6']
            #     target['captions'] = Qwcap_CoT_q6
            # if num > 0.9:
            #     Qwcap_pose = self.Qwen_pose_cap[img_anno['file_name']]['pose']
            #     target['captions'] = Qwcap_pose
            # print(target['captions'])
            # if len(caption_list) != 0:
            #     num = int(random.random() * len(caption_list))
            #     # print(num, len(caption_list))
            #     target['captions'] = caption_list[num]
            # else:
            #     target['captions'] = ""
            # print('sub_obj_pairs: ', sub_obj_pairs)
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_labels'] = torch.zeros((0, len(self.text_label_ids)), dtype=torch.float32)
                target['image_hoi_labels'] = torch.zeros((len(self.text_label_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['hoi_labels'] = torch.as_tensor(hoi_labels, dtype=torch.float32)
                target['image_hoi_labels'] = target['hoi_labels'].clone().sum(0).to(torch.bool).to(torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
            # image_hoi_caption_list = []
            # image_hoi_labels = target['image_hoi_labels'].tolist()
            # for k in range(len(image_hoi_labels)):
            #     if image_hoi_labels[k] == 1:
            #         image_hoi_caption_list.append(self.hoi_string[k][20:])
            # random.shuffle(image_hoi_caption_list)
            # image_captions = 'The interactions present in the picture include : '
            # if len(image_hoi_caption_list) == 0:
            #     image_captions = 'There is no valid interactions in the picture.'
            # for p in range(len(image_hoi_caption_list)) :
            #     image_hoi_caption = image_hoi_caption_list[p]
            #     if p != len(image_hoi_caption_list)-1:
            #         image_captions = image_captions + image_hoi_caption + ','
            #     else:
            #         image_captions = image_captions + image_hoi_caption + '.'
            # target['captions'] = image_captions
            # num = random.random()
            # if num > 0.33:
            #     Qwcap_CoT_q1 = self.Qwen_CoT_cap[img_anno['file_name']]['q1']
            #     target['captions'] = Qwcap_CoT_q1
            # if num > 0.66:
            #     Qwcap_pose = self.Qwen_pose_cap[img_anno['file_name']]['pose']
            #     target['captions'] = Qwcap_pose
            # target['llava_captions'] = self.llava_train[target['filename']]
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            # target['llava_captions'] = self.llava_test[target['filename']]

            if self._transforms is not None:
                img_0, _ = self._transforms[0](img, None)
                img, _ = self._transforms[1](img_0, None)
            blip_inputs = self.vis_processors(img_0)
            target['blip_inputs'] = blip_inputs

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)


        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        if len(self.unseen_index) == 0:
            # no unseen categoruy, use rare to evaluate
            counts = defaultdict(lambda: 0)
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    counts[triplet] += 1
            self.rare_triplets = []
            self.non_rare_triplets = []
            for triplet, count in counts.items():
                if count < 10:
                    self.rare_triplets.append(triplet)
                else:
                    self.non_rare_triplets.append(triplet)
            print("rare:{}, non-rare:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))
        else:
            self.rare_triplets = []
            self.non_rare_triplets = []
            for img_anno in annotations:
                hois = img_anno['hoi_annotation']
                bboxes = img_anno['annotations']
                for hoi in hois:
                    triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                            self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                            self._valid_verb_ids.index(hoi['category_id']))
                    if hoi['hoi_category_id'] - 1 in self.unseen_index:
                        self.rare_triplets.append(triplet)
                    else:
                        self.non_rare_triplets.append(triplet)
            print("unseen:{}, seen:{}".format(len(self.rare_triplets), len(self.non_rare_triplets)))

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ]))]
            ),
            normalize
            ]

    if image_set == 'val':
        return [T.Compose([
            T.RandomResize([800], max_size=1333),

        ]),
        normalize
        ]

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json',
                  root / 'clip_feats_pool' / 'train2015'),
        'val': (
            root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json',
            root / 'clip_feats_pool' / 'test2015')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file, clip_feats_folder = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, clip_feats_folder,
                            transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, args=args)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
