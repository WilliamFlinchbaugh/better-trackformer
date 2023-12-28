# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou


VIS_THRESHOLD = 0.25


def generate_coco_from_mot(split_name='train', seqs_names=None,
                           root_split='seqs', mots=False, mots_vis=False,
                           frame_range=None, data_root='data/MOT17', seq_path='seqs'):
    """
    Generates COCO data from MOT.
    """
    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    root_split_path = os.path.join(data_root, root_split)
    coco_dir = os.path.join(data_root, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "vehicle",
                                  "name": "car",
                                  "id": 1},
                                  {"supercategory": "vehicle",
                                  "name": "van",
                                  "id": 2}]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(data_root, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0

    seqs = sorted(os.listdir(root_split_path))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)

    for seq in seqs:
        # CONFIG FILE
        config = configparser.ConfigParser()
        config_file = os.path.join(root_split_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])
        else:
            raise NotImplementedError

        seg_list_dir = os.listdir(os.path.join(root_split_path, seq, 'frames'))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1

            os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'frames', img),
                       os.path.join(coco_dir, f"{seq}_{img}"))

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}
    for seq in seqs:
        # GT FILE
        gt_file_path = os.path.join(root_split_path, seq, 'gt', 'gt.txt')
        if not os.path.isfile(gt_file_path):
            print('gt file not found')
            return

        seq_annotations = []
        seq_annotations_per_frame = {}
        with open(gt_file_path, "r") as gt_file:
            reader = csv.reader(gt_file, delimiter=',')

            for row in reader:
                bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                bbox = [int(c) for c in bbox]

                area = bbox[2] * bbox[3]
                frame_id = int(row[0])
                image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.png", None)
                if image_id is None:
                    continue
                track_id = int(row[1])

                annotation = {
                    "id": annotation_id,
                    "bbox": bbox,
                    "image_id": image_id,
                    "segmentation": [],
                    "ignore": 0,
                    "visibility": 1,
                    "area": area,
                    "iscrowd": 0,
                    "seq": seq,
                    "category_id": 1 if int(row[-2]) == 2 else 2,
                    "track_id": track_id}

                seq_annotations.append(annotation)
                if frame_id not in seq_annotations_per_frame:
                    seq_annotations_per_frame[frame_id] = []
                seq_annotations_per_frame[frame_id].append(annotation)

                annotation_id += 1

        annotations['annotations'].extend(seq_annotations)
        print(len(seq_annotations), 'annotations in', seq)

            
    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


def check_coco_from_mot(coco_dir='data/MOT17/mot17_train_coco', annotation_file='data/MOT17/annotations/mot17_train_coco.json', img_id=None):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    # coco_dir = os.path.join(data_root, split)
    # annotation_file = os.path.join(coco_dir, 'annotations.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['van', 'car'])
    if img_id == None:
        img_ids = coco.getImgIds(catIds=cat_ids)
        index = np.random.randint(0, len(img_ids))
        img_id = img_ids[index]
    img = coco.loadImgs(img_id)[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig('annotations.png')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('set_folder')
    argParser.add_argument('split_name')
    
    args = argParser.parse_args()
    
    data_root = os.path.join('data', args.set_folder)
    
    with open(os.path.join(data_root, f'{args.split_name}.json'), 'r') as f:
        seq_split = json.load(f)
        train_seqs = seq_split['train']
        test_seqs = seq_split['test']
    
    frame_range = {'start':0.05, 'end':1.0}

    # TRAIN SET
    generate_coco_from_mot(
        f'{args.split_name}_train_coco',
        frame_range=frame_range,
        seqs_names=train_seqs,
        data_root=data_root,
        root_split='seqs')
    # VAL SET
    generate_coco_from_mot(
        f'{args.split_name}_val_coco',
        frame_range=frame_range,
        seqs_names=test_seqs,
        data_root=data_root,
        root_split='seqs')
    
    # ALL SET
    generate_coco_from_mot(
        f'{args.split_name}_all_coco',
        frame_range=frame_range,
        seqs_names=test_seqs.extend(train_seqs),
        data_root=data_root,
        root_split='seqs')
    
