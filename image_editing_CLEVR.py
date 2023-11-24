import os
import numpy as np
import argparse
import json
import torch
from PIL import Image
from os.path import join, basename, dirname
from collections import Counter
from dataloader import build_clevr_loader
from utils import (Config, load_pickle, save_array_as_image, tripleToObjID_savemapping, refine_box,
                   resize_image_np, print_triple, box2coordinates, find_target_obj_index,)
from tqdm import tqdm
import pickle
from matplotlib.pyplot import imshow,show

parser = argparse.ArgumentParser()
parser.add_argument('--CONFIG_PATH', type=str, default="configs/clevr_batchsize1.yaml", help='config file')
parser.add_argument('--MODE', type=str, default="reposition", choices=["remove", "reposition", "replace", "addition"])
parser.add_argument('--TRIPLES2ROI_MODEL_NAME', type=str, default="data/CLEVR_SIMSG/triples_info/triples2box_prediction.pkl")
parser.add_argument('--SAVE_FOLDER', type=str, default="results/CLEVR_test_control_map")
parser.add_argument('--SPLIT', type=str, default="test", help="train/val/test")
args = parser.parse_args()
print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config(args.CONFIG_PATH)
clevr_vocab, clevr_dataloader = build_clevr_loader(config.config_file,
                                                         split=args.SPLIT,
                                                         shuffle=False)
object_idx_to_name = clevr_vocab["object_idx_to_name"]
object_name_to_idx = clevr_vocab["object_name_to_idx"]
print("there are %d samples"%len(clevr_dataloader))


if args.TRIPLES2ROI_MODEL_NAME:
    with open(args.TRIPLES2ROI_MODEL_NAME, 'rb') as f:
        clevr_id2box = pickle.load(f)

for MODE in ["remove", "reposition", "replace", "addition"]:
    image_folder = join(args.SAVE_FOLDER, MODE, "control")
    image_org_folder = join(args.SAVE_FOLDER, MODE, "images_org")
    image_gt_folder = join(args.SAVE_FOLDER, MODE, "images_gt")
    mask_folder = join(args.SAVE_FOLDER, MODE, "masks")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(image_org_folder, exist_ok=True)
    os.makedirs(image_gt_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

sample_id = 0
sg4im_info = {}
bg_guided_info = {}
for data in tqdm(clevr_dataloader, desc="CLEVR dataloader"):
    (imgs_target, imgs_source, objs_target, objs_source, boxes_target, boxes_source,
     triples_target, triples_source, all_obj_to_img, all_triple_to_img, img_paths) = data
    img_id = os.path.basename(img_paths[0])
    sg4im_info[img_id] = {}
    image_source_np = np.transpose(imgs_source[0].cpu().numpy(), (1, 2, 0))
    image_target_np = np.transpose(imgs_target[0].cpu().numpy(), (1, 2, 0))
    H,W,C = image_source_np.shape

    # Get mode from target scene graph - source scene graph, or image id, using sets
    triples_objid_source, mapping_source = tripleToObjID_savemapping(triples_source, objs_source)
    triples_objid_target, mapping_target = tripleToObjID_savemapping(triples_target, objs_target)

    graph_set_source = Counter(tuple(row) for row in triples_objid_source)
    obj_set_source = Counter([int(obj) for obj in objs_source])
    graph_set_target = Counter(tuple(row) for row in triples_objid_target)
    obj_set_target = Counter([int(obj) for obj in objs_target])

    cropped_parts = []
    cropped_boxes = []


    if len(objs_target) > len(objs_source):
        mode='addition'
        graph_changes = graph_set_target - graph_set_source
        obj_changes = list(obj_set_target - obj_set_source) # category of new object
        target_obj_index = find_target_obj_index(graph_changes, mapping_target, obj_changes) # new object position in objs_target
        target_obj_category = objs_target[target_obj_index]
        target_obj_box = boxes_target[target_obj_index] # target box
        pred_obj_box_v0 = clevr_id2box[img_id]
        pred_obj_box = refine_box(pred_obj_box_v0, target_obj_box)

        inpainting_mask = np.ones_like(image_source_np[..., 0])
        left, right, top, bottom = box2coordinates(pred_obj_box, image_source_np.shape)
        inpainting_mask[top:bottom, left:right] = 0
        edited_image_source_np = image_source_np * inpainting_mask[..., None]

        img_save_name = img_id
        mask_save_name = img_save_name[:-4] + "_mask.jpg"
        save_array_as_image(edited_image_source_np, join(args.SAVE_FOLDER, mode, "control", img_save_name))
        save_array_as_image(image_source_np, join(args.SAVE_FOLDER, mode, "images_org", img_save_name))
        save_array_as_image(image_target_np, join(args.SAVE_FOLDER, mode, "images_gt", img_save_name))
        save_array_as_image(inpainting_mask, join(args.SAVE_FOLDER, mode, "masks", mask_save_name))
        sg4im_info[img_id]["target_object"] = object_idx_to_name[target_obj_category]

    elif len(objs_target) < len(objs_source):
        mode = "remove"
        graph_changes = graph_set_source - graph_set_target
        obj_changes = list(obj_set_source - obj_set_target)
        try:
            source_obj_index = find_target_obj_index(graph_changes, mapping_source, obj_changes) # new object position in objs_target
        except:
            continue
        source_obj_category = int(objs_source[source_obj_index])
        source_obj_box = boxes_source[source_obj_index]

        inpainting_mask = np.ones_like(image_source_np[..., 0])
        # use ground truth bounding box to generate intermediate image, for training
        left, right, top, bottom = box2coordinates(source_obj_box, image_source_np.shape)
        inpainting_mask[top:bottom, left:right] = 0
        edited_image_source_np = image_source_np*inpainting_mask[...,None]

        img_save_name = img_id
        mask_save_name = img_save_name[:-4] + "_mask.jpg"
        save_array_as_image(edited_image_source_np, join(args.SAVE_FOLDER, mode, "control", img_save_name))
        save_array_as_image(image_source_np, join(args.SAVE_FOLDER, mode, "images_org", img_save_name))
        save_array_as_image(image_target_np, join(args.SAVE_FOLDER, mode, "images_gt", img_save_name))
        save_array_as_image(inpainting_mask, join(args.SAVE_FOLDER, mode, "masks", mask_save_name))
        sg4im_info[img_id]["remove_object"] = object_idx_to_name[source_obj_category]


    elif torch.all(torch.eq(objs_target, objs_source)):
        mode = "reposition"
        graph_changes_target_object = (graph_set_source - graph_set_target) + (graph_set_target - graph_set_source)
        graph_target_changes = graph_set_target-graph_set_source
        graph_source_changes = graph_set_source-graph_set_target
        if not graph_target_changes: # sample_id: 129 & 2088, bad samples, since the scene graph has NOT been modified
            continue

        idx_cnt = np.zeros((25, 1)) # 25: object categories
        for [s, p, o] in list(graph_changes_target_object):
            idx_cnt[s] += 1
            idx_cnt[o] += 1
        obj_changes = idx_cnt.argmax(0) # most frequent object: target object

        source_obj_index = find_target_obj_index(graph_source_changes, mapping_source, obj_changes)
        source_obj_category = int(objs_source[source_obj_index])
        source_obj_box = boxes_source[source_obj_index]
        target_obj_index = find_target_obj_index(graph_target_changes, mapping_target, obj_changes)
        target_obj_category = int(objs_target[target_obj_index])
        target_obj_box = boxes_target[target_obj_index]
        pred_obj_box_v0 = clevr_id2box[img_id]
        pred_obj_box = refine_box(pred_obj_box_v0, target_obj_box)

        inpainting_mask1 = np.ones_like(image_source_np[...,0])
        inpainting_mask2 = np.ones_like(image_source_np[..., 0])

        left1, right1, top1, bottom1 = box2coordinates(source_obj_box, image_source_np.shape)
        inpainting_mask1[top1:bottom1, left1:right1] = 0

        left2, right2, top2, bottom2 = box2coordinates(pred_obj_box, image_source_np.shape)
        inpainting_mask2[top2:bottom2, left2:right2] = 0

        edited_image_source_np = image_source_np*inpainting_mask2[...,None]


        img_save_name = img_id
        mask_save_name1 = img_save_name[:-4] + "_source.jpg"
        mask_save_name2 = img_save_name[:-4] + "_target.jpg"

        save_array_as_image(edited_image_source_np, join(args.SAVE_FOLDER, mode, "control", img_save_name))
        save_array_as_image(image_source_np, join(args.SAVE_FOLDER, mode, "images_org", img_save_name))
        save_array_as_image(image_target_np, join(args.SAVE_FOLDER, mode, "images_gt", img_save_name))

        save_array_as_image(inpainting_mask1, join(args.SAVE_FOLDER, mode, "masks", mask_save_name1))
        save_array_as_image(inpainting_mask2, join(args.SAVE_FOLDER, mode, "masks", mask_save_name2))
        sg4im_info[img_id]["remove_object"] = object_idx_to_name[source_obj_category]
        sg4im_info[img_id]["target_object"] = object_idx_to_name[target_obj_category]

    elif len(objs_target) == len(objs_source):
        offset = 0
        mode = "replace"
        graph_source_changes = graph_set_source - graph_set_target
        obj_source_changes = list(obj_set_source - obj_set_target)
        graph_target_changes = graph_set_target - graph_set_source
        obj_target_changes = list(obj_set_target - obj_set_source)

        try:
            source_obj_index = find_target_obj_index(graph_source_changes, mapping_source, obj_source_changes)
        except:
            continue

        source_obj_category = int(objs_source[source_obj_index])
        source_obj_box = boxes_source[source_obj_index]
        target_obj_index = find_target_obj_index(graph_target_changes, mapping_target, obj_target_changes)
        target_obj_category = int(objs_target[target_obj_index])
        target_obj_box = boxes_target[target_obj_index]


        inpainting_mask = np.ones_like(image_source_np[..., 0])
        left,right,top,bottom = box2coordinates(source_obj_box, image_source_np.shape)
        left,right,top,bottom = max(0,left-offset), min(W,right+offset), max(0,top-offset), min(H,bottom+offset)
        inpainting_mask[top:bottom, left:right] = 0
        edited_image_source_np = image_source_np * inpainting_mask[..., None]

        img_save_name = img_id
        mask_save_name = img_save_name[:-4] + "_mask.jpg"
        save_array_as_image(edited_image_source_np, join(args.SAVE_FOLDER, mode, "control", img_save_name))
        save_array_as_image(image_source_np, join(args.SAVE_FOLDER, mode, "images_org", img_save_name))
        save_array_as_image(image_target_np, join(args.SAVE_FOLDER, mode, "images_gt", img_save_name))
        save_array_as_image(inpainting_mask, join(args.SAVE_FOLDER, mode, "masks", mask_save_name))
        sg4im_info[img_id]["remove_object"] = object_idx_to_name[source_obj_category]
        sg4im_info[img_id]["target_object"] = object_idx_to_name[target_obj_category]
    else:
        assert False, "operation mode is invalid"
    # print(mode)
    sg4im_info[img_id]["mode"] = mode
    sample_id+=1

with open(join(args.SAVE_FOLDER, "info_clevr_editing.json"), "w") as f:
    json.dump(sg4im_info, f)



