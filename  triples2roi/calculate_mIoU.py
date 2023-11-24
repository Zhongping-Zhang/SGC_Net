import pickle
from os.path import join, basename, dirname
import argparse
import numpy as np
from tqdm import tqdm

def calculate_iou(bbox_a, bbox_b): # bbox_a: (left, top, right, bottom)
    left_inter = max(bbox_a[0], bbox_b[0])
    top_inter = max(bbox_a[1], bbox_b[1])
    right_inter = min(bbox_a[2], bbox_b[2])
    bottom_inter = min(bbox_a[3], bbox_b[3])

    if left_inter >= right_inter or top_inter >= bottom_inter:
        intersection = 0
    else:
        intersection = (right_inter - left_inter) * (bottom_inter - top_inter)

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    iou = intersection / union
    return iou

def calculate_precision_recall_f1(iou_values, threshold):
    tp = sum(iou > threshold for iou in iou_values)
    fp = sum(iou <= threshold for iou in iou_values)
    fn = sum(iou_values) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triples_info_folder', type=str, default="data/CLEVR_SIMSG/triples_info")
    parser.add_argument('--max_triples_num',type=int, default=5)
    parser.add_argument('--model_name', type=str, default="multi_triples_lstm")
    parser.add_argument('--image_size',type=int, default=256)
    args = parser.parse_args()
    print(args)

    gt_triples_info = join(args.triples_info_folder, "triples_test_info.pkl")
    pred_triples_info = join(args.triples_info_folder, "triples2box_prediction.pkl")


    with open(gt_triples_info, 'rb') as f:
        triples_info = pickle.load(f)

    with open(pred_triples_info, 'rb') as f:
        pred_bbox = pickle.load(f)


    reposition_IoUs = []
    addition_IoUs = []
    for img_id in tqdm(pred_bbox.keys()):
        mode = triples_info[img_id]['mode']
        gt_box = triples_info[img_id]['target_box'] # (x_1, y_1, x_2, y_2), (left, top, right, bottom)
        pred_box = np.array(pred_bbox[img_id])

        iou = calculate_iou(gt_box, pred_box)
        if mode=='addition':
            addition_IoUs.append(iou)
        else:
            reposition_IoUs.append(iou)

    all_IoUs = reposition_IoUs+addition_IoUs

    print("mIoU", np.mean(all_IoUs))
    print("addition mIoU", np.mean(addition_IoUs))
    print("reposition mIoU", np.mean(reposition_IoUs))









