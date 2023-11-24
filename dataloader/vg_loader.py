import json
from dataloader.data.vg import VgSceneGraphDataset, vg_collate_fn, vg_uncollate_fn
import torch

def build_vg_train_dataset(args):
    print("building %s train dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        vg_vocab = json.load(f)
    dataset_kwargs = {
        'vocab': vg_vocab,
        'h5_path': args.train_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_objects': args.max_objects_per_image,
        'max_samples': args.num_train_samples,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    vg_train_dataset = VgSceneGraphDataset(**dataset_kwargs)
    return vg_vocab, vg_train_dataset

def build_vg_val_dataset(args):
    print("building %s validation dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        vg_vocab = json.load(f)
    dataset_kwargs = {
        'vocab': vg_vocab,
        'h5_path': args.val_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    vg_val_dataset = VgSceneGraphDataset(**dataset_kwargs)
    return vg_vocab, vg_val_dataset

def build_vg_test_dataset(args):
    print("building %s test dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        vg_vocab = json.load(f)
    dataset_kwargs = {
        'vocab': vg_vocab,
        'h5_path': args.test_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
    }
    vg_test_dataset = VgSceneGraphDataset(**dataset_kwargs)
    return vg_vocab, vg_test_dataset

def build_vg_loader(args, split="test", shuffle=False):
    if "train" in split:
        vg_vocab, vg_dataset = build_vg_train_dataset(args)
    elif "val" in split:
        vg_vocab, vg_dataset = build_vg_val_dataset(args)
    elif "test" in split:
        vg_vocab, vg_dataset = build_vg_test_dataset(args)
    else:
        assert False, "indicate a valid split"

    # shuffle=True if "train" in split else False

    loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': shuffle,
    'collate_fn': vg_collate_fn,
    }
    vg_loader = torch.utils.data.DataLoader(vg_dataset, **loader_kwargs)

    return vg_vocab, vg_loader





if __name__=="__main__":
    import argparse
    import os
    from tqdm import tqdm
    from matplotlib.pyplot import imshow,show

    def int_tuple(s):
        return tuple(int(i) for i in s.split(','))

    def argument_parser(DATA_DIR="datasets/vg", H5_DIR="/home/grad3/zpzhang/H5_Files/vg_h5"):
        # helps parsing the same arguments in a different script
        parser = argparse.ArgumentParser()

        # Dataset options
        parser.add_argument('--dataset', default='vg', choices=['vg', 'clevr'])
        parser.add_argument('--vocab_json', default=os.path.join(DATA_DIR, 'vocab.json'))
        parser.add_argument('--train_h5', default=os.path.join(H5_DIR, 'train.h5'))
        parser.add_argument('--val_h5', default=os.path.join(H5_DIR, 'val.h5'))
        parser.add_argument('--test_h5', default=os.path.join(H5_DIR, 'test.h5'))
        parser.add_argument('--image_dir', default=os.path.join(DATA_DIR, 'images'))
        parser.add_argument('--image_size', default='256,256', type=int_tuple)
        parser.add_argument('--normalize_images', default=False, type=bool)
        parser.add_argument('--max_objects_per_image', default=10, type=int)
        parser.add_argument('--num_train_samples', default=None, type=int)
        parser.add_argument('--num_val_samples', default=1024, type=int)
        parser.add_argument('--use_orphaned_objects', default=True, type=bool)
        parser.add_argument('--include_relationships', default=True, type=bool)
        # DataLoader options
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--loader_num_workers', default=4, type=int)

        args = parser.parse_args()
        # print(args)
        return args

    DATA_DIR="datasets/vg"
    H5_DIR="/home/grad3/zpzhang/H5_Files/vg_h5"
    args = argument_parser(DATA_DIR, H5_DIR)
    vg_vocab, vg_dataloader = build_vg_loader(args, split="train")
    vg_vocab, vg_val_dataloader = build_vg_loader(args, split="val")
    clevr_vocab, vg_test_dataloader = build_vg_loader(args, split="test")

    test_img_paths = []
    for data in tqdm(vg_test_dataloader):
        (all_imgs, all_objs, all_boxes, all_triples,
         all_obj_to_img, all_triple_to_img, test_img_path) = data
        """
        all_imgs: (img_num_per_batch,3,256,256)
        all_objs: (obj_num_per_batch,)
        all_boxes: (obj_num_per_batch,4)
        all_triples: (triples_per_batch,3)
        all_obj_to_img: (obj_num_per_batch,), map object to image
        all_triple_to_img: (triples_per_batch), map triple to image
        """
        test_img_paths+=test_img_path
        # break
    with open("datasets/vg/image_paths/test_img_paths.json", "w") as fp:
        json.dump(test_img_paths,fp)


    print("finished")
