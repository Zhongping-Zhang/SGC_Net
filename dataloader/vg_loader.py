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






