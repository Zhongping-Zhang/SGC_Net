import json
from dataloader.data.clevr import ClevrSceneGraphWithPairsDataset, clevr_collate_fn_withpairs
import torch

def build_clevr_train_dataset(args):
    print("building fully supervised %s train dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        clevr_vocab = json.load(f)
    dset_kwargs = {
        'vocab': clevr_vocab,
        'h5_path': args.train_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_samples': args.num_train_samples,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
        }
    clevr_train_dataset = ClevrSceneGraphWithPairsDataset(**dset_kwargs)
    return clevr_vocab, clevr_train_dataset


def build_clevr_val_dataset(args):
    print("building fully supervised %s validation dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        clevr_vocab = json.load(f)
    dset_kwargs = {
        'vocab': clevr_vocab,
        'h5_path': args.val_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
        }
    clevr_val_dataset = ClevrSceneGraphWithPairsDataset(**dset_kwargs)
    return clevr_vocab, clevr_val_dataset

def build_clevr_test_dataset(args):
    print("building fully supervised %s test dataset" % args.dataset)
    with open(args.vocab_json, 'r') as f:
        clevr_vocab = json.load(f)
    dset_kwargs = {
        'vocab': clevr_vocab,
        'h5_path': args.test_h5,
        'image_dir': args.image_dir,
        'image_size': args.image_size,
        'normalize_images': args.normalize_images,
        'max_objects': args.max_objects_per_image,
        'use_orphaned_objects': args.use_orphaned_objects,
        'include_relationships': args.include_relationships,
        }
    clevr_test_dataset = ClevrSceneGraphWithPairsDataset(**dset_kwargs)
    return clevr_vocab, clevr_test_dataset

def build_clevr_loader(args, split="test", shuffle=False):
    if "train" in split:
        clevr_vocab, clevr_dataset = build_clevr_train_dataset(args)
    elif "val" in split:
        clevr_vocab, clevr_dataset = build_clevr_val_dataset(args)
    elif "test" in split:
        clevr_vocab, clevr_dataset = build_clevr_test_dataset(args)
    else:
        assert False, "indicate a valid split"

    loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': shuffle,
    'collate_fn': clevr_collate_fn_withpairs,
    }
    clevr_loader = torch.utils.data.DataLoader(clevr_dataset, **loader_kwargs)

    return clevr_vocab, clevr_loader

