import os
import argparse
import yaml
from omegaconf import OmegaConf

def int_tuple(s):
    return tuple(int(i) for i in s.split(','))

def read_yaml_file(filename):
    with open(filename, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    return yaml_cfg


class Config(object):
    def __init__(self, config_path=None, DATA_DIR="data_dir", H5_DIR="h5_dir"):
        if config_path:
            self.config_file = OmegaConf.load(config_path)
            self.DATA_DIR = os.path.dirname(self.config_file.image_dir)
            self.H5_DIR = os.path.dirname(self.config_file.train_h5)
        else:
            self.parser = argparse.ArgumentParser()
            self.DATA_DIR=DATA_DIR
            self.H5_DIR=H5_DIR

    def parse(self, save=True, save_path="configs/default_args.yaml"):
        DATA_DIR = self.DATA_DIR
        H5_DIR = self.H5_DIR

        # Dataset options
        self.parser.add_argument('--dataset', default='vg', choices=['vg', 'clevr'])
        self.parser.add_argument('--vocab_json', default=os.path.join(DATA_DIR, 'vocab.json'))
        self.parser.add_argument('--train_h5', default=os.path.join(H5_DIR, 'train.h5'))
        self.parser.add_argument('--val_h5', default=os.path.join(H5_DIR, 'val.h5'))
        self.parser.add_argument('--test_h5', default=os.path.join(H5_DIR, 'test.h5'))
        self.parser.add_argument('--image_dir', default=os.path.join(DATA_DIR, 'images'))
        # self.parser.add_argument('--image_size', default='256,256', type=int_tuple)
        self.parser.add_argument('--image_size', type=int, nargs="*", default=[512,512] )
        self.parser.add_argument('--normalize_images', default=False, type=bool)
        self.parser.add_argument('--max_objects_per_image', default=10, type=int)
        self.parser.add_argument('--num_train_samples', default=None, type=int)
        self.parser.add_argument('--use_orphaned_objects', default=True, type=bool)
        self.parser.add_argument('--include_relationships', default=True, type=bool)
        # DataLoader options
        self.parser.add_argument('--batch_size', default=32, type=int)
        self.parser.add_argument('--loader_num_workers', default=4, type=int)

        self.args = self.parser.parse_args()

        if save:
            self.save(self.args, save_path=save_path)

    def save(self, args, save_path="configs/default_args.yaml"):
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path, 'w') as yamlfile:
            yaml.dump(vars(args), yamlfile)
        print("save arguments to %s successfully"%save_path)










