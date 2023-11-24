import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class clevr_triples2roi(Dataset):
    def __init__(self,
                 triples_info_file: str = 'data/CLEVR_SIMSG/triples_info/triples_train_info.pkl',
                 max_triples_num: int = 7, # maximum triples
                 ):
        super(Dataset, self).__init__()
        self.triples_info_file = triples_info_file
        self.max_triples_num = max_triples_num

        with open(self.triples_info_file,'rb') as f:
            triples_info = pickle.load(f)

        self.valid_img_ids, self.valid_data = [],[]

        for sample_img_id, sample_data in triples_info.items():
            if "triples_info" in sample_data:
                self.valid_img_ids.append(sample_img_id)
                self.valid_data.append(sample_data)
        print("total samples: ", len(triples_info))
        print("valid samples: ", len(self.valid_data))


    def __getitem__(self, index: int):
        img_id = self.valid_img_ids[index]
        data = self.valid_data[index]

        target_object = data["target_object"]
        target_box = data["target_box"]

        input_boxes = np.zeros((self.max_triples_num,4))
        input_objects = np.zeros((self.max_triples_num))
        predicates = np.zeros((self.max_triples_num))
        subject_indicators = np.zeros((self.max_triples_num))
        triple_masks = np.zeros((self.max_triples_num))

        for i in range(min(self.max_triples_num, len(data["triples_info"]))):
            triple = data["triples_info"][i]
            input_object = triple["input_object"]
            input_box = triple["input_box"]
            predicate = triple["predicate"]
            subject_indicator = triple["subject_indicator"]

            input_objects[i] = input_object
            input_boxes[i,:] = input_box
            predicates[i] = predicate
            subject_indicators[i] = subject_indicator
            triple_masks[i] = 1

        return {
                "input_objects": input_objects,
                "input_boxes": input_boxes,
                "predicates": predicates,
                "subject_indicators": subject_indicators,
                "triple_masks": triple_masks,
                "target_object": target_object,
                "target_box": target_box,
                "img_id": img_id,
                }

    def __len__(self):
        return len(self.valid_data)


class vg_triples2roi(Dataset):
    def __init__(self,
                 triples_info_file: str = 'datasets/vg/triples_info/triples_train_info.pkl',
                 max_triples_num: int=1,
                 ):
        super(Dataset, self).__init__()
        self.triples_info_file = triples_info_file
        self.max_triples_num = max_triples_num
        with open(self.triples_info_file,'rb') as f:
            triples_info = pickle.load(f)

        self.valid_img_ids, self.valid_data = [], []
        for sample_img_id, sample_data in triples_info.items():
            self.valid_img_ids.append(sample_img_id)
            self.valid_data.append(sample_data)

    def __getitem__(self, index: int):
        img_id = self.valid_img_ids[index]
        data = self.valid_data[index]
        data["img_id"] = img_id
        return data

    def __len__(self):
        return len(self.valid_data)


