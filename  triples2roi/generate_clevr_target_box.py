import json
import torch
import argparse
from tqdm import tqdm
from os.path import join, basename, dirname
from triples2roi.triples_dataloader import clevr_triples2roi

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--triples_info_folder', type=str, default="data/CLEVR_SIMSG/triples_info")
parser.add_argument('--max_triples_num',type=int, default=5)
parser.add_argument('--model_name', type=str, default="triples2roi/logs/multi_triples_lstm_clevr_tnum_5/epoch-15.pkl")
args = parser.parse_args()
print(args)

data_test = clevr_triples2roi(triples_info_file=join(args.triples_info_folder, "triples_test_info.pkl"), max_triples_num=args.max_triples_num)
test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size = len(data_test),
            num_workers = 0,
            shuffle = False,
            )
print("test set length: ", len(data_test))

with open(join(dirname(args.triples_info_folder), "target/vocab.json"), "r") as f:
    vocab = json.load(f)
model = torch.load(args.model_name)

with torch.no_grad():
    model.eval()#model.eavl() fix the BN and Dropout
    for data in tqdm(test_loader):
        input_objects = data["input_objects"].type(LongTensor)
        input_boxes = data["input_boxes"].type(FloatTensor)
        predicates = data["predicates"].type(LongTensor)
        subject_indicators = data["subject_indicators"].type(LongTensor)
        triple_masks = data["triple_masks"].type(LongTensor)
        target_object = data["target_object"].type(LongTensor)
        target_box = data["target_box"].type(FloatTensor)
        img_ids = data["img_id"]
        output = model(input_objects, input_boxes, predicates, subject_indicators, target_object)

output_np = output.cpu().numpy()
clevr_id2box = {}
for index, img_id in enumerate(img_ids):
    clevr_id2box[img_id] = list(output_np[index])

import pickle
with open(join(args.triples_info_folder,'triples2box_prediction.pkl'), 'wb') as f:
    pickle.dump(clevr_id2box,f,protocol=pickle.HIGHEST_PROTOCOL)


