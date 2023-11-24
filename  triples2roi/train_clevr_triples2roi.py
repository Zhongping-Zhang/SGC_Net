import os
from os.path import join, basename, dirname
import json
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F

from triples2roi.triples_model import multi_triples_lstm
from triples2roi.triples_dataloader import clevr_triples2roi

# hyperparameters
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_MAPPING={"multi_triples_lstm": multi_triples_lstm,
                }

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=4e-3)
parser.add_argument('--num_epoch', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--save_epoch_step', type=int, default=1)
parser.add_argument('--triples_info_folder', type=str, default="datasets/CLEVR_SIMSG/triples_info")
parser.add_argument('--max_triples_num',type=int, default=5)
parser.add_argument('--model_name', type=str, default="multi_triples_lstm")
args = parser.parse_args()
print(args)


write_log = True

model_folder = args.model_name+"_clevr_tnum_%d"%args.max_triples_num
os.makedirs("triples2roi/logs/"+model_folder, exist_ok=True)

# dataset
data_train = clevr_triples2roi(triples_info_file=join(args.triples_info_folder, "triples_train_info.pkl"), max_triples_num=args.max_triples_num)
data_val = clevr_triples2roi(triples_info_file=join(args.triples_info_folder, "triples_val_info.pkl"), max_triples_num=args.max_triples_num)
data_test = clevr_triples2roi(triples_info_file=join(args.triples_info_folder, "triples_test_info.pkl"), max_triples_num=args.max_triples_num)
print("train set length: ", len(data_train))
print("validation set length: ", len(data_val))
print("test set length: ", len(data_test))

# dataloader
train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size = args.batch_size,
            num_workers = 0,
            shuffle = True,
            )
val_loader = torch.utils.data.DataLoader(
            data_val,
            batch_size = len(data_val),
            num_workers = 0,
            shuffle = False,
            )
test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size = len(data_test),
            num_workers = 0,
            shuffle = False,
            )

with open(join(dirname(args.triples_info_folder), "target/vocab.json"), "r") as f:
    vocab = json.load(f)


# optimizer + scheduler
model = MODEL_MAPPING[args.model_name](vocab).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.95)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

#train
def train(epoch, write_log=True):
    model.train()
    for data in tqdm(train_loader):
        input_objects = data["input_objects"].type(LongTensor)
        input_boxes = data["input_boxes"].type(FloatTensor)
        predicates = data["predicates"].type(LongTensor)
        subject_indicators = data["subject_indicators"].type(LongTensor)
        triple_masks = data["triple_masks"].type(LongTensor)
        target_object = data["target_object"].type(LongTensor)
        target_box = data["target_box"].type(FloatTensor)

        optimizer.zero_grad()

        output = model(input_objects, input_boxes, predicates, subject_indicators, target_object)
        loss = F.mse_loss(output, target_box)

        loss.backward()
        optimizer.step()
        # scheduler.step()
        scheduler.step(loss)

    if write_log:
        f.write("epoch {}: loss {} \n".format(epoch, loss))


def test(epoch, write_log=True):
    with torch.no_grad():
        model.eval()  # model.eavl() fix the BN and Dropout
        test_mse_loss = 0
        test_mae_loss = 0

        print("test_loader READY =====> ")

        prediction = []
        img_ids = []
        for data in tqdm(test_loader):
            input_objects = data["input_objects"].type(LongTensor)
            input_boxes = data["input_boxes"].type(FloatTensor)
            predicates = data["predicates"].type(LongTensor)
            subject_indicators = data["subject_indicators"].type(LongTensor)
            triple_masks = data["triple_masks"].type(LongTensor)
            target_object = data["target_object"].type(LongTensor)
            target_box = data["target_box"].type(FloatTensor)

            output = model(input_objects, input_boxes, predicates, subject_indicators, target_object)

        print(target_box.shape, output.shape)
        test_mse_loss += F.mse_loss(output, target_box)
        test_mae_loss += F.l1_loss(output, target_box)

        if write_log:
            f.write("validation loss for epoch {}: mse {}; mae {} \n".format(epoch, test_mse_loss, test_mae_loss))
        print('test set: mse loss: {:.4f}; mae loss: {:.4f}\n'
              .format(test_mse_loss, test_mae_loss))


f = open("triples2roi/logs/"+model_folder+"/log.txt", "w")
for epoch in range(args.num_epoch):
    train(epoch, write_log)
    test(epoch, write_log)
    print('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
    if epoch % args.save_epoch_step == args.save_epoch_step - 1:
        torch.save(model, "triples2roi/logs/" + model_folder + "/epoch-%d.pkl" % (epoch + 1))
f.close()

print("train_finish")

