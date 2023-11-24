import numpy as np
from PIL import Image
import pickle
from collections import Counter




def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data




########################################################
############ process triple information ################
########################################################
def tripleToObjID_savemapping(triples, objs, savemapping=True):
    """
        map: map object position to object category
    """
    triples_new = []
    map={}
    for [s,p,o] in triples:
        triple = (int(objs[s]),int(p),int(objs[o]))
        map[triple] = [s,p,o]
        triples_new.append(triple)
    return triples_new, map

def find_target_obj_index(graph_changes, mapping_target, obj_changes):
    candidate_target_obj_index = []

    for s,p,o in graph_changes:
        if s==obj_changes[0]:
            candidate_target_obj_index.append(int(mapping_target[(s,p,o)][0]))
        elif o==obj_changes[0]:
            candidate_target_obj_index.append(int(mapping_target[(s,p,o)][2]))
        else:
            assert False, "target object is not in graph_changes (modified triples)"
    return Counter(candidate_target_obj_index).most_common(1)[0][0]
    # return candidate_target_obj_index


########################################################
########## helper function for image editing ###########
########################################################

ANIMAL_LIST = ["bird", "elephant", "sheep", "horse", "zebra", "cat", "dog", "bear",]
VEHICLES_LIST = ["car", "bus", "boat", "motorcycle", "plane", "train", "bike", "vehicle",]
PLANT_LIST = ["tree", "grass", "bush", "cloud", "plant", "banana", "chair",]
HUMAN_LIST = ["person","man","woman","boy","girl","child","people"]
OBJECT_SEGMENTATION_LIST = ANIMAL_LIST + VEHICLES_LIST + HUMAN_LIST

ALL_AUGMENTED_SEGMENTATION_LIST = \
    ["car", "bus", "boat", "motorcycle", "plane", "train", "bike", "vehicle",
     "bird", "elephant", "sheep", "horse", "zebra", "cat", "dog", "bear",
     "tree", "grass", "bush", "cloud", "plant", "banana", "chair",
     "ocean", "road", "sand", "snow", "street", "field",
     "person", "man", "woman", "boy", "girl", "child", "people", ]



# return left, right, top, bottom for the input bounding box
def box2coordinates(box, image_size):
    left = max(0, box[0] * image_size[1])  # box[0]
    right = min(image_size[1], box[2] * image_size[1])  # box[2]
    top = max(0, box[1] * image_size[0])  # box[1]
    bottom = min(image_size[0], box[3] * image_size[0])  # box[3]
    return int(left), int(right), int(top), int(bottom)

def refine_box(pred_obj_box, target_box):
    # resize predicted box to the same ratio of target box
    center_x, center_y = (pred_obj_box[0]+pred_obj_box[2])/2, (pred_obj_box[1]+pred_obj_box[3])/2
    width, height = (target_box[2]-target_box[0]), (target_box[3]-target_box[1])
    left, right, top, bottom = max(0,center_x-width/2), min(256,center_x+width/2), max(0,center_y-height/2), min(256,center_y+height/2)
    return np.array([left,top,right,bottom])


def save_array_as_image(image_np, save_name):
    if image_np.max()<1.1:
        image_np = (image_np * 255).astype(np.uint8)
    Image.fromarray(image_np).save(save_name)

def resize_image_np(image_np, target_size):
    if image_np.max()<1.1:
        image_np = (image_np*255).astype(np.uint8)
    image = Image.fromarray(image_np)
    resized_image = image.resize(size=target_size, resample=Image.Resampling.BILINEAR)
    resized_image_np = np.array(resized_image)
    return resized_image_np

def print_triple(triple, objs, vg_vocab, print_triple=True):
    object_idx_to_name = vg_vocab["object_idx_to_name"]
    text = (object_idx_to_name[objs[triple[0]]] + " " +
            vg_vocab["pred_idx_to_name"][triple[1]] + " "
            + object_idx_to_name[objs[triple[2]]])
    if print_triple:
        print(text)
    return text


def replace_obj_constrained(id, box, num_samples=7):
    """
    copied from SIMSG (https://github.com/he-dhamo/simsg/tree/master)
    automatically change object id, mapping given class to a class with similar size
    uses bbox to contrain the mapping, based on how close the object is
    returns list of changed ids for object replacement, of max size [num_samples]
    """
    objects = [6, 129, 116, 57, 127, 130] # 6:person, 57:bush, 116:elephant, 127:dog, 129:sheep, 130:zebra, 137:animal
    bg = [169, 60, 61, 141] # backgrounds, 60:street, 61:field, 141:sand, 169:ocean,
    vehicles = [100, 19, 70, 143] # 19:car, 70:boat, 100:bus, 143:motorcycle
    sky_obj = [21, 80] # 21: cloud, 80: bird
    if (id in objects or id==20 or id==3 or id==58) and (box[2]-box[0]<0.3):
        if id in objects:
          objects.remove(id)
        new_ids = np.random.choice(objects, num_samples)
        new_ids = list(dict.fromkeys(new_ids))
    elif id in bg:
        bg.remove(id)
        new_ids = np.random.choice(bg, num_samples)
        new_ids = list(dict.fromkeys(new_ids))
    elif id in vehicles and ((box[2] - box[0]) + (box[3] - box[1]) < 0.5):
        vehicles.remove(id)
        new_ids = np.random.choice(vehicles, num_samples)
        new_ids = list(dict.fromkeys(new_ids))
    elif id == 176 and ((box[2] - box[0]) + (box[3] - box[1]) < 0.1):
        new_ids = np.random.choice(sky_obj, num_samples)
        new_ids = list(dict.fromkeys(new_ids))
    else:
        new_ids = []
    return new_ids



