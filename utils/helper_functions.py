import json
from PIL import Image
from utils.Relation import Relation
import numpy as np
import numpy.ma as ma

def load_json(filename):
    with open(filename) as f:
        content = json.load(f)
    return content

def save_json_per_image(path, image_id, all_relations):
    file_name = path
    content = image_id - 1
    json_object = json.dumps(all_relations[content], indent=4)
    with open(file_name, "w") as outfile:
        outfile.write(json_object)

def show_cropped_image(path, location, show=True):
    im = Image.open(path)
    x, y, w, h = location
    left = x
    top = y
    right = x + w
    bottom = y + h
    cropped = im.crop((left, top, right, bottom))
    if show:
        cropped.show()
    return cropped

def read_svo(svo, stemmer):
    svo = svo.split(",")
    sub = stemmer.stem(svo[0].lower())
    predicate = stemmer.stem(svo[1].lower())
    object = stemmer.stem(svo[2].lower())
    return (sub, predicate, object)

def create_masked_image(image, mask_array, index):
    size = image.size
    image = np.asarray(image).copy()
    predict = ma.masked_where(mask_array == index, mask_array)
    if len(predict.mask.shape) == 0:
        return Image.fromarray(image.astype('uint8'), 'RGB'), None
    mask = np.expand_dims(predict.mask[0], axis=2)
    mask = np.concatenate((mask, mask, mask), axis=2)
    result = image * mask
    result_image = Image.fromarray(result.astype('uint8'), 'RGB')
    return result_image, result

def read_relation(path, image_id, idx_in_json):
    with open(path) as f:
        content = json.load(f)
    relation = Relation(content, image_id, idx_in_json)
    return relation

def create_masked_image_lite(image, mask_array, index):
    image = np.asarray(image).copy()
    predict = ma.masked_where(mask_array == index, mask_array)
    if len(predict.mask.shape) == 0:
        return None
    mask = np.expand_dims(predict.mask[0], axis=2)
    mask = np.concatenate((mask, mask, mask), axis=2)
    result = image * mask
    return result

def crop_svo(image, mask):
    mask = np.sum(mask, axis=2)
    columns = mask.sum(axis = 0)
    rows = mask.sum(axis=1)
    top = np.min(np.nonzero(rows))
    bottom = np.max(np.nonzero(rows))
    left = np.min(np.nonzero(columns))
    right = np.max(np.nonzero(columns))
    cropped = image.crop((left, top, right, bottom))
    cropped_with_black = Image.new(mode="RGB", size=image.size)
    cropped_with_black.paste(cropped, box = (left, top, right, bottom))
    return cropped_with_black, (left, top, right, bottom), cropped

def merge_sub_obj(image, sub, obj, sub_loc, obj_loc):
    sub_obj = Image.new(mode="RGB", size=image.size)
    sub_obj.paste(sub, box = (sub_loc[0], sub_loc[1], sub_loc[2], sub_loc[3]))
    sub_obj.paste(obj, box = (obj_loc[0], obj_loc[1], obj_loc[2], obj_loc[3]))
    return sub_obj

def crop(image, location, with_black):
    x, y, w, h = location
    left = x
    top = y
    right = x + w
    bottom = y + h
    cropped = image.crop((left, top, right, bottom))
    if with_black:
        cropped_with_black = Image.new(mode="RGB", size=image.size)
        cropped_with_black.paste(cropped, box = (x, y, x+w, y+h))
        return cropped, cropped_with_black
    return cropped

def prepare_cropped_image(image, relation, with_black=True, background=False):
    sub_loc = relation.get_sub_location()
    obj_loc = relation.get_obj_location()
    sub, sub_black = crop(image, sub_loc, with_black)
    obj, obj_black = crop(image, obj_loc, with_black)
    copy_image = 0
    if with_black:
        sub_obj = Image.new(mode="RGB", size=image.size)
        sub_obj.paste(sub, box = (sub_loc[0], sub_loc[1], sub_loc[0]+sub_loc[2], sub_loc[1]+sub_loc[3]))
        sub_obj.paste(obj, box = (obj_loc[0], obj_loc[1], obj_loc[0]+obj_loc[2], obj_loc[1]+obj_loc[3]))
        sub_black_mask = Image.new(mode="RGB", size=sub.size)
        obj_black_mask = Image.new(mode="RGB", size=obj.size)
        if background:
            copy_image = image.copy()
            copy_image.paste(sub_black_mask, box = (sub_loc[0], sub_loc[1], sub_loc[0]+sub_loc[2], sub_loc[1]+sub_loc[3]))
            copy_image.paste(obj_black_mask, box = (obj_loc[0], obj_loc[1], obj_loc[0]+obj_loc[2], obj_loc[1]+obj_loc[3]))
        return sub_black, obj_black, sub_obj, copy_image
    else:
        return sub, obj

def complete_sentence(svo):
    vowels = set(["a", "e", "i", 'o', "u"])
    sentence = ""
    s, v, o = svo
    s = s.lower()
    v = v.lower()
    o = o.lower()
    if s[-1] != "s":
        if s[0] in vowels:
            sentence = "an " + s + " is "
        else:
            sentence = "a " + s + " is "
    else:
        sentence = s + " are "
    sentence += v + " "
    if o[-1] != "s":
        if o[0]  in vowels:
            sentence += "an " + o + "."
        else:
            sentence += "a " + o + "."
    else:
        sentence += o + "."
    return sentence