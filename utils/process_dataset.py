import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL
# from clip.clip import * 
import numpy as np
# from utils.helper_functions import prepare_cropped_image, read_relation, read_svo, create_masked_image, create_masked_image_lite, crop_svo, merge_sub_obj

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import torch
import argparse
from torchvision import transforms, utils
from torchvision.transforms import Resize, CenterCrop, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import multiprocessing as mp
import pandas as pd
import copy
import random
import pyarrow as pa
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pyarrow as pa
ps = PorterStemmer()




class ImageTextDataset(Dataset):
    def __init__(self, relationship_dir, text_csv_path, image_dir, with_black=True, transform=None):
        self.image_dir = image_dir
        self.relationship_dir = relationship_dir
        self.text_df = pd.read_pickle(text_csv_path)
        self.with_black = with_black
        self.transform = transform


        sample_df = self.text_df.rename(columns={'text': 'sentence', 'image_id': 'pos_image_id'})

        # Shuffle the DataFrame
        shuffled_df = sample_df.sample(frac=1).reset_index(drop=True)
        
        # Find where the 'pos_image_id' in the shuffled DataFrame is the same as in the original
        same_image_mask = (shuffled_df['pos_image_id'] == sample_df['pos_image_id'])

        # Re-shuffle those specific rows
        while same_image_mask.any():
            shuffled_df.loc[same_image_mask] = shuffled_df.loc[same_image_mask].sample(frac=1).reset_index(drop=True)
            same_image_mask = (shuffled_df['pos_image_id'] == sample_df['pos_image_id'])

        sample_df['neg_image_id'] = shuffled_df['pos_image_id']
        sample_df['neg_image_id'] = sample_df['neg_image_id'].astype('Int64')  # This will convert to integer but keep NaN as NaN
        sample_df.to_csv("./comvg_train.csv", index=False)
    
    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.text_df.iloc[[idx]]
        raw_sentence = row.text.item()[:-1]
        raw_sentence = raw_sentence.split(" ")
        sentence = tokenize(row.text.item())
        image_filename = str(row.image_id.item()) + ".jpg"
        relation_filename = row.relation_json_path.item()

        image = os.path.join(self.image_dir, image_filename)
        image = Image.open(image).convert('RGB')
        relation = read_relation(relation_filename, row.image_id.item(), row.relation_idx.item())

        sub = relation.get_sub_name().split(" ")[0].lower()
        obj = relation.get_obj_name().split(" ")[0].lower()
        pred = relation.get_predicate().split(" ")[0].lower()

        sub_index = 0
        obj_index = 0
        predicate_index = 0
        length = len(raw_sentence)
        
        if self.with_black:
            subject, object, subj_obj, _ = prepare_cropped_image(image, relation, self.with_black)
            if self.transform:
                subject = self.transform(subject)
                object = self.transform(object)
                subj_obj = self.transform(subj_obj)
                image = self.transform(image)
            sample = {"text": sentence, "image": image, "subject_text": tokenize(relation.get_sub_name().lower()), "object_text": tokenize(relation.get_obj_name().lower()), 
                                        "predicate": tokenize(relation.predicate.lower()), "subject_image": subject, "object_image": object, "subject_object_image": subj_obj, "sub_idx": sub_index, "obj_idx": obj_index, "predicate_idx": predicate_index, "length": length, "string":row.text.item(), "image_idx": row.image_id.item()}
        else:
            subject, object = prepare_cropped_image(image, relation, self.with_black)
            sample = {"text": sentence, "image": image, "subject_text": relation.get_sub_name(), "object_text": relation.get_obj_name(), 
                                        "predicate": relation.predicate, "subject_image": subject, "object_image": object}
        return sample
    

def image_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h)) 
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0



def main():

    data="vqa"

    torch.manual_seed(42)

    transforms = transforms.Compose([Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                                    CenterCrop(size=(224, 224)),
                                    transforms.ToTensor(),
                                    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))]
                                    )


    if data == "comvg":
        image_folder='./Visual_Genome/VG_100K'
        relation_folder='/parsed_relation'
        text_path='./text_df.pkl'
        with_black=True
        dataset = ImageTextDataset(relation_folder, text_path, image_folder, with_black=with_black, transform = transforms)
        train, val = torch.utils.data.random_split(dataset, [56490, len(dataset)-56490])
        train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=12)
    elif data == "vqa":
        dataset = pd.read_csv("./data/vqa_text_val.csv")
        dataset = pd.read_csv("./data/vqa_text_train.csv")
        print(dataset.head())
    else:
        print("Please specify the correct dataset")
        return


if __name__=="__main__":
    main()