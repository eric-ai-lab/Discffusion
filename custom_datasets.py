import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import PIL
import numpy as np
import torch
from torchvision import datasets
from glob import glob
import pandas as pd
import ast
import os.path as osp
import pickle
import time
import re
from datasets import load_dataset
from utils import image_preprocess

def get_dataset(dataset_name, root_dir, transform=None, resize=512, mode='eval'):
    if dataset_name == 'Refcocog':
        return RefcocogDataset(root_dir, transform, resize=resize, mode=mode) 
    elif dataset_name == 'winoground':
        if mode == 'validation':
            raise ValueError(f'No split found for {mode}')
        return WinogroundDataset(root_dir, transform, resize=resize, mode=mode)
    elif dataset_name == 'vqa':
        return VQADataset_train(root_dir, transform) 
    elif dataset_name == 'vqa_other':
        return VQADataset_eval(root_dir, transform, type='others') 
    elif dataset_name == 'vqa_binary':
        return VQADataset_eval(root_dir, transform, type='binary') 
    elif dataset_name == 'ComVG':
        if mode == 'validation':
            raise ValueError(f'No split found for {mode}')
        return ComVGClassificationDataset(root_dir, transform, mode=mode)
    elif dataset_name == 'ComVG_sub':
        if mode == 'validation':
            raise ValueError(f'No split found for {mode}')
        return ComVGClassificationDataset(root_dir, transform, mode=mode, neg_type='subject')
    elif dataset_name == 'ComVG_obj':
        if mode == 'validation':
            raise ValueError(f'No split found for {mode}')
        return ComVGClassificationDataset(root_dir, transform, mode=mode, neg_type='object')
    elif dataset_name == 'ComVG_verb':
        if mode == 'validation':
            raise ValueError(f'No split found for {mode}')
        return ComVGClassificationDataset(root_dir, transform, mode=mode, neg_type='verb')
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    



class REFER:
    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'refcocog/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs('+splitBy+').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time()-tic))
    
    def createIndex(self):
        print('creating index...')
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids)==len(cat_ids)==len(ref_ids)==len(split)==0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']] # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getImgPath(self, img_id):
        image_info = self.Imgs[img_id]
        return os.path.join(self.IMAGE_DIR, image_info['file_name'])
    
    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

          

class RefcocogRetrieval:
    def __init__(self, data_root, dataset='refcocog', splitBy='google', mode='text'):
    # def __init__(self, data_root, dataset='refcocog', splitBy='unc'):
        self.refer = REFER(data_root, dataset, splitBy)
        self.mode = mode
        self.prepare_data()

    def prepare_data(self):
        self.pairs = []
        all_image_ids = list(self.refer.Imgs.keys())
        all_refs = list(self.refer.Refs.values())

        for ref_id, ref in self.refer.Refs.items():
            pos_img_id = ref['image_id']
            for sent in ref['sentences']:
                pos_text = sent['raw']
                self.pairs.append((sent['sent_id'], pos_text, pos_img_id, 1))
                
                if self.mode == 'image':
                    negative_samples = 0
                    while negative_samples < 9:
                        neg_img_id = random.choice(all_image_ids)
                        if neg_img_id != pos_img_id:
                            self.pairs.append((sent['sent_id'], pos_text, neg_img_id, 0))
                            negative_samples += 1
                elif self.mode == 'text':
                    negative_text_samples = 0
                    while negative_text_samples < 9:
                        neg_ref = random.choice(all_refs)
                        neg_text = random.choice(neg_ref['sentences'])['raw']
                        if neg_text != pos_text:
                            self.pairs.append((sent['sent_id'], neg_text, pos_img_id, 0))
                            negative_text_samples += 1

    def get_pairs(self):
        return self.pairs
    
    

class RefcocogDataset(Dataset):

    def __init__(self, root_dir, transform, resize=512, save_path="./saved_images", mode='eval', train_data_path='./refcoco_train_data.pkl'):
        self.transform = transform
        self.root_dir = root_dir
        self.retrieval = RefcocogRetrieval('..')
        self.resize = resize
        self.save_path = save_path
        self.mode = mode
        os.makedirs(self.save_path, exist_ok=True)

        all_pairs = self.retrieval.get_pairs()
        total_pairs = len(all_pairs)
        print(f"Length of all pairs: {total_pairs}")
        if os.path.exists(train_data_path):
            with open(train_data_path, 'rb') as f:
                train_data = pickle.load(f)
            print(f"Loading existing training data from {train_data_path}")
        else:
            total_pairs = len(all_pairs)
            train_size = int(0.05 * total_pairs)
            train_size -= train_size % 10  
            train_data = all_pairs[:train_size]
            with open(train_data_path, 'wb') as f:
                pickle.dump(train_data, f)
            print(f"Creating and saving new training data to {train_data_path}")

        train_size = len(train_data)
        if mode == 'train':
            print(f"Using {train_size} training pairs")
            self.pairs = train_data
        elif mode == 'eval':  
            print(f"Using {total_pairs - train_size} evaluation pairs")
            self.pairs = all_pairs[train_size:]
        else:
            raise ValueError(f'Unknown mode {mode}')

    @staticmethod
    def sanitize_text(text):
        text = re.sub(r'[^\w\s]', '', text) 
        text = text.replace(' ', '_') 
        return text[:50]  

    def save_image(self, img, path, prefix, text):
        sanitized_text = self.sanitize_text(text)
        text_save_path = os.path.join(self.save_path, sanitized_text, prefix)
        os.makedirs(text_save_path, exist_ok=True)  
        img_filename = os.path.basename(path)
        img.save(os.path.join(text_save_path, img_filename))

    def __getitem__(self, idx):
        idx *= 10
        _, text, pos_img_id, _ = self.pairs[idx]
        img_path = self.retrieval.refer.getImgPath(pos_img_id)
        pos_img = Image.open(img_path).convert("RGB")
        
        texts = [text]
        
        for i in range(1, 10):
            _, neg_text, _, _ = self.pairs[idx + i]
            texts.append(neg_text)

        if self.transform:
            pos_img = self.transform(pos_img).unsqueeze(0)
        else:
            pos_img = pos_img.resize((self.resize, self.resize))
            pos_img = image_preprocess(pos_img)

        return ([img_path], [pos_img]), texts, 0

        
    def vis_image(self, idx):
        idx *= 10  
        _, text, pos_img_id, _ = self.pairs[idx]
        img_path = self.retrieval.refer.getImgPath(pos_img_id)
        pos_img = Image.open(img_path).convert("RGB")

        imgs = [pos_img]
        imgpaths = [img_path]

        self.save_image(pos_img, img_path, "positive", text)
                    
        for i in range(1, 10):
            _, _, neg_img_id, _ = self.pairs[idx + i]
            negimg_path = self.retrieval.refer.getImgPath(neg_img_id)
            neg_img = Image.open(negimg_path).convert("RGB")
            imgs.append(neg_img)
            imgpaths.append(negimg_path)

            self.save_image(neg_img, negimg_path, "negative", text)

        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [image_preprocess(img) for img in imgs_resize] 

        return (imgpaths, imgs_resize), [text], 0
        
    def __len__(self):
        return len(self.pairs) // 10


    
class ImageTextDataset(Dataset):
    def __init__(self, relationship_dir, text_csv_path, image_dir, with_black=True, transform=None):
        self.image_dir = image_dir
        self.relationship_dir = relationship_dir
        self.text_df = pd.read_pickle(text_csv_path)
        self.with_black = with_black
        self.transform = transform
    
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

        sub_index = raw_sentence.index(sub)
        obj_index = raw_sentence.index(obj)
        predicate_index = raw_sentence.index(pred)
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
                     
    
    

def get_train_dataset(args):
    dataset = ImageTextDataset(args.relation_folder, args.text_path, args.image_folder, with_black=args.with_black, transform = transforms)

    total_len = len(dataset)
    train_len = int(total_len * 0.05) 
    val_len = total_len - train_len 
    train, val = torch.utils.data.random_split(dataset, [train_len, val_len])
    return train, val
        


class ComVGClassificationDataset(Dataset):

    def __init__(self, root_dir, transform, resize=512, neg_type='all', save_path="./saved_images", mode='eval'):
        self.transform = transform
        self.root_dir = root_dir
        self.data = self.load_data(self.root_dir, neg_type=neg_type, mode=mode)
        self.resize = resize
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  

    @staticmethod
    def sanitize_text(text):
        text = re.sub(r'[^\w\s]', '', text)  
        text = text.replace(' ', '_')  
        return text[:50]  

    def save_image(self, img, path, prefix, text):
        sanitized_text = self.sanitize_text(text)
        text_save_path = os.path.join(self.save_path, sanitized_text, prefix)

        index = 0
        base_sanitized_text = sanitized_text
        while os.path.exists(text_save_path):
            index += 1
            sanitized_text = f"{base_sanitized_text}_{index}"
            text_save_path = os.path.join(self.save_path, sanitized_text, prefix)

        os.makedirs(text_save_path, exist_ok=True)  
        img_filename = os.path.basename(path)
        img.save(os.path.join(text_save_path, img_filename))

    def load_data(self, data_dir, neg_type='all', mode='eval', trainsplit = None):
        print("Loading data from {} with {}".format(data_dir, mode))
        
        dataset = []

        if mode == 'eval':
            split_file = os.path.join(data_dir, 'compositional_visual_genome.csv')
            text_df = pd.read_csv(split_file)

            for _, row in text_df.iterrows():
                if neg_type != 'all' and row['neg_value'] != neg_type:
                    continue

                sentence = row['sentence']
                pos_file = os.path.join('../Visual_Genome/vg_concept', str(row['pos_image_id']) + ".jpg")
                neg_file = os.path.join('../Visual_Genome/vg_concept', str(row['neg_image_id']) + ".jpg")
                dataset.append((pos_file, neg_file, sentence))

            print("Loaded {} data".format(len(dataset)))
            return dataset
        
        elif mode == 'train':
            split_file = os.path.join(data_dir, 'compositional_visual_genome.csv')
            total_rows = sum(1 for line in open(split_file)) - 1 
            nrows_5_percent = int(total_rows * 0.05)
            if trainsplit is None:
                split_file = os.path.join(data_dir, 'comvg_train.csv')
                total_rows = sum(1 for line in open(split_file)) - 1 
                nrows_5_percent = int(total_rows * 0.05)
                text_df = pd.read_csv(split_file, nrows=nrows_5_percent)

                for _, row in text_df.iterrows():
                    sentence = row['sentence']
                    pos_file = os.path.join('../Visual_Genome/VG_100K', str(row['pos_image_id']) + ".jpg")
                    neg_file = os.path.join('../Visual_Genome/VG_100K', str(row['neg_image_id']) + ".jpg")
                    dataset.append((pos_file, neg_file, sentence))

                print("Loaded {} data".format(len(dataset)))
                return dataset
            else:
                text_df = pd.read_csv(split_file, nrows=nrows_5_percent)
                

                for _, row in text_df.iterrows():
                    if neg_type != 'all' and row['neg_value'] != neg_type:
                        continue

                    sentence = row['sentence']
                    pos_file = os.path.join('../Visual_Genome/vg_concept', str(row['pos_image_id']) + ".jpg")
                    neg_file = os.path.join('../Visual_Genome/vg_concept', str(row['neg_image_id']) + ".jpg")
                    dataset.append((pos_file, neg_file, sentence))

                print("Loaded {} data".format(len(dataset)))
                return dataset

    def __getitem__(self, idx):
        pos_file, neg_file, text = self.data[idx]
        pos_img = Image.open(pos_file).convert("RGB")
        neg_img = Image.open(neg_file).convert("RGB")

        self.save_image(pos_img, pos_file, "positive", text)
        self.save_image(neg_img, neg_file, "negative", text)

        imgs = [pos_img, neg_img]
        imgpaths = [pos_file, neg_file]

        if self.transform:
            imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
        else:
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            imgs_resize = [image_preprocess(img) for img in imgs_resize]

        return (imgpaths, imgs_resize), [text], 0

    def __len__(self):
        return len(self.data)
    


class VQADataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, mode='eval', split='test', type='all'):
        self.root_dir = "../vqav2"
        self.resize = resize
        self.mode = mode
        if 'val' in mode:
            self.data = self.load_data(f'{self.root_dir}/{split}.txt', type=type)
        else:
            data_path = os.path.join(root_dir, 'vqa_text_train.csv')
            full_data = self.load_train_data(pd.read_csv(data_path))
            self.data = full_data[:int(len(full_data) * 0.05)] 
        self.transform = transform


    def load_train_data(self, df):
        dataset = []
        for _, row in df.iterrows():
            sentence = row['sentence']
            pos_img_id = f"COCO_train2014_{str(row['pos_image_id']).zfill(12)}.jpg"
            neg_img_ids = [f"COCO_train2014_{str(row[f'neg_image_id_{i+1}']).zfill(12)}.jpg" for i in range(9)]
            dataset.append((pos_img_id, neg_img_ids, sentence))
        return dataset
    
    
    def load_data(self, file_path, type='all'):
        dataset = []
        all_imgs = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                img_path, text = line.strip().split('*')
                question_type = None
                if "The question is asking about yes or no" in text:
                    question_type = "binary"
                elif "The question is asking about others" in text:
                    question_type = "others"
                
                if type != "all" and type != question_type:
                    continue
                
                text = text.split(': ', 1)[-1]
                
                all_imgs.append(img_path)
                dataset.append((img_path, text))
        for i in range(len(dataset)):
            img_path, text = dataset[i]
            negative_samples = [neg for neg in all_imgs if neg != img_path]
            negative_samples = random.sample(negative_samples, 9)  
            dataset[i] = (img_path, negative_samples, text)
        return dataset
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos_img_path, neg_img_paths, text = self.data[idx]
        if 'val' in self.mode:
            pos_img = Image.open(f'{self.root_dir}/val2014/{pos_img_path}').convert("RGB")
            neg_imgs = [Image.open(f'{self.root_dir}/val2014/{neg_path}').convert("RGB") for neg_path in neg_img_paths]
        else:
            pos_img = Image.open(f'{self.root_dir}/train2014/{pos_img_path}').convert("RGB")
            neg_imgs = [Image.open(f'{self.root_dir}/train2014/{neg_path}').convert("RGB") for neg_path in neg_img_paths]
        
        if self.transform:
            pos_img_resize = self.transform(pos_img).unsqueeze(0)
            neg_imgs_resize = [self.transform(img).unsqueeze(0) for img in neg_imgs]
        else:
            pos_img_resize = pos_img.resize((self.resize, self.resize))
            pos_img_resize = image_preprocess(pos_img_resize)  
            neg_imgs_resize = [img.resize((self.resize, self.resize)) for img in neg_imgs]
            neg_imgs_resize = [image_preprocess(img) for img in neg_imgs_resize]
        return ([pos_img_path] + neg_img_paths, [pos_img_resize] + neg_imgs_resize), [text], 0
    



class VQADataset_train(Dataset):
    def __init__(self, root_dir, transform, resize=512):
        self.root_dir = "../vqav2"
        self.resize = resize
        data_path = os.path.join(root_dir, 'vqa_text_train.csv')
        full_data = self.load_train_data(pd.read_csv(data_path))
        self.data = full_data[:int(len(full_data) * 0.05)] 
        self.transform = transform


    def load_train_data(self, df):
        dataset = []
        for _, row in df.iterrows():
            sentence = row['sentence']
            pos_img_id = f"COCO_train2014_{str(row['pos_image_id']).zfill(12)}.jpg"
            neg_sentences = [row[f'neg_sentence_{i+1}'] for i in range(9)]
            dataset.append((pos_img_id, neg_sentences, sentence))
        return dataset
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos_img_path, neg_samples, text = self.data[idx]
        pos_img = Image.open(f'{self.root_dir}/train2014/{pos_img_path}').convert("RGB")

        if self.transform:
            pos_img_resize = self.transform(pos_img).unsqueeze(0)
        else:
            pos_img_resize = pos_img.resize((self.resize, self.resize))
            pos_img_resize = image_preprocess(pos_img_resize)

        return ([pos_img_path], [pos_img_resize]), [text] + neg_samples, 0  
    


class VQADataset_eval(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='test3', type='all'):
        self.root_dir = "../vqav2"
        self.resize = resize
        self.neg_samples_file = os.path.join(self.root_dir, f'{split}_neg_samples.json')
        
        if type == 'others':
            data_path = os.path.join(root_dir, 'vqa_text_test.csv')
            self.data = self.load_val_data(pd.read_csv(data_path))
        elif type == 'binary':
            self.data = self.load_data(f'{self.root_dir}/{split}.txt', type=type)
        self.transform = transform


    def save_neg_samples(self, dataset):
        with open(self.neg_samples_file, 'w') as file:
            json.dump(dataset, file)


    def load_val_data(self, df):
        dataset = []
        for _, row in df.iterrows():
            sentence = row['sentence']
            pos_img_id = f"COCO_val2014_{str(row['pos_image_id']).zfill(12)}.jpg"
            neg_sentences = [row[f'neg_sentence_{i+1}'] for i in range(9)]
            dataset.append((pos_img_id, neg_sentences, sentence))
        return dataset
    

    def load_data(self, file_path, type='all'):
        if os.path.exists(self.neg_samples_file):
            with open(self.neg_samples_file, 'r') as file:
                return json.load(file)
                
        dataset = []
        all_imgs = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                img_path, text = line.strip().split('*')
                question_type = None
                if "The question is asking about yes or no" in text:
                    question_type = "binary"
                elif "The question is asking about others" in text:
                    question_type = "others"
                
                if type != "all" and type != question_type:
                    continue
                
                text = text.split(': ', 1)[-1]
                
                all_imgs.append(img_path)
                dataset.append((img_path, text))

                
        for i in range(len(dataset)):
            img_path, text = dataset[i]
            negative_samples = [neg for neg in all_imgs if neg != img_path]
            negative_samples = random.sample(negative_samples, 9)  
            dataset[i] = (img_path, negative_samples, text)

        self.save_neg_samples(dataset)
        return dataset
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos_img_path, neg_samples, text = self.data[idx]
        pos_img = Image.open(f'{self.root_dir}/val2014/{pos_img_path}').convert("RGB")

        if self.transform:
            pos_img_resize = self.transform(pos_img).unsqueeze(0)
        else:
            pos_img_resize = pos_img.resize((self.resize, self.resize))
            pos_img_resize = image_preprocess(pos_img_resize)

        if isinstance(neg_samples[0], str) and neg_samples[0].endswith('.jpg'):  # binary
            neg_imgs = [Image.open(f'{self.root_dir}/val2014/{neg_path}').convert("RGB") for neg_path in neg_samples]
            
            if self.transform:
                neg_imgs_resize = [self.transform(img).unsqueeze(0) for img in neg_imgs]
            else:
                neg_imgs_resize = [img.resize((self.resize, self.resize)) for img in neg_imgs]
                neg_imgs_resize = [image_preprocess(img) for img in neg_imgs_resize]
            
            return ([pos_img_path] + neg_samples, [pos_img_resize] + neg_imgs_resize), [text], 0  

        else:  # others
            return ([pos_img_path], [pos_img_resize]), [text] + neg_samples, 0  
    
