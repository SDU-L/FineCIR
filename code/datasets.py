"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import pickle
from tqdm import trange

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
def save_jsonl(file_path, datalist):
    with open(file_path, "w", encoding='utf-8') as f:
        for entry in datalist:
            f.write(json.dumps(entry) + '\n')

class FineCIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path 
        self.caption_dir = self.path + 'captions'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform

        # train data
        self.cirr_data = read_jsonl(os.path.join(self.caption_dir, "cirr.fine.train.jsonl"))

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys()) 

        # val data
        if not os.path.exists(os.path.join(self.path, 'fcirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'fcirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'fcirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'fcirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'fcirr_val_targets.pkl'))

        # test data
        if not os.path.exists(os.path.join(self.path, 'fcirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'fcirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'fcirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'fcirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'fcirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'fcirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'fcirr_test_queries.pkl'))


    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['finemt']
        target_name = caption['target_hard']


        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name], 0)

        out['target_img_data'] = self.get_img(self.train_image_path[target_name], 0)

        out['mod'] = {'str':mod_str}
        return out

    
    def get_img(self, img_path, stage=0):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        return img
    
    def get_val_queries(self):
        val_data = read_jsonl(os.path.join(self.caption_dir, "cirr.fine.val.jsonl"))

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())
        
        test_queries = []
        
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['finemt']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_path[reference_name], 1)
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_path[target_name], 1)
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids

            test_queries.append(out)

        test_targets = []
        
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_path[name], 1)

            test_targets.append(out)

        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        print("test_image_name")
        for i in trange(len(test_data)):

            out = {}
            caption = test_data[i]

            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']], 1)

            out['reference_name'] = caption['reference']
            out['mod'] = caption['finemt']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in trange(len(test_image_name)):
            name = test_image_name[i]

            data = self.get_img(test_image_path[name], 1)
            image_name.append(name)
            image_data.append([data])
        return image_name, image_data, queries


class FineFashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        
        if not os.path.exists(os.path.join(self.path, 'ffashion_iq_data.json')):
            self.fashioniq_data = []
            self.train_init_process()
            with open(os.path.join(self.path, 'ffashion_iq_data.json'), 'w') as f:
                json.dump(self.fashioniq_data, f)

            self.test_queries_dress, self.test_targets_dress = self.get_test_data('dress')
            self.test_queries_shirt, self.test_targets_shirt = self.get_test_data('shirt')
            self.test_queries_toptee, self.test_targets_toptee = self.get_test_data('toptee')
            save_obj(self.test_queries_dress, os.path.join(self.path, 'test_queries_dress.pkl'))
            save_obj(self.test_targets_dress, os.path.join(self.path, 'test_targets_dress.pkl'))
            save_obj(self.test_queries_shirt, os.path.join(self.path, 'test_queries_shirt.pkl'))
            save_obj(self.test_targets_shirt, os.path.join(self.path, 'test_targets_shirt.pkl'))
            save_obj(self.test_queries_toptee, os.path.join(self.path, 'test_queries_toptee.pkl'))
            save_obj(self.test_targets_toptee, os.path.join(self.path, 'test_targets_toptee.pkl'))

        else:
            with open(os.path.join(self.path, 'ffashion_iq_data.json'), 'r') as f:
                self.fashioniq_data = json.load(f) 
            self.test_queries_dress = load_obj(os.path.join(self.path, 'test_queries_dress.pkl'))
            self.test_targets_dress = load_obj(os.path.join(self.path, 'test_targets_dress.pkl'))
            self.test_queries_shirt = load_obj(os.path.join(self.path, 'test_queries_shirt.pkl'))
            self.test_targets_shirt = load_obj(os.path.join(self.path, 'test_targets_shirt.pkl'))
            self.test_queries_toptee = load_obj(os.path.join(self.path, 'test_queries_toptee.pkl'))
            self.test_targets_toptee = load_obj(os.path.join(self.path, 'test_targets_toptee.pkl'))


    def train_init_process(self):
        for name in ['dress', 'shirt', 'toptee']:
            ref_captions = read_jsonl(os.path.join(self.caption_dir, "fiq.fine.{}.{}.jsonl".format(name, 'train')))

            print(len(ref_captions))
            for triplets in ref_captions:
                ref_id = triplets['candidate']
                tag_id = triplets['target']

                cap = triplets['finemt']
                self.fashioniq_data.append({
                    'target': name + '_' + tag_id,
                    'candidate': name + '_' + ref_id,
                    'finemt': cap
                })


    def __len__(self):
        return len(self.fashioniq_data)

    def __getitem__(self, idx):
        caption = self.fashioniq_data[idx]
        mod_str = caption['finemt']
        candidate = caption['candidate']
        target = caption['target']
    
        out = {}
        out['source_img_data'] = self.get_img(candidate, stage=0)#candidate_img#

        out['target_img_data'] = self.get_img(target, stage=0)#target_img#
        
        out['mod'] = {'str': mod_str}

        return out

    def get_img(self, image_name, stage=0):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        return img
    


    def get_test_data(self, name):
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
            images = json.load(f)

        ref_captions = read_jsonl(os.path.join(self.caption_dir, "fiq.fine.{}.{}".format(name, 'val')))
        test_queries = []
        for idx in trange(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = caption['finemt']
            candidate = caption['candidate']
            target = caption['target']

            out = {}
            out['source_img_id'] = images.index(candidate)
            out['target_img_id'] = images.index(target)
            out['source_img_data'] = self.get_img(name + '_' + candidate, stage=0)#candidate_img#

            out['target_img_data'] = self.get_img(name + '_' + target, stage=0)#target_img#
            
            out['mod'] = {'str': mod_str}

            test_queries.append(out)

        test_targets_id = []
        test_targets = []
        for i in test_queries:
            if i['source_img_id'] not in test_targets_id:
                test_targets_id.append(i['source_img_id'])
            if i['target_img_id'] not in test_targets_id:
                test_targets_id.append(i['target_img_id'])
        
        for i in test_targets_id:
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(name + '_' + images[i], stage=1)       
            test_targets.append(out)
        return test_queries, test_targets



