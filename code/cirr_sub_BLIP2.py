import os
import csv
import json
import utils
import torch
import clip
import numpy as np
from torch.utils.data import dataloader
import torchvision
import datasets

from tqdm import tqdm as tqdm


"""get cirr testset result, save to json"""
def test_cirr_submit_result(model, testset, save_dir, name, txt_processors, batch_size = 16):
    # eval
    model.eval()

    # query feature
    test_queries = testset.test_queries
    all_queries = []
    imgs = []
    mods = []
    pairid = []
    subset = []
    reference_name = []
    for i, data in enumerate(tqdm(test_queries)):
        imgs += [data['reference_data']]
        mods += [data['mod']]
        pairid += [data['pairid']]
        reference_name += [data['reference_name']]
        subset.append(list(data['subset']))
        if len(imgs) >= batch_size or i == len(test_queries) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            q = model.extract_retrieval_compose(imgs, mods).data.cpu().numpy()
            all_queries += [q]
            imgs = []
            mods = []
    all_queries = np.concatenate(all_queries)

    # targets feature
    candidate_names, candidate_img = testset.test_name_list, testset.test_img_data
    candidate_features = []
    imgs = []
    for i, img_data in enumerate(tqdm(candidate_img)):
        imgs += [img_data]
        if len(imgs) >= batch_size or i == len(candidate_img) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float().cuda()
            features = model.extract_retrieval_target(imgs).data.cpu().numpy()
            candidate_features += [features]
            imgs = []
    candidate_features = np.concatenate(candidate_features) 
    sims = np.matmul(all_queries, candidate_features)
    sims = sims.squeeze()
    sims = sims.max(-1)
    for i, t in enumerate(test_queries):
        sims[i, candidate_names.index(t['reference_name'])] = -10e10
    sims = -sims
    sorted_inds = np.argsort(sims, axis=-1)
    sorted_ind_names = np.array(candidate_names)[sorted_inds]

    mask = torch.tensor(sorted_ind_names != np.repeat(np.array(reference_name), len(candidate_names)).reshape(len(sorted_ind_names),-1)) # (M,N)
    sorted_ind_names = sorted_ind_names[mask].reshape(sorted_ind_names.shape[0], sorted_ind_names.shape[1] - 1) # (M,N-1)

    subset = np.array(subset)
    subset_mask = (sorted_ind_names[..., None] == subset[:, None, :]).sum(-1).astype(bool) # (M,N-1) label elements in subset
    sorted_subset_names = sorted_ind_names[subset_mask].reshape(sorted_ind_names.shape[0], -1) # (M,6)

    pairid_to_gengeral_pred = {str(int(pair_id)): prediction[:50].tolist()  for pair_id, prediction in zip(pairid, sorted_ind_names)}
    pairid_to_subset_pred = {str(int(pair_id)): prediction[:3].tolist() for pair_id, prediction in zip(pairid, sorted_subset_names)}

    general_submission = {'version': 'rc2', 'metric': 'recall'}
    subset_submission = {'version': 'rc2', 'metric': 'recall_subset'}

    general_submission.update(pairid_to_gengeral_pred)
    subset_submission.update(pairid_to_subset_pred)

    print('save cirr test result')
    with open(os.path.join(save_dir, f'CIRR_pred_ranks_recall{name}.json'), 'w+') as f:
        json.dump(general_submission, f, sort_keys=True)
        
    with open(os.path.join(save_dir, f'CIRR_pred_ranks_recall_subset{name}.json'), 'w+') as f:
        json.dump(subset_submission, f, sort_keys=True)

if __name__ == '__main__':
    from lavis.models import load_model_and_preprocess
    blip_model_name = "FineCIR"
    backbone = "pretrain"
    model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device="cuda")

    from data_utils import squarepad_transform, targetpad_transform

    transform = "targetpad"
    input_dim = 224
    target_ratio = 1.25
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    testset = datasets.FineCIRR(path="...",transform=preprocess)
    import sys
    model_dir = sys.argv[1]
    import os
    
    file_ls = os.listdir(model_dir) 
    for i in file_ls:
        if ".pt" in i and f'CIRR_pred_ranks_recall{i[:-3]}.json' not in file_ls:
            model = torch.load(os.path.join(model_dir, i))
            print(i[:-3] + " start")
            test_cirr_submit_result(model, save_dir=model_dir, testset=testset, batch_size=128, name=i[:-3], txt_processors=txt_processors)
            print(i[:-3] + " end")