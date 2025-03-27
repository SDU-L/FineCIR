import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F
import pickle
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def test(params, model, testset, category, txt_processors):
    model.eval()
    if category == 'dress':
        (test_queries, test_targets, name) = (testset.test_queries_dress, testset.test_targets_dress, 'dress')
    elif category == 'shirt':
        (test_queries, test_targets, name) = (testset.test_queries_shirt, testset.test_targets_shirt, 'shirt')
    elif category == 'toptee':
        (test_queries, test_targets, name) = (testset.test_queries_toptee, testset.test_targets_toptee, 'toptee')

    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in tqdm(test_queries):
                imgs += [t['source_img_data']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    f = model.extract_retrieval_compose(imgs, mods)
                    f = f.data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    sims = np.matmul(all_queries, all_imgs)
    sims = sims.squeeze()
    sims = sims.max(-1)

    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    saved_list = []

    nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
    for k in [1, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
            if test_targets_id.index(test_queries[i]['target_img_id']) not in nns[:50]:
                saved_list.append(test_queries[i]['mod']['str'])
        r = 100 * r / len(nn_result)
        out += [('{}_r{}'.format(name, k), r)]
    print(saved_list)
    return out

def test_cirr_valset(params, model, testset, txt_processors):
    test_queries, test_targets = testset.val_queries, testset.val_targets
    with torch.no_grad():
        all_queries = []
        all_imgs = []
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in tqdm(test_queries):
                imgs += [t['source_img_data']]
                mods += [t['mod']['str']]
                if len(imgs) >= params.batch_size or t is test_queries[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    f = model.extract_retrieval_compose(imgs, mods).data.cpu().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)

            # compute all image features
            imgs = []
            for t in tqdm(test_targets):
                imgs += [t['target_img_data']]
                if len(imgs) >= params.batch_size or t is test_targets[-1]:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float().cuda()
                    imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()
                    all_imgs += [imgs]
                    imgs = []
            all_imgs = np.concatenate(all_imgs)

    sims = np.matmul(all_queries, all_imgs) 
    sims = sims.squeeze()
    sims = sims.max(-1)
    
    test_targets_id = []
    for i in test_targets:
        test_targets_id.append(i['target_img_id'])
    for i, t in enumerate(test_queries):
        sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


    nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] # (m,n)

    # all set recalls
    cirr_out = []
    for k in [1, 5, 10, 50]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                r += 1
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_r{}'.format(params.dataset,k), r)]

    # subset recalls
    err = []
    for k in [1, 2, 3]:
        r = 0.0
        for i, nns in enumerate(nn_result):

            subset = np.array([test_targets_id.index(idx) for idx in test_queries[i]['subset_id']]) # (6)
            subset_mask = (nns[..., None] == subset[None, ...]).sum(-1).astype(bool) # (n,1)==(1,6) => (n,6) => (n)
            subset_label = nns[subset_mask] # (6)
            if test_targets_id.index(test_queries[i]['target_img_id']) in subset_label[:k]:
                r += 1
            if test_targets_id.index(test_queries[i]['target_img_id']) not in subset_label[:1]:
                err.append(test_queries[i]['mod']['str'])
        r = 100 * r / len(nn_result)
        cirr_out += [('{}_subset_r{}'.format(params.dataset, k), r)]
    # print(err)
    return cirr_out
