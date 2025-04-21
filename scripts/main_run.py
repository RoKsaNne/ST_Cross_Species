import argparse
import numpy as np
import os
import torch
import pickle 
from datasets import *
from preprocess import *
import random
import time
from train import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    # laod the argument inputs
    parser = argparse.ArgumentParser(description='new_model_mmf')
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--name', type=str, default='run')
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument('--species1_name', type=str, default=None)
    parser.add_argument('--species2_name', type=str, default=None)
    parser.add_argument('--preprocess', action='store_true', help='Preprocessing spatial data')
    parser.add_argument('--distance_thres_ref', type=float, default=20)
    parser.add_argument('--distance_thres_target', type=float, default=20)
    parser.add_argument('--celltype_name_ref', type=str, default='cell_type', help='the obs_name for cell types to use for reference')
    parser.add_argument('--celltype_name_target', type=str, default='cell_type', help='the obs_name for cell types to use for target')
    parser.add_argument('--loc_name_ref', type=str, default='spatial', help='the name of location stored in obsm of reference')
    parser.add_argument('--loc_name_target', type=str, default='spatial', help='the name of location stored in obsm of target')
    parser.add_argument('--region_name_ref', type=str, default=None, help='If slices or different regions exist in reference')
    parser.add_argument('--region_name_target', type=str, default=None, help='If slices or different regions exist in target')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('-b', '--batch-size', default=120, type=int)
    ### loss weight
    parser.add_argument('--beta_cls', type=float, default=1.0)
    parser.add_argument('--beta_mmd', type=float, default=1.0)

    parser.add_argument('--GAT_head', type=int, default=4) 
    parser.add_argument('--latent_dim', type=int, default=128) 
    parser.add_argument('--shared_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--ref_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--target_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--hvg', type=int, default=2000)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    set_seed(args.seed)
    args.savedir = str(args.savedir) + str(args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    print(f"The saving directory set to {args.savedir}", flush=True)

    start_time = time.time()

    # read datasets
    # TODO
    if args.preprocess:
        adata_ref, adata_target, homo_genes_df = read_dataset(args.ref_path, args.target_path, args.species1_name, args.species2_name)
        adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = extract_and_preprocess_data(args, adata_ref, adata_target, args.species1_name, args.species2_name, homo_genes_df)
    
        adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = load_preprocessed(args.savedir)
        ref_X_homo, ref_X_nonhomo, ref_y, ref_edges, inverse_dict_ref, target_X_homo, target_X_nonhomo, target_y_c, target_edges, inverse_dict_target = load_dataset(args,
                                                                                                                                adata_ref_homo, adata_ref_nonhomo, 
                                                                                                                                adata_target_homo, adata_target_nonhomo,
                                                                                                                                args.distance_thres_ref, args.distance_thres_target, 
                                                                                                                                args.celltype_name_ref, args.celltype_name_target, 
                                                                                                                                args.loc_name_ref, args.loc_name_target,
                                                                                                                                args.region_name_ref, args.region_name_target)
        with open(f'{args.savedir}/inverse_dict_test.pkl', 'wb') as f:
            pickle.dump(inverse_dict_ref, f)
        with open(f'{args.savedir}/inverse_dict_target.pkl', 'wb') as f:
            pickle.dump(inverse_dict_target, f)
        
        dataset = GraphDataset(ref_X_homo, ref_X_nonhomo, ref_y, target_X_homo, target_X_nonhomo, target_y_c, ref_edges, target_edges)
        torch.save(dataset.ref_data, f'{args.savedir}/ref_data.pt')
        torch.save(dataset.target_data, f'{args.savedir}/target_data.pt')
    else: 
        dataset = GraphDataset.from_saved(f'{args.savedir}/ref_data.pt', f'{args.savedir}/target_data.pt')
    cross_gae = cross_GAE(args, dataset)
    cross_gae.train()
    pred_labels_ref, gt_ref, latent_ref, pred_labels_target, gt_target,latent_target = cross_gae.pred()
    np.save(f'{args.savedir}/ref_pred.npy', pred_labels_ref)
    np.save(f'{args.savedir}/ref_gt.npy', gt_ref) 
    np.save(f'{args.savedir}/ref_latent.npy', latent_ref) 
    np.save(f'{args.savedir}/target_pred.npy', pred_labels_target)
    np.save(f'{args.savedir}/target_gt.npy', gt_target) 
    np.save(f'{args.savedir}/target_latent.npy', latent_target) 

    print("--- %s seconds ---" % (time.time() - start_time),flush=True)
         
if __name__ == '__main__':
    main()
