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
import optuna

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


def objective(trial, args, dataset, study):
    # setup the hyperparameters to train
    start_time = time.time()
    args.beta_cls = trial.suggest_float("beta_cls", 0.1, 100, log=True)
    args.alpha = trial.suggest_float("alpha", 0.001, 100, log=True)
    
    beta_mmd_choice = trial.suggest_categorical("beta_mmd_choice", [0, "log_sample"])
    if beta_mmd_choice == 0:
        args.beta_mmd = 0
    else:
        args.beta_mmd = trial.suggest_float("beta_mmd", 0.1, 10, log=True)
    beta_coral_choice = trial.suggest_categorical("beta_coral_choice", [0, "log_sample"])
    if beta_coral_choice == 0:
        args.beta_coral = 0
    else:
        args.beta_coral = trial.suggest_float("beta_coral", 0.1, 10, log=True)
    args.beta_kl = trial.suggest_float("beta_kl", 0.001, 10, log=True)
    seed = trial.suggest_int("seed", 1, 1314, step=1)

    set_seed(seed)

    if args.homolog_only:
        cross_gae = cross_GAE_homo(args, dataset)
    elif args.VAE:
        cross_gae = cross_GAE_VAE(args, dataset)
    else:
        cross_gae = cross_GAE(args, dataset)
    cross_gae.train()
    
    ref_data, tgt_data = cross_gae.dataset.ref_data, cross_gae.dataset.target_data

    ref_loader = ClusterLoader(ClusterData(ref_data, num_parts=cross_gae.args.batch_size, recursive=False),
                                batch_size=1, shuffle=True)
    tgt_loader = ClusterLoader(ClusterData(tgt_data, num_parts=cross_gae.args.batch_size, recursive=False),
                                    batch_size=1, shuffle=True)

    pred_labels_ref, gt_ref, latent_ref, pred_labels_target, gt_target, latent_target, \
    latent_ref_homo, latent_tgt_homo, ref_homo_mean, ref_nonhomo_mean, tgt_homo_mean, tgt_nonhomo_mean = \
        cross_gae.pred(ref_loader, tgt_loader)

    pred_labels_ref = pred_labels_ref.to(dtype=torch.long)
    gt_ref = gt_ref.to(dtype=torch.long)

    pred_labels_target = pred_labels_target.to(dtype=torch.long)
    gt_target = gt_target.to(dtype=torch.long)

    correct_predictions_train = (pred_labels_ref == gt_ref).float()
    train_accuracy = correct_predictions_train.mean()

    correct_predictions_test = (pred_labels_target == gt_target).float()
    test_accuracy = correct_predictions_test.mean()


    train_accuracy = correct_predictions_train.mean()
    test_accuracy = correct_predictions_test.mean()

    print(f'Reference Species Prediction Accuracy: {train_accuracy}.')
    print(f'Target Species Prediction Accuracy: {test_accuracy}.')

    # only save the results when the model is the currently the best
    try:
        best_accuracy = test_accuracy > study.best_value
    except ValueError:
        best_accuracy = True
    if best_accuracy:
        file_path = (f'{args.savedir}/best_accuracy.npy')
        with open(file_path, "w") as file:
            file.write(f"Train Accuracy: {train_accuracy}\n")
            file.write(f"Test Accuracy: {test_accuracy}\n")
        np.save(f'{args.savedir}/ref_pred.npy', pred_labels_ref)
        np.save(f'{args.savedir}/ref_gt.npy', gt_ref) 
        np.save(f'{args.savedir}/ref_latent.npy', latent_ref) 
        np.save(f'{args.savedir}/target_pred.npy', pred_labels_target)
        np.save(f'{args.savedir}/target_gt.npy', gt_target) 
        np.save(f'{args.savedir}/target_latent.npy', latent_target) 
        # save reconstructions
        np.save(os.path.join(args.savedir, 'ref_latent_homo.npy'), latent_ref_homo)
        np.save(os.path.join(args.savedir, 'tgt_latent_homo.npy'), latent_tgt_homo)
        np.save(os.path.join(args.savedir, 'ref_homo_mean.npy'), ref_homo_mean)
        np.save(os.path.join(args.savedir, 'ref_nonhomo_mean.npy'), ref_nonhomo_mean)
        np.save(os.path.join(args.savedir, 'tgt_homo_mean.npy'), tgt_homo_mean)
        np.save(os.path.join(args.savedir, 'tgt_nonhomo_mean.npy'), tgt_nonhomo_mean)

    print("--- %s seconds ---" % (time.time() - start_time),flush=True)
    return test_accuracy

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

    parser.add_argument('--k_ref', type=int, default=11)
    parser.add_argument('--k_target', type=int, default=11)

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
    parser.add_argument('-b', '--batch-size', default=10, type=int)
    ### loss weight
    parser.add_argument('--beta_cls', type=float, default=1.0)
    parser.add_argument('--beta_mmd', type=float, default=1.0)
    parser.add_argument('--beta_kl', type=float, default=0.1)
    parser.add_argument('--beta_coral', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)

    parser.add_argument('--GAT_head', type=int, default=1) 
    parser.add_argument('--latent_dim', type=int, default=128) 
    parser.add_argument('--hidden_dim', type=int, default=512) 
    parser.add_argument('--shared_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--ref_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--target_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--hvg', type=int, default=5000)

    parser.add_argument('--homo_gene_id', type=str, default='Gene stable ID')

    # for testing purpose
    parser.add_argument('--homolog_only', action='store_true', help='Using only the homologous genes for training')
    parser.add_argument('--identity_graph', action='store_true', help='Build identity metrices')
    parser.add_argument('--VAE', action='store_true', help='Make the model architecture to VAE')
    parser.add_argument('--denoise', action='store_true', help='Make the model architecture to Denoise VAE')
    parser.add_argument('--condition', action='store_true', help='Make the model architecture to Condition VAE')
    # somewhere in argument setup
    parser.add_argument('--pretrain_epochs', type=int, default=25,
                        help='Number of epochs to pretrain autoencoder on reconstruction only')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    set_seed(args.seed)
    

    start_time = time.time()

    # read datasets
    # TODO
    if args.preprocess:
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        print(f"The saving directory set to {args.savedir}", flush=True)
        adata_ref, adata_target, homo_genes_df = read_dataset(
            args.ref_path,
            args.target_path,
            args.species1_name,
            args.species2_name
        )
        adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = \
            extract_and_preprocess_data(
                args,
                adata_ref,
                adata_target,
                args.species1_name,
                args.species2_name,
                homo_genes_df,
                args.homo_gene_id
            )
    
        # adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = load_preprocessed(args.savedir)
        # ref_X_homo, ref_X_nonhomo, ref_y, ref_edges, inverse_dict_ref, target_X_homo, target_X_nonhomo, target_y_c, target_edges, inverse_dict_target = load_dataset(args,
        #                                                                                                                         adata_ref_homo, adata_ref_nonhomo, 
        #                                                                                                                         adata_target_homo, adata_target_nonhomo,
        #                                                                                                                         args.distance_thres_ref, args.distance_thres_target, 
        #                                                                                                                         args.celltype_name_ref, args.celltype_name_target, 
        #                                                                                                                         args.loc_name_ref, args.loc_name_target,
        #                                                                                                                         args.region_name_ref, args.region_name_target)
        dataset, inverse_dict_ref, inverse_dict_target = load_dataset(
            args,
            adata_ref_homo,
            adata_ref_nonhomo,
            adata_target_homo,
            adata_target_nonhomo,
            args.k_ref,
            args.k_target,
            args.celltype_name_ref,
            args.celltype_name_target,
            args.loc_name_ref,
            args.loc_name_target,
            args.region_name_ref,
            args.region_name_target,
            args.identity_graph
        )
        os.makedirs(args.savedir, exist_ok=True)
        with open(f'{args.savedir}/inverse_dict_ref.pkl', 'wb') as f:
            pickle.dump(inverse_dict_ref, f)
        with open(f'{args.savedir}/inverse_dict_target.pkl', 'wb') as f:
            pickle.dump(inverse_dict_target, f)
        
        # dataset = read_dataset(ref_X_homo, ref_X_nonhomo, ref_y, target_X_homo, target_X_nonhomo, target_y_c, ref_edges, target_edges)
        torch.save(dataset.ref_data, os.path.join(args.savedir, 'ref_data.pt'))
        torch.save(dataset.target_data, os.path.join(args.savedir, 'target_data.pt'))

        print("Saved graph data to:")
        print(f"  {args.savedir}/ref_data.pt")
        print(f"  {args.savedir}/target_data.pt")
        print(f"Inverse dicts saved to:")
        print(f"  {args.savedir}/inverse_dict_ref.pkl")
        print(f"  {args.savedir}/inverse_dict_target.pkl")
    else: 
        dataset = GraphDataset.from_saved(f'{args.savedir}/ref_data.pt', f'{args.savedir}/target_data.pt')

    args.savedir = str(args.savedir) + '/' + str(args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)


    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, args, dataset, study), n_trials=500)

         
if __name__ == '__main__':
    main()
