import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import ClusterData, ClusterLoader
from tqdm import tqdm
from plot import *
import os

class cross_GAE:
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        # input dims
        args.shared_input_dim = dataset.ref_data.homo_x.shape[-1]
        args.ref_input_dim    = dataset.ref_data.nonhomo_x.shape[-1]
        args.target_input_dim = dataset.target_data.nonhomo_x.shape[-1]
        args.num_classes      = int(torch.unique(dataset.ref_data.y).numel())

        self.model = models.cross_GAE(
            shared_x_dim=args.shared_input_dim,
            ref_x_dim=args.ref_input_dim,
            target_x_dim=args.target_input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            GAT_head=args.GAT_head,
            num_classes=args.num_classes
        ).to(args.device)

    def train_epoch(self, ref_loader, tgt_loader, optimizer, epoch):
        self.model.train()
        sum_loss = sum_recons = sum_cls = sum_mmd = sum_coral = 0.0
        
        ref_loader = ref_loader
        tgt_iter = cycle(tgt_loader)
        
        for idx, ref_batch in enumerate(tqdm(ref_loader, desc="Training")):
            ref_batch = ref_batch.to(self.args.device)
            tgt_batch = next(tgt_iter).to(self.args.device)

            optimizer.zero_grad()

            outputs = self.model(ref_data=ref_batch, target_data=tgt_batch)
            # unpack
            latent_embed_mmd = {
                'ref_homo':    outputs['ref_homo_latent'],
                'ref_nonhomo': outputs['ref_nonhomo_latent'],
                'target_homo':    outputs.get('target_homo_latent'),
                'target_nonhomo': outputs.get('target_nonhomo_latent')
            }
            recons = outputs['recons']
            ref_logits = outputs['ref_logits']
            


            # losses
            loss_dict = self.model.loss_function(
                ref_batch.homo_x, ref_batch.nonhomo_x,
                tgt_batch.homo_x, tgt_batch.nonhomo_x,
                recons['ref_homo'], recons['ref_nonhomo'],
                recons['tgt_homo'], recons['tgt_nonhomo'],
                ref_logits, ref_batch.y, 
                self.args.alpha,
                self.args.beta_cls, self.args.beta_mmd, self.args.beta_coral
            )

            sum_loss   += loss_dict['loss'].item()
            sum_recons += loss_dict['Reconstruction_Loss'].item()
            sum_cls    += loss_dict['Classification_Loss'].item()
            sum_mmd    += loss_dict['MMD_Loss'].item()
            sum_coral    += loss_dict['CORAL_Loss'].item()

            loss_dict['loss'].backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={sum_loss/(idx+1):.4f}, "
              f"recons={sum_recons/(idx+1)*self.args.alpha:.4f}, "
              f"cls={sum_cls/(idx+1)*self.args.beta_cls:.4f}, "
              f"mmd={sum_mmd/(idx+1)*self.args.beta_mmd:.4f}, "
              f"coral={sum_coral/(idx+1)*self.args.beta_coral:.4f}")

        torch.cuda.empty_cache()
        return loss_dict
    
    def pred(self, ref_loader, tgt_loader):
        self.model.eval()
        ref_data, tgt_data = self.dataset.ref_data, self.dataset.target_data

        # ref_loader = ClusterLoader(ClusterData(ref_data, num_parts=self.args.batch_size, recursive=False),
        #                            batch_size=1, shuffle=False)
        # tgt_loader = ClusterLoader(ClusterData(tgt_data, num_parts=self.args.batch_size, recursive=False),
        #                            batch_size=1, shuffle=False)

        all_ref_logits = []
        all_ref_y      = []
        all_ref_latent = []
        all_ref_latent_homo = []
        all_ref_recon_homo = []
        all_ref_recon_nonhomo = []

        all_tgt_logits = []
        all_tgt_y      = []
        all_tgt_latent = []
        all_tgt_latent_homo = []
        all_tgt_recon_homo = []
        all_tgt_recon_nonhomo = []

        with torch.no_grad():
            # reference
            for batch in tqdm(ref_loader, desc="Predict Ref"):
                batch = batch.to(self.args.device)
                out = self.model(ref_data=batch)
                # collect
                all_ref_logits.append(out['ref_logits'].cpu())
                all_ref_y.append(batch.y.cpu())
                latent = torch.cat([out['ref_homo_latent'], out['ref_nonhomo_latent']], dim=1)
                all_ref_latent.append(latent.cpu())
                all_ref_latent_homo.append(out['ref_homo_latent'].cpu())
            # target
            for batch in tqdm(tgt_loader, desc="Predict Tgt"):
                batch = batch.to(self.args.device)
                out = self.model(target_data=batch)
                all_tgt_logits.append(out['target_logits'].cpu())
                all_tgt_y.append(batch.y.cpu())
                latent = torch.cat([out['target_homo_latent'], out['target_nonhomo_latent']], dim=1)
                all_tgt_latent.append(latent.cpu())
                all_tgt_latent_homo.append(out['target_homo_latent'].cpu())

        # stack
        logits_ref   = torch.cat(all_ref_logits)
        y_ref        = torch.cat(all_ref_y)
        latent_ref   = torch.cat(all_ref_latent)
        latent_ref_homo   = torch.cat(all_ref_latent_homo)

        logits_tgt   = torch.cat(all_tgt_logits)
        y_tgt        = torch.cat(all_tgt_y)
        latent_tgt   = torch.cat(all_tgt_latent)
        latent_tgt_homo   = torch.cat(all_tgt_latent_homo)

        # save reconstructions
        save_dir = self.args.savedir
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'ref_latent_homo.npy'), latent_ref_homo)
        np.save(os.path.join(save_dir, 'tgt_latent_homo.npy'), latent_tgt_homo)

        # predictions
        preds_ref = torch.argmax(F.softmax(logits_ref, dim=1), dim=1)
        preds_tgt = torch.argmax(F.softmax(logits_tgt, dim=1), dim=1)

        acc_ref = (preds_ref == y_ref).float().mean()
        acc_tgt = (preds_tgt == y_tgt).float().mean()
        print(f"Ref Acc: {acc_ref:.4f}, Tgt Acc: {acc_tgt:.4f}")

        return preds_ref, y_ref, latent_ref, preds_tgt, y_tgt, latent_tgt

    def train(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.lr,
                               weight_decay=self.args.wd)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        # initialize history
        # losses = {'total':[], 'recons':[], 'cls':[], 'mmd':[]}
        overall_losses = {}
        acc_ref_list = []
        acc_tgt_list = []

        ref_data, tgt_data = self.dataset.ref_data, self.dataset.target_data
        ref_loader = ClusterLoader(ClusterData(ref_data,
                               num_parts=self.args.batch_size,
                               recursive=False),
                                   batch_size=1, shuffle=True)
        tgt_loader = ClusterLoader(ClusterData(tgt_data,
                               num_parts=self.args.batch_size,
                               recursive=False),
                                   batch_size=1, shuffle=True)

        for ep in range(self.args.epochs):
            # one epoch of training
            ld = self.train_epoch(ref_loader, tgt_loader, optimizer, ep)

            # record losses
            # losses['total'].append(ld['loss'].item())
            # losses['recons'].append(ld['Reconstruction_Loss'].item())
            # losses['cls'].append(ld['Classification_Loss'].item())
            # losses['mmd'].append(ld['MMD_Loss'].item())
            for loss_name, loss_value in ld.items():
                # If this loss term is already in the dictionary, append the value
                if loss_name not in overall_losses:
                    overall_losses[loss_name] = []
                overall_losses[loss_name].append(loss_value.item())

            # run prediction & compute accuracies
            preds_ref, y_ref, _, preds_tgt, y_tgt, _, _, _, _, _, _, _ = self.pred(ref_loader, tgt_loader)
            acc_ref = (preds_ref == y_ref).float().mean().item()
            acc_tgt = (preds_tgt == y_tgt).float().mean().item()
            acc_ref_list.append(acc_ref)
            acc_tgt_list.append(acc_tgt)

        # make sure output directory for plots exists
        plot_dir = os.path.join(self.args.savedir, 'plot')
        os.makedirs(plot_dir, exist_ok=True)

        # plot losses
        plot_separate_loss_curves(
            overall_losses,
            plot_dir
        )

        # plot accuracies
        plot_acc_curves(
            acc_ref_list,
            acc_tgt_list,
            plot_dir
        )

        return overall_losses

class cross_GAE_VAE(cross_GAE):
    def __init__(
        self,
        args,
        dataset
    ):
        super().__init__(args, dataset)
        self.model = models.cross_GAE_VAE(
            shared_x_dim=args.shared_input_dim,
            ref_x_dim=args.ref_input_dim,
            target_x_dim=args.target_input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            GAT_head=args.GAT_head,
            num_classes=args.num_classes,
            condition=args.condition,
            denoise=args.denoise
        ).to(args.device)

    def train_epoch(self, ref_loader, tgt_loader, optimizer, epoch):
        self.model.train()
        sum_loss = sum_recons = sum_cls = sum_mmd = sum_coral = sum_kl =  0.0

        ref_loader = ref_loader
        tgt_iter = cycle(tgt_loader)

        for idx, ref_batch in enumerate(tqdm(ref_loader, desc="Training")):
            ref_batch = ref_batch.to(self.args.device)
            tgt_batch = next(tgt_iter).to(self.args.device)

            optimizer.zero_grad()

            outputs = self.model(ref_data=ref_batch, target_data=tgt_batch)
            recons = outputs['recons']
            ref_logits = outputs['ref_logits']
            tgt_logits = outputs['target_logits']

            # losses
            loss_dict = self.model.loss_function(
                ref_batch.homo_x, ref_batch.nonhomo_x,
                tgt_batch.homo_x, tgt_batch.nonhomo_x,
                outputs['latent'], ref_logits, ref_batch.y,
                self.args.alpha,
                self.args.beta_cls, self.args.beta_mmd, self.args.beta_coral, self.args.beta_kl,
                tgt_logits, self.args.condition
            )

            sum_loss   += loss_dict['loss'].item()
            sum_recons += loss_dict['Reconstruction_Loss'].item()
            sum_cls    += loss_dict['Classification_Loss'].item()
            sum_mmd    += loss_dict['MMD_Loss'].item()
            sum_coral  += loss_dict['CORAL_Loss'].item()
            sum_kl  += loss_dict['KL_Loss'].item()

            loss_dict['loss'].backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={sum_loss/(idx+1):.4f}, "
            f"recons={sum_recons/(idx+1)*self.args.alpha:.4f}, "
            f"cls={sum_cls/(idx+1)*self.args.beta_cls:.4f}, "
            f"mmd={sum_mmd/(idx+1)*self.args.beta_mmd:.4f}, "
            f"coral={sum_coral/(idx+1)*self.args.beta_coral:.4f}, "
            f"kl={sum_kl/(idx+1)*self.args.beta_kl:.4f}")
        
        torch.cuda.empty_cache()
        return loss_dict

    def pred(self, ref_loader, tgt_loader):
        self.model.eval()
        ref_data, tgt_data = self.dataset.ref_data, self.dataset.target_data

        # ref_loader = ClusterLoader(ClusterData(ref_data, num_parts=self.args.batch_size, recursive=False),
        #                            batch_size=1, shuffle=False)
        # tgt_loader = ClusterLoader(ClusterData(tgt_data, num_parts=self.args.batch_size, recursive=False),
        #                            batch_size=1, shuffle=False)

        all_ref_logits = []
        all_ref_y      = []
        all_ref_latent = []
        all_ref_latent_homo = []
        all_ref_recon_homo = []
        all_ref_recon_nonhomo = []
        all_ref_mean_homo = []
        all_ref_mean_nonhomo = []

        all_tgt_logits = []
        all_tgt_y      = []
        all_tgt_latent = []
        all_tgt_latent_homo = []
        all_tgt_recon_homo = []
        all_tgt_recon_nonhomo = []
        all_tgt_mean_homo = []
        all_tgt_mean_nonhomo = []

        with torch.no_grad():
            # reference
            for batch in tqdm(ref_loader, desc="Predict Ref"):
                batch = batch.to(self.args.device)
                out = self.model(ref_data=batch)
                # collect
                all_ref_logits.append(out['ref_logits'].cpu())
                all_ref_y.append(batch.y.cpu())
                latent = torch.cat([out['latent']['ref_homo_latent'], out['latent']['ref_nonhomo_latent']], dim=1)
                all_ref_latent.append(latent.cpu())
                all_ref_latent_homo.append(out['latent']['ref_homo_latent'].cpu())

                all_ref_mean_homo.append(out['latent']['ref_homo_mean'].cpu())
                all_ref_mean_nonhomo.append(out['latent']['ref_nonhomo_mean'].cpu())
                

            # target
            for batch in tqdm(tgt_loader, desc="Predict Tgt"):
                batch = batch.to(self.args.device)
                out = self.model(target_data=batch)
                all_tgt_logits.append(out['target_logits'].cpu())
                all_tgt_y.append(batch.y.cpu())
                latent = torch.cat([out['latent']['target_homo_latent'], out['latent']['target_nonhomo_latent']], dim=1)
                all_tgt_latent.append(latent.cpu())
                all_tgt_latent_homo.append(out['latent']['target_homo_latent'].cpu())

                all_tgt_mean_homo.append(out['latent']['target_homo_mean'].cpu())
                all_tgt_mean_nonhomo.append(out['latent']['target_nonhomo_mean'].cpu())

        # stack
        logits_ref   = torch.cat(all_ref_logits)
        y_ref        = torch.cat(all_ref_y)
        latent_ref   = torch.cat(all_ref_latent)
        latent_ref_homo   = torch.cat(all_ref_latent_homo)
        ref_homo_mean   = torch.cat(all_ref_mean_homo)
        ref_nonhomo_mean   = torch.cat(all_ref_mean_nonhomo)

        logits_tgt   = torch.cat(all_tgt_logits)
        y_tgt        = torch.cat(all_tgt_y)
        latent_tgt   = torch.cat(all_tgt_latent)
        latent_tgt_homo   = torch.cat(all_tgt_latent_homo)
        tgt_homo_mean   = torch.cat(all_tgt_mean_homo)
        tgt_nonhomo_mean   = torch.cat(all_tgt_mean_nonhomo)
        

        # predictions
        preds_ref = torch.argmax(F.softmax(logits_ref, dim=1), dim=1)
        preds_tgt = torch.argmax(F.softmax(logits_tgt, dim=1), dim=1)

        acc_ref = (preds_ref == y_ref).float().mean()
        acc_tgt = (preds_tgt == y_tgt).float().mean()
        print(f"Ref Acc: {acc_ref:.4f}, Tgt Acc: {acc_tgt:.4f}")

        return preds_ref, y_ref, latent_ref, preds_tgt, y_tgt, latent_tgt, latent_ref_homo, latent_tgt_homo, ref_homo_mean, ref_nonhomo_mean, tgt_homo_mean, tgt_nonhomo_mean