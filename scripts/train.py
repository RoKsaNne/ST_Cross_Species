import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import ClusterData, ClusterLoader
from tqdm import tqdm
import os

class cross_GAE:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        args.shared_input_dim = dataset.ref_data.homo_x.shape[-1]
        args.ref_input_dim = dataset.ref_data.nonhomo_x.shape[-1]
        args.target_input_dim = dataset.target_data.nonhomo_x.shape[-1]
        args.num_classes = len(set(dataset.ref_data.y))
        self.model = models.cross_GAE(shared_x_dim=args.shared_input_dim, ref_x_dim=args.ref_input_dim, target_x_dim=args.target_input_dim, 
                                      latent_dim = args.latent_dim, shared_hidden_dims = args.shared_hidden_dims, ref_hidden_dims = args.ref_hidden_dims, 
                                       target_hidden_dims = args.target_hidden_dims, GAT_head=args.GAT_head, num_classes=args.num_classes)
        self.model = self.model.to(self.args.device)

    def train_epoch(self, args, model, device, dataset, optimizer, epoch):
        model.train()
        sum_loss, sum_loss_recons, sum_loss_cls, sum_loss_mmd = 0.0, 0.0, 0.0, 0.0
        ref_data, target_data = dataset.ref_data, dataset.target_data

        ref_cluster_data = ClusterData(ref_data, num_parts=self.args.batch_size, recursive=False)
        ref_loader = ClusterLoader(ref_cluster_data, batch_size=1, shuffle=True, num_workers=0)

        target_cluster_data = ClusterData(target_data, num_parts=self.args.batch_size, recursive=False)
        target_loader = ClusterLoader(target_cluster_data, batch_size=1, shuffle=True, num_workers=0)

        target_loader_iter = cycle(target_loader)
        
        for batch_idx, ref_batch in enumerate(tqdm(ref_loader, desc="Training batches")):
            ref_batch = ref_batch.to(device)
            target_batch = next(target_loader_iter)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            latent_embed_mmd, _, _, recons_ref_homo, recons_ref_nonhomo, recons_target_homo, recons_target_nonhomo, ref_logits, _ = model(ref_batch, target_batch)

            loss_dict = model.loss_function(
                ref_batch.homo_x, ref_batch.nonhomo_x, target_batch.homo_x, target_batch.nonhomo_x,
                recons_ref_homo, recons_ref_nonhomo, recons_target_homo, recons_target_nonhomo,
                latent_embed_mmd, ref_logits, ref_batch.y, args.beta_cls, args.beta_mmd
            )

            sum_loss += loss_dict['loss']
            sum_loss_recons += loss_dict['Reconstruction_Loss']
            sum_loss_cls += loss_dict['Classification_Loss']
            sum_loss_mmd += loss_dict['MMD_Loss']

            loss_dict['loss'].backward()
            optimizer.step()
        print(f'Epoch {epoch}:')
        print('loss: {:.6f}'.format(sum_loss / (batch_idx + 1)))
        print('Reconstruction loss: {:.6f}'.format(sum_loss_recons / (batch_idx + 1)))
        print('Similarity loss: {:.6f}'.format(sum_loss_cls / (batch_idx + 1)*args.beta_cls))
        print('MMD loss: {:.6f}'.format(sum_loss_cls / (batch_idx + 1)*args.beta_mmd))

        del latent_embed_mmd, recons_ref_homo, recons_ref_nonhomo, recons_target_homo, recons_target_nonhomo, ref_logits
        torch.cuda.empty_cache()
        return loss_dict
    
    def pred(self, dataset):
        self.model.eval()
        all_outputs_ref = []
        all_gt_ref = []
        all_latent_ref = []
        all_outputs_target = []
        all_gt_target = []
        all_latent_target = []

        ref_data, target_data = dataset.ref_data, dataset.target_data
        ref_cluster_data = ClusterData(ref_data, num_parts=self.args.batch_size, recursive=False)
        ref_loader = ClusterLoader(ref_cluster_data, batch_size=1, shuffle=True, num_workers=0)

        target_cluster_data = ClusterData(target_data, num_parts=self.args.batch_size, recursive=False)
        target_loader = ClusterLoader(target_cluster_data, batch_size=1, shuffle=True, num_workers=0)
        target_loader_iter = cycle(target_loader)

        with torch.no_grad():
            for batch_idx, ref_batch in enumerate(ref_loader):
                ref_batch = ref_batch.to(self.args.device)
                target_batch = next(target_loader_iter)
                target_batch = target_batch.to(self.args.device)

                # Forward pass
                ref_latent, target_latent, _, _, _, _, ref_logits, target_logits = self.model(ref_batch, target_batch)
                
                # Save logits and label features
                all_outputs_ref.append(ref_logits.cpu())
                all_gt_ref.append(ref_batch.y)
                all_latent_ref.append(ref_latent)
                all_outputs_target.append(target_logits.cpu())
                all_gt_target.append(target_batch.y)
                all_latent_target.append(target_latent)

        logits_ref = torch.cat(all_outputs_ref, dim=0)
        gt_ref = torch.cat(all_gt_ref, dim=0)
        latent_ref = torch.cat(all_latent_ref, dim=0)
        logits_target = torch.cat(all_outputs_target, dim=0)
        gt_target = torch.cat(all_gt_target, dim=0)
        latent_target = torch.cat(all_latent_target, dim=0)

        pred_labels_ref = torch.argmax(logits_ref, dim=1)
        pred_labels_target = torch.argmax(logits_target, dim=1)

        return [pred_labels_ref, gt_ref, latent_ref, pred_labels_target, gt_target,latent_target]

    def train(self):
        total_losses = []
        recons_losses = []
        cls_losses = []
        mmd_losses = []
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        for epoch in range(self.args.epochs):
            loss_dict = self.train_epoch(self.args, self.model, self.args.device, self.dataset, optimizer, epoch)
            total_losses.append(loss_dict['loss'].item())
            recons_losses.append(loss_dict['Reconstruction_Loss'].item())
            cls_losses.append(loss_dict['Classification_Loss'].item())
            mmd_losses.append(loss_dict['MMD_Loss'].item())

        plot_dir = self.args.savedir + '/plot'
        os.makedirs(plot_dir, exist_ok=True)
        plot_separate_loss_curves(cls_losses, recons_losses, mmd_losses, total_losses, saplot_dire_dir)

        
        pred_labels_ref, ref_gt, latent_ref, pred_labels_target, target_gt, latent_target = self.pred(self.dataset)

        correct_predictions_train = (pred_labels_ref == ref_gt).float()
        correct_predictions_test = (pred_labels_target == target_gt).float()

        train_accuracy = correct_predictions_train.mean()
        test_accuracy = correct_predictions_test.mean()
        
        model_cpu = self.model.to('cpu')
        
        return [pred_labels_ref, ref_gt, train_accuracy, latent_ref, pred_labels_target, target_gt, test_accuracy, latent_target, total_losses, recons_losses, cls_losses, model_cpu]