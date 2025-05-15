import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .gat_conv import GATConv
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .loss import *
class MLP(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128, dropout=0.2):
        super(MLP, self).__init__()
        self.latent_dim = latent_dim
        self.Sh = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.Drop = nn.Dropout(p=dropout)
        self.Predictor = nn.Linear(hidden_dim, output_dim)
        # weight inits
        nn.init.kaiming_normal_(self.Sh.weight, mode='fan_in')
        nn.init.constant_(self.Sh.bias, 0)
        nn.init.constant_(self.Predictor.bias, 0)

    def forward(self, x):
        x = self.bn1(self.Sh(x))
        x = self.Drop(x)
        x = F.selu(x)
        yhat = self.Predictor(x)
        return yhat

class GATBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, heads=1):
        super().__init__()
        # Encoder layers
        self.encoder_gat1 = GATConv(in_dim, hidden_dim, heads=heads,
                                    concat=False, dropout=0.1,
                                    add_self_loops=False, bias=False)
        self.encoder_gat2 = GATConv(hidden_dim, latent_dim, heads=heads,
                                    concat=False, dropout=0,
                                    add_self_loops=False, bias=False)
        # Decoder layers
        self.decoder_gat1 = GATConv(latent_dim, hidden_dim, heads=heads,
                                    concat=False, dropout=0,
                                    add_self_loops=False, bias=False)
        self.decoder_gat2 = GATConv(hidden_dim, in_dim, heads=heads,
                                    concat=False, dropout=0.1,
                                    add_self_loops=False, bias=False)

    def encode(self, x, edge_index):
        h1 = F.elu(self.encoder_gat1(x, edge_index))
        latent = self.encoder_gat2(h1, edge_index, attention=False)
        return latent

    def decode(self, latent, edge_index):
        # Tie decoder weights to encoder
        self.decoder_gat1.lin_src.data = self.encoder_gat2.lin_src.transpose(0, 1)
        self.decoder_gat1.lin_dst.data = self.encoder_gat2.lin_dst.transpose(0, 1)
        self.decoder_gat2.lin_src.data = self.encoder_gat1.lin_src.transpose(0, 1)
        self.decoder_gat2.lin_dst.data = self.encoder_gat1.lin_dst.transpose(0, 1)

        h3 = F.relu(self.decoder_gat1(latent, edge_index,
                                     attention=True,
                                     tied_attention=self.encoder_gat1.attentions))
        recon = self.decoder_gat2(h3, edge_index, attention=False)
        return recon

    def forward(self, x, edge_index):
        latent = self.encode(x, edge_index)
        recon = self.decode(latent, edge_index)
        return latent, recon


class cross_GAE(nn.Module):
    """
    Cross-domain graph autoencoder using TBlock for shared and non-homogeneous parts,
    with proper reconstruction from combined latent.
    """
    def __init__(
        self,
        shared_x_dim: int,
        ref_x_dim: int,
        target_x_dim: int,
        hidden_dim: int,
        latent_dim: int,
        GAT_head: int = 1,
        num_classes: int = None,
        noise_std: float = 0.1
    ):
        super().__init__()
        self.noise_std = noise_std
        # Autoencoder blocks
        self.shared_block = GATBlock(shared_x_dim + 2, hidden_dim, latent_dim, heads=GAT_head)
        self.ref_block    = GATBlock(ref_x_dim,    hidden_dim, latent_dim, heads=GAT_head)
        self.target_block = GATBlock(target_x_dim, hidden_dim, latent_dim, heads=GAT_head)

        # FC layers to map concatenated latents back to block latent space
        self.ref_fc    = nn.Linear(2*latent_dim, latent_dim)
        self.target_fc = nn.Linear(2*latent_dim, latent_dim)

        # Classifier on concatenated latents
        # self.classifier = nn.Linear(2 * latent_dim, num_classes)
        self.classifier = MLP(
            latent_dim= 2*latent_dim,
            output_dim=num_classes,
        )


    def forward(self, ref_data=None, target_data=None):
        outputs = {}

        # Reference branch
        if ref_data is not None:
            # encode both parts
            rh = ref_data.homo_x
            rnh = ref_data.nonhomo_x
            num_nodes = rh.size(0)
            one_hot_ref = torch.tensor([1, 0], device=rh.device)
            one_hot_ref = one_hot_ref.unsqueeze(0).repeat(num_nodes, 1)

            # concatenate one-hot
            rh_input = torch.cat([rh, one_hot_ref], dim=1)

            # add Gaussian noise
            rh_noisy  = rh_input  + torch.randn_like(rh_input)  * self.noise_std
            rnh_noisy = rnh + torch.randn_like(rnh) * self.noise_std


            rh_latent = self.shared_block.encode(rh_noisy, ref_data.edge_index)
            rnh_latent = self.ref_block.encode(rnh_noisy, ref_data.edge_index)
            # concat for classification & reconstruction
            ref_comb = torch.cat([rh_latent, rnh_latent], dim=1)
            # classification
            outputs['ref_logits'] = self.classifier(ref_comb)
          

            # reconstruct
            # ref_comb = self.ref_fc(ref_comb)
            rh_latent_input = torch.cat([rh_latent, one_hot_ref], dim=1)
            rh_recon  = self.shared_block.decode(rh_latent, ref_data.edge_index)
            rnh_recon = self.ref_block.decode(rnh_latent, ref_data.edge_index)
            # save
            outputs['ref_homo_latent']    = rh_latent
            outputs['ref_nonhomo_latent'] = rnh_latent
        else:
            rh_recon = rnh_recon = outputs.get('ref_logits', None)

        # Target branch
        if target_data is not None:
            th = target_data.homo_x
            tnh = target_data.nonhomo_x

            num_nodes = th.size(0)
            one_hot_tgt = torch.tensor([0, 1], device=th.device)
            one_hot_tgt = one_hot_tgt.unsqueeze(0).repeat(num_nodes, 1)

            # concatenate one-hot
            th_input = torch.cat([th, one_hot_tgt], dim=1)

            th_noisy  = th_input  + torch.randn_like(th_input)  * self.noise_std
            tnh_noisy = tnh + torch.randn_like(tnh) * self.noise_std

            th_latent  = self.shared_block.encode(th_noisy,    target_data.edge_index)
            tnh_latent = self.target_block.encode(tnh_noisy, target_data.edge_index)
            tgt_comb = torch.cat([th_latent, tnh_latent], dim=1)

            outputs['target_logits'] = self.classifier(tgt_comb)

            # tgt_comb = self.ref_fc(tgt_comb)
            th_latent_input = torch.cat([th_latent, one_hot_tgt], dim=1)

            th_recon  = self.shared_block.decode(th_latent,  target_data.edge_index)
            tnh_recon = self.target_block.decode(tnh_latent, target_data.edge_index)
            outputs['target_homo_latent']    = th_latent
            outputs['target_nonhomo_latent'] = tnh_latent
        else:
            th_recon = tnh_recon = outputs.get('target_logits', None)

        outputs['recons'] = {
            'ref_homo':    rh_recon,
            'ref_nonhomo': rnh_recon,
            'tgt_homo':    th_recon,
            'tgt_nonhomo': tnh_recon,
        }
        return outputs

    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the loss function.
        """
        ### Parse Input
        # Origin
        ref_homo = args[0]
        ref_nonhomo = args[1]
        target_homo = args[2]
        target_nonhomo = args[3]
        # Recon
        recons_ref_homo = args[4]
        recons_ref_nonhomo = args[5]
        recons_target_homo = args[6]
        recons_target_nonhomo = args[7]
        # Embed
        latent_embed_mmd = args[8]
        ref_homo_latent = latent_embed_mmd['ref_homo']
        target_homo_latent = latent_embed_mmd['target_homo']
        ref_nonhomo_latent = latent_embed_mmd['ref_nonhomo']
        target_nonhomo_latent = latent_embed_mmd['target_nonhomo']
        # Pred
        ref_logits = args[9]
        ref_y = args[10]
        # Loss Weight
        alpha = args[11]
        beta_cls = args[12]
        beta_mmd = args[13]
        beta_coral = args[14]

        num_nodes = ref_homo.size(0)
        one_hot_ref = torch.tensor([1, 0], device=ref_homo.device)
        one_hot_ref = one_hot_ref.unsqueeze(0).repeat(num_nodes, 1)

        # concatenate one-hot
        rh_input = torch.cat([ref_homo, one_hot_ref], dim=1)

        num_nodes = target_homo.size(0)
        one_hot_tgt = torch.tensor([0, 1], device=target_homo.device)
        one_hot_tgt = one_hot_tgt.unsqueeze(0).repeat(num_nodes, 1)

        # concatenate one-hot
        th_input = torch.cat([target_homo, one_hot_tgt], dim=1)
        
        ### Compute Loss
        recons_loss = F.mse_loss(rh_input, recons_ref_homo) + F.mse_loss(th_input, recons_target_homo)
        + F.mse_loss(ref_nonhomo, recons_ref_nonhomo) + F.mse_loss(target_nonhomo, recons_target_nonhomo)
        
        # replace the cls loss witht the version consider the class weights
        # class_counts = torch.bincount(ref_y)
        # class_weights = 1.0 / (class_counts.float() + 1e-8)
        # class_weights = class_weights / class_weights.sum() * len(class_counts)
        # classifier_loss = F.cross_entropy(ref_logits, ref_y, weight=class_weights.to(ref_logits.device))
        # labels = ref_y.detach().cpu()
        # num_classes = ref_logits.size(1)
        # class_counts = torch.bincount(labels, minlength=num_classes).float()
        # eps = 1e-6
        # inv_counts = 1.0 / (class_counts + eps)
        # inv_counts[class_counts == 0] = 0.0  
        # inv_counts = inv_counts / inv_counts.sum() * num_classes
        # class_weights = inv_counts.to(ref_logits.device).type(ref_logits.dtype)

        # classifier_loss = F.cross_entropy(ref_logits, ref_y, weight=class_weights)
        classifier_loss = F.cross_entropy(ref_logits, ref_y)
        

        ref_comb = torch.cat([ref_homo_latent, ref_nonhomo_latent], dim=1)
        tgt_comb = torch.cat([target_homo_latent, target_nonhomo_latent], dim=1)

        mmd_loss = mmd(ref_comb, tgt_comb)
        coral_loss = CORAL(ref_comb, tgt_comb, ref_comb.device)

        loss = torch.mean(alpha*recons_loss + beta_cls*classifier_loss + beta_mmd*mmd_loss + beta_coral * coral_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(),  "Classification_Loss": classifier_loss.detach(), 
                "MMD_Loss": mmd_loss.detach(), "CORAL_Loss": coral_loss.detach()}


