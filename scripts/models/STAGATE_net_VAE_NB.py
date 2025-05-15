import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .gat_conv import GATConv
from torch.distributions import NegativeBinomial
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
        self.latent_dim = latent_dim
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
    
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.log_var_layer = nn.Linear(latent_dim, latent_dim)
        self.log_theta = nn.Parameter(torch.zeros(in_dim))

        # Tie decoder weights to encoder
        self.decoder_gat1.lin_src.data = self.encoder_gat2.lin_src.transpose(0, 1)
        self.decoder_gat1.lin_dst.data = self.encoder_gat2.lin_dst.transpose(0, 1)
        self.decoder_gat2.lin_src.data = self.encoder_gat1.lin_src.transpose(0, 1)
        self.decoder_gat2.lin_dst.data = self.encoder_gat1.lin_dst.transpose(0, 1)

        nn.init.kaiming_normal_(self.mean_layer.weight, mode='fan_in')
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.kaiming_normal_(self.log_var_layer.weight, mode='fan_in')
        nn.init.constant_(self.log_var_layer.bias, -5)   # start with small variance


    def encode(self, x, edge_index):
        h1 = F.elu(self.encoder_gat1(x, edge_index))
        hidden = self.encoder_gat2(h1, edge_index, attention=False)

        mean = self.mean_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mean, log_var

    def decode(self, latent, edge_index):

        h3 = F.elu(self.decoder_gat1(latent, edge_index,
                                     attention=True,
                                     tied_attention=self.encoder_gat1.attentions))
        recon = self.decoder_gat2(h3, edge_index, attention=False)
        mu = F.softplus(recon)
        return mu
    
    def reparameterize(self, mean, log_var):
        var = (0.5 * log_var).exp()
        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, edge_index):
        mean, log_var = self.encode(x, edge_index)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        z = self.reparameterize(mean, log_var)
        mu = self.decode(z, edge_index)
        # Build NB distribution: θ = exp(log_theta)
        theta = torch.exp(self.log_theta)
        # For torch.distributions.NegativeBinomial, we need `total_count`=θ, `probs`=θ/(θ+μ)
        # probs = theta.unsqueeze(0) / (theta.unsqueeze(0) + mu + 1e-8)
        # px_dist = NegativeBinomial(total_count=theta, probs=probs)

        return mean, log_var, mu, theta, z

class cross_GAE_VAE(nn.Module):
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
        condition = True,
        denoise = True
    ):
        super().__init__()

        self.condition = condition
        self.denoise = denoise
        if self.denoise:
            self.noise_std = 0.1

        # Autoencoder blocks
        if self.condition:
            self.shared_block = GATBlock(shared_x_dim + 1, hidden_dim, latent_dim, heads=GAT_head)
        else:
            self.shared_block = GATBlock(shared_x_dim, hidden_dim, latent_dim, heads=GAT_head)
        self.ref_block    = GATBlock(ref_x_dim,    hidden_dim, latent_dim, heads=GAT_head)
        self.target_block = GATBlock(target_x_dim, hidden_dim, latent_dim, heads=GAT_head)

        # Classifier on concatenated latents
        # self.classifier = nn.Linear(2 * latent_dim, num_classes)
        self.classifier = MLP(
            latent_dim= 2*latent_dim,
            output_dim=num_classes,
        )

    def forward(self, ref_data=None, target_data=None):
        outputs = {'latent': {}, 'recons': {}}

        # Reference branch
        if ref_data is not None:
            # encode both parts
            rh = ref_data.homo_x
            rnh = ref_data.nonhomo_x

            if self.condition:
                N = ref_data.homo_x.size(0)
                one_hot_ref = torch.ones(N, 1, device=rh.device)
                rh_input = torch.cat([rh, one_hot_ref], dim=1)
            else:
                rh_input = rh
            
            # add Gaussian noise
            if self.denoise:
                rh_input  = rh_input  + torch.randn_like(rh_input)  * self.noise_std
                rnh = rnh + torch.randn_like(rnh) * self.noise_std
            else:
                rh_input = rh_input
                rnh = rnh

            # Encode
            rh_mean, rh_log_var, rh_mu, rh_theta, rh_z = self.shared_block(rh_input, ref_data.edge_index)
            rnh_mean, rnh_log_var, rnh_mu, rnh_theta, rnh_z = self.ref_block(rnh, ref_data.edge_index)

            # concat for classification & reconstruction
            ref_comb = torch.cat([rh_mean, rnh_mean], dim=1)
            # classification
            outputs['ref_logits'] = self.classifier(ref_comb)

            # save
            outputs['latent'].update({
                'ref_homo_mean':        rh_mean,
                'ref_homo_log_var':     rh_log_var,
                'ref_homo_mu':          rh_mu,
                'ref_homo_theta':       rh_theta,
                'ref_homo_latent':      rh_z,

                'ref_nonhomo_mean':     rnh_mean,
                'ref_nonhomo_log_var':  rnh_log_var,
                'ref_nonhomo_mu':       rnh_mu,
                'ref_nonhomo_theta':    rnh_theta,
                'ref_nonhomo_latent':   rnh_z
            })
        else:
            rh_recon = rnh_recon = outputs.get('ref_logits', None)

        # Target branch
        if target_data is not None:
            th = target_data.homo_x
            tnh = target_data.nonhomo_x

            if self.condition:
                N = th.size(0)
                one_hot_tgt = torch.zeros(N, 1, device=th.device)
                th_input = torch.cat([th, one_hot_tgt], dim=1)
            else:
                th_input = th

            if self.denoise:
                th_input  = th_input  + torch.randn_like(th_input)  * self.noise_std
                tnh = tnh + torch.randn_like(tnh) * self.noise_std
            else:
                th_input = th_input
                tnh = tnh

            th_mean, th_log_var, th_mu, th_theta, th_z      = self.shared_block(th_input, target_data.edge_index)
            tnh_mean, tnh_log_var, tnh_mu, tnh_theta, tnh_z = self.target_block(tnh, target_data.edge_index)
            

            tgt_comb = torch.cat([th_mean, tnh_mean], dim=1)

            outputs['target_logits'] = self.classifier(tgt_comb)

            outputs['latent'].update({
                'target_homo_mean':         th_mean,
                'target_homo_log_var':      th_log_var,
                'target_homo_mu':           th_mu,
                'target_homo_theta':        th_theta,
                'target_homo_latent':       th_z,

                'target_nonhomo_mean':      tnh_mean,
                'target_nonhomo_log_var':   tnh_log_var,
                'target_nonhomo_mu':        tnh_mu,
                'target_nonhomo_theta':     tnh_theta,
                'target_nonhomo_latent':    tnh_z
            })

        else:
            th_recon = tnh_recon = outputs.get('target_logits', None)

        return outputs

    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the loss function.
        """
        ### Parse Input
        # Origin
        ref_homo        = args[0]
        ref_nonhomo     = args[1]
        target_homo     = args[2]
        target_nonhomo  = args[3]
        # Embed
        latent_dict             = args[4]
        # Pred
        ref_logits  = args[5]
        ref_y       = args[6]
        # Loss Weight
        alpha       = args[7]
        beta_cls    = args[8]
        beta_mmd    = args[9]
        beta_coral  = args[10]
        beta_kl     = args[11]

        tgt_logits  = args[12]


        # Flag
        condition   = args[13]
        if condition:
            N_ref = ref_homo.size(0)
            one_hot_ref   = torch.ones(N_ref, 1, device=ref_homo.device)
            rh_input      = torch.cat([ref_homo,   one_hot_ref], dim=1)

            N_tgt = target_homo.size(0)
            one_hot_tgt   = torch.zeros(N_tgt, 1, device=target_homo.device)
            th_input      = torch.cat([target_homo, one_hot_tgt], dim=1)

        else: 
            rh_input =ref_homo
            th_input =target_homo

        ref_homo_nll  = -log_nb_positive(
                            rh_input,
                            latent_dict['ref_homo_mu'], 
                            latent_dict['ref_homo_theta']
                        ).mean()

        tgt_homo_nll  = -log_nb_positive(
                            th_input,
                            latent_dict['target_homo_mu'], 
                            latent_dict['target_homo_theta']
                        ).mean()

        ref_nonhomo_nll = -log_nb_positive(
                            ref_nonhomo,
                            latent_dict['ref_nonhomo_mu'],
                            latent_dict['ref_nonhomo_theta']
                        ).mean()

        tgt_nonhomo_nll = -log_nb_positive(
                            target_nonhomo,
                            latent_dict['target_nonhomo_mu'],
                            latent_dict['target_nonhomo_theta']
                        ).mean()
        recons_loss = (
            ref_homo_nll
        + ref_nonhomo_nll
        + tgt_homo_nll
        + tgt_nonhomo_nll
        )
        
        # KL Divergence Loss
        def kl(m, lv):
            return -0.5 * torch.mean(1 + lv - m.pow(2) - lv.exp())
        kl_divergence = kl(latent_dict['ref_homo_mean'],    latent_dict['ref_homo_log_var'])
        kl_divergence += kl(latent_dict['target_homo_mean'], latent_dict['target_homo_log_var'])
        
        classifier_loss = F.cross_entropy(ref_logits, ref_y)

        ref_comb = torch.cat([latent_dict['ref_homo_mean'], latent_dict['ref_nonhomo_mean']], dim=1)
        tgt_comb = torch.cat([latent_dict['target_homo_mean'], latent_dict['target_nonhomo_mean']], dim=1)
        # mmd_loss = lmmd_loss(ref_comb, tgt_comb, ref_logits, tgt_logits)
        mmd_loss = mmd(ref_comb, tgt_comb)
        coral_loss = CORAL(ref_comb, tgt_comb, ref_comb.device)

        loss = torch.mean(alpha*recons_loss + beta_cls*classifier_loss + beta_mmd*mmd_loss + beta_coral * coral_loss + beta_kl * kl_divergence)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(),  "Classification_Loss": classifier_loss.detach(), 
                "MMD_Loss": mmd_loss.detach(), "CORAL_Loss": coral_loss.detach(),
                "KL_Loss": kl_divergence.detach()}


