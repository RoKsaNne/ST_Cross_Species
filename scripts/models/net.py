import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATv2Conv
from typing import List, Callable, Union, Any, TypeVar, Tuple
from .loss import *

class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return self.act(x)

class cross_GAE(nn.Module):
    def __init__(self, shared_x_dim: int, ref_x_dim: int, target_x_dim: int, latent_dim: int, shared_hidden_dims: List = None, ref_hidden_dims: List = None, target_hidden_dims: List = None, GAT_head:int=4, num_classes: int=None):
        super(cross_GAE, self).__init__()
        self.shared_x_dim = shared_x_dim 
        self.ref_x_dim = ref_x_dim 
        self.target_x_dim = target_x_dim 
        self.latent_dim = latent_dim
        self.shared_hidden_dims = shared_hidden_dims
        self.ref_hidden_dims = ref_hidden_dims
        self.target_hidden_dims = target_hidden_dims
        self.GAT_head = GAT_head
        self.num_classes = num_classes

        #print(shared_x_dim, ref_x_dim, target_x_dim, latent_dim, shared_hidden_dims) 2000 2000 2000 128 [128, 128]

        ### Encoders
        # Shared Encoder
        modules_encoder_shared = []
        shared_dim = self.shared_x_dim
        for h_dim in shared_hidden_dims:
                modules_encoder_shared.append(
                    GATBlock(shared_dim, h_dim, self.GAT_head))
                shared_dim = h_dim*self.GAT_head

        self.encoder_shared = nn.Sequential(*modules_encoder_shared)
        self.linear_shared = nn.Linear(shared_dim, self.latent_dim)

        # Reference Non-homo Encoder
        modules_encoder_ref = []
        ref_dim = self.ref_x_dim
        for h_dim in ref_hidden_dims:
                    modules_encoder_ref.append(
                        GATBlock(ref_dim, h_dim, self.GAT_head))
                    ref_dim = h_dim*self.GAT_head   

        self.encoder_ref = nn.Sequential(*modules_encoder_ref)
        self.linear_ref = nn.Linear(ref_dim, self.latent_dim)

        # Traget Non-homo Encoder
        modules_encoder_target = []
        target_dim = self.target_x_dim
        for h_dim in target_hidden_dims:
                    modules_encoder_target.append(
                        GATBlock(target_dim, h_dim, self.GAT_head))
                    target_dim = h_dim*self.GAT_head   

        self.encoder_target = nn.Sequential(*modules_encoder_target)
        self.linear_target = nn.Linear(target_dim, self.latent_dim)

        ### Shared FC Layers
        self.decoder_linear_ref = nn.Linear(2*self.latent_dim, self.latent_dim)
        self.decoder_linear_target = nn.Linear(2*self.latent_dim, self.latent_dim)

        ### Decoders
        shared_hidden_dims_rev = shared_hidden_dims[::-1]
        ref_hidden_dims_rev = ref_hidden_dims[::-1]
        target_hidden_dims_rev = target_hidden_dims[::-1]

        # Shared Decoder
        modules_decoder_shared = []
        shared_dim = self.latent_dim
        for h_dim in shared_hidden_dims_rev:
            modules_decoder_shared.append(
                GATBlock(shared_dim, h_dim, self.GAT_head)
            )
            shared_dim = h_dim * self.GAT_head

        self.decoder_shared = nn.Sequential(*modules_decoder_shared)
        self.final_layer_shared = nn.Linear(shared_dim, self.shared_x_dim)

        # Reference Non-homo Decoder
        modules_decoder_ref = []
        ref_dim = self.latent_dim
        for h_dim in ref_hidden_dims_rev:
            modules_decoder_ref.append(
                GATBlock(ref_dim, h_dim, self.GAT_head)
            )
            ref_dim = h_dim * self.GAT_head

        self.decoder_ref = nn.Sequential(*modules_decoder_ref)
        self.final_layer_ref = nn.Linear(ref_dim, self.ref_x_dim)

        # Target Non-homo Decoder
        modules_decoder_target = []
        target_dim = self.latent_dim
        for h_dim in target_hidden_dims_rev:
            modules_decoder_target.append(
                GATBlock(target_dim, h_dim, self.GAT_head)
            )
            target_dim = h_dim * self.GAT_head

        self.decoder_target = nn.Sequential(*modules_decoder_target)
        self.final_layer_target = nn.Linear(target_dim, self.target_x_dim)

        ### Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )


    def encode(self, ref_homo, target_homo, ref_nonhomo, target_nonhomo, ref_edge, target_edge):
        """
        Encodes the input by passing through the encoder network
        and returns the latent representation
        """
        ### GAT Encoder
        latent_embed_mmd = {}
        # Reference Shared Embed
        ref_homo_latent = ref_homo
        for layer in self.encoder_shared:
            ref_homo_latent = layer(ref_homo_latent, ref_edge)
        ref_homo_latent = self.linear_shared(ref_homo_latent)

        # Target Shared Embed
        target_homo_latent = target_homo
        for layer in self.encoder_shared:
            target_homo_latent = layer(target_homo_latent, target_edge)
        target_homo_latent = self.linear_shared(target_homo_latent)
        
        # Reference Nonhomo Embed
        ref_nonhomo_latent = ref_nonhomo
        for layer in self.encoder_ref:
            ref_nonhomo_latent = layer(ref_nonhomo_latent, ref_edge)
        ref_nonhomo_latent = self.linear_ref(ref_nonhomo_latent)
        
        # Target Nonhomo Embed
        target_nonhomo_latent = target_nonhomo
        for layer in self.encoder_target:
            target_nonhomo_latent = layer(target_nonhomo_latent, target_edge)
        target_nonhomo_latent = self.linear_target(target_nonhomo_latent)

        # Latent Embedding Dict
        latent_embed_mmd['ref_homo'] = ref_homo_latent
        latent_embed_mmd['target_homo'] = target_homo_latent
        latent_embed_mmd['ref_nonhomo'] = ref_nonhomo_latent
        latent_embed_mmd['target_nonhomo'] = target_nonhomo_latent

        # Embedding Concatnate
        ref_latent = torch.cat([ref_homo_latent, ref_nonhomo_latent], dim=1)
        target_latent = torch.cat([target_homo_latent, target_nonhomo_latent], dim=1)

        ### Shared FC Layer
        #TODO: Test FC layer weight sharing
        # ref_latent = self.decoder_linear_ref(ref_latent)
        # target_latent = self.decoder_linear_target(target_latent)
        ref_latent = self.decoder_linear_ref(ref_latent)
        target_latent = self.decoder_linear_ref(target_latent)

        return latent_embed_mmd, ref_latent, target_latent
    
    def decode_recons(self, ref_latent, target_latent, ref_edge, target_edge):
        """
        Maps the given latent codes
        onto the sequence space.
        """
        ### GAT Decoder
        # Reference Shared Embed
        ref_homo = ref_latent
        for layer in self.decoder_shared:
            ref_homo = layer(ref_homo, ref_edge)
        ref_homo = self.final_layer_shared(ref_homo)

        # Target Shared Embed
        target_homo = target_latent
        for layer in self.decoder_shared:
            target_homo = layer(target_homo, target_edge)
        target_homo = self.final_layer_shared(target_homo)

        # Reference Nonhomo Embed
        ref_nonhomo = ref_latent
        for layer in self.decoder_ref:
            ref_nonhomo = layer(ref_nonhomo, ref_edge)
        ref_nonhomo = self.final_layer_ref(ref_nonhomo)

        # Target Nonhomo Embed
        target_nonhomo = target_latent
        for layer in self.decoder_target:
            target_nonhomo = layer(target_nonhomo, target_edge)
        target_nonhomo = self.final_layer_target(target_nonhomo)

        return ref_homo, ref_nonhomo, target_homo, target_nonhomo

    # def forward(self, ref_data, target_data):
    #     ref_homo, ref_nonhomo, ref_edge = ref_data.homo_x, ref_data.nonhomo_x, ref_data.edge_index
    #     target_homo, target_nonhomo, target_edge = target_data.homo_x, target_data.nonhomo_x, target_data.edge_index

    #     latent_embed_mmd, ref_latent, target_latent = self.encode(ref_homo, target_homo, ref_nonhomo, target_nonhomo, ref_edge, target_edge)
    #     recons_ref_homo, recons_ref_nonhomo, recons_target_homo, recons_target_nonhomo = self.decode_recons(ref_latent, target_latent, ref_edge, target_edge)
    #     ref_logits = self.classifier(ref_latent)
    #     target_logits = self.classifier(target_latent)

    #     # not needed for now, but maybe later
    #     recons_ref = torch.cat([recons_ref_homo, recons_ref_nonhomo], dim=1)
    #     recons_target = torch.cat([recons_target_homo, recons_target_nonhomo], dim=1)
        
    #     return  [latent_embed_mmd, ref_latent, target_latent, recons_ref_homo, recons_ref_nonhomo, recons_target_homo, recons_target_nonhomo, ref_logits, target_logits]
    def forward(self, ref_data=None, target_data=None):
        latent_embed_mmd = {}
        ref_latent = target_latent = None
        recons_ref_homo = recons_ref_nonhomo = None
        recons_target_homo = recons_target_nonhomo = None
        ref_logits = target_logits = None

        if ref_data is not None:
            ref_latent, (ref_homo_latent, ref_nonhomo_latent) = self._encode_side(
                ref_data.homo_x, ref_data.nonhomo_x, ref_data.edge_index,
                self.encoder_shared, self.linear_shared,
                self.encoder_ref, self.linear_ref
            )
            latent_embed_mmd['ref_homo'] = ref_homo_latent
            latent_embed_mmd['ref_nonhomo'] = ref_nonhomo_latent

            recons_ref_homo, recons_ref_nonhomo = self._decode_side(
                ref_latent, ref_data.edge_index,
                self.decoder_shared, self.final_layer_shared,
                self.decoder_ref, self.final_layer_ref
            )

            ref_logits = self._classify_side(ref_latent)

        if target_data is not None:
            target_latent, (target_homo_latent, target_nonhomo_latent) = self._encode_side(
                target_data.homo_x, target_data.nonhomo_x, target_data.edge_index,
                self.encoder_shared, self.linear_shared,
                self.encoder_target, self.linear_target
            )
            latent_embed_mmd['target_homo'] = target_homo_latent
            latent_embed_mmd['target_nonhomo'] = target_nonhomo_latent

            recons_target_homo, recons_target_nonhomo = self._decode_side(
                target_latent, target_data.edge_index,
                self.decoder_shared, self.final_layer_shared,
                self.decoder_target, self.final_layer_target
            )

            target_logits = self._classify_side(target_latent)

        return [
            latent_embed_mmd,
            ref_latent,
            target_latent,
            recons_ref_homo,
            recons_ref_nonhomo,
            recons_target_homo,
            recons_target_nonhomo,
            ref_logits,
            target_logits
        ]
        
    def _encode_side(self, homo_x, nonhomo_x, edge_index, shared_encoder, shared_linear, nonhomo_encoder, nonhomo_linear):
        # Encode shared part
        shared = homo_x
        for layer in shared_encoder:
            shared = layer(shared, edge_index)
        shared_latent = shared_linear(shared)

        # Encode non-homogeneous part
        nonhomo = nonhomo_x
        for layer in nonhomo_encoder:
            nonhomo = layer(nonhomo, edge_index)
        nonhomo_latent = nonhomo_linear(nonhomo)

        # Concat and FC
        concat_latent = torch.cat([shared_latent, nonhomo_latent], dim=1)
        final_latent = self.decoder_linear_ref(concat_latent)

        return final_latent, (shared_latent, nonhomo_latent)

    def _decode_side(self, latent, edge_index, shared_decoder, shared_final, nonhomo_decoder, nonhomo_final):
        # Shared decode
        shared = latent
        for layer in shared_decoder:
            shared = layer(shared, edge_index)
        shared_out = shared_final(shared)

        # Non-homogeneous decode
        nonhomo = latent
        for layer in nonhomo_decoder:
            nonhomo = layer(nonhomo, edge_index)
        nonhomo_out = nonhomo_final(nonhomo)

        return shared_out, nonhomo_out

    def _classify_side(self, latent):
        return self.classifier(latent)

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
        beta_cls = args[11]
        beta_mmd = args[12]
        
        ### Compute Loss
        recons_loss = F.mse_loss(ref_homo, recons_ref_homo) + F.mse_loss(ref_nonhomo, recons_ref_nonhomo)+ F.mse_loss(target_homo, recons_target_homo) + F.mse_loss(target_nonhomo, recons_target_nonhomo)
        classifier_loss = F.cross_entropy(ref_logits, ref_y)
        # TODO: MMD loss for ref and target may also be hyper parameters
        mmd_loss = mmd(ref_homo_latent, ref_nonhomo_latent) + mmd(target_homo_latent, target_nonhomo_latent)

        loss = torch.mean(recons_loss + beta_cls*classifier_loss + beta_mmd*mmd_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(),  "Classification_Loss": classifier_loss.detach(), "MMD_Loss": mmd_loss.detach()}


