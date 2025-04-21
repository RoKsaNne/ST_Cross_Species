
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd
import scanpy as sc
from collections import Counter
from preprocess import preprocess

def get_edge_index_standard(pos, distance_thres):
    # construct edge indexes in one region
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list

def get_edge_index_standard_region(pos, regions, distance_thres):
    # construct edge indexes when there is region information
    edge_list = []
    regions_unique = np.unique(regions)
    for reg in regions_unique:
        locs = np.where(regions == reg)[0]
        pos_region = pos[locs, :]
        dists = pairwise_distances(pos_region)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
        for (i, j) in region_edge_list:
            edge_list.append([locs[i], locs[j]])
    return edge_list

def filter_adata(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True, log1p=False)
    percent_mito = 5
    filter_mincells = 3
    filter_mingenes = 200

    adata = adata[adata.obs["n_genes_by_counts"] > filter_mingenes, :].copy()
    adata = adata[adata.obs["pct_counts_mt"] < percent_mito, :].copy()
    adata = adata[:, adata.var["n_cells_by_counts"] > filter_mincells].copy()

    return adata

def read_dataset(ref_path, target_path, species1, species2):
    species1_adata = sc.read_h5ad(ref_path)
    species2_adata = sc.read_h5ad(target_path)

    species1_adata = filter_adata(species1_adata)
    species2_adata = filter_adata(species2_adata)

    # assign the deconvolution probability to cell_type
    # TODO: Hardcode cell type columns, edit in future
    species2_cell_prob = species2_adata.obs.iloc[:,10:24]
    print(species2_cell_prob.columns)
    species2_cell_prob['cell_type'] = species2_cell_prob.idxmax(axis=1)
    species2_adata.obs['cell_type'] = species2_cell_prob['cell_type']
    species1_cell_prob = species1_adata.obs.iloc[:,10:28]
    print(species1_cell_prob.columns)
    species1_cell_prob['cell_type'] = species1_cell_prob.idxmax(axis=1)
    species1_adata.obs['cell_type'] = species1_cell_prob['cell_type']

    homo_genes_df = pd.read_csv(f'/oscar/data/yma16/Project/Cross_species/data/homolog_genes/{species1}_{species2}.tsv', sep='\t')
    return species1_adata, species2_adata, homo_genes_df

# def load_dataset(args, adata_ref, adata_target, species1, species2, homo_genes_df, distance_thres_ref, distance_thres_target, celltype_name_ref, celltype_name_target, loc_name_ref, loc_name_target, region_name_ref=None, region_name_target=None):
    
#     # n_samples = 10000
#     # sampled_indices_ref = np.random.choice(adata_ref.n_obs, n_samples, replace=False)
#     # adata_ref = adata_ref[sampled_indices_ref].copy()

#     # sampled_indices_target = np.random.choice(adata_target.n_obs, n_samples, replace=False)
#     # adata_target = adata_target[sampled_indices_target].copy()
    
#     # rename cell types to lower case to keep consistency
#     adata_ref.obs[celltype_name_ref] = adata_ref.obs[celltype_name_ref].str.lower()
#     adata_target.obs[celltype_name_target] = adata_target.obs[celltype_name_target].str.lower()

#     # extract the homologous genes
#     species1_gene_names = homo_genes_df['Gene name']
#     species2_gene_names = homo_genes_df[f'{species2} gene name']

#     valid_pairs = []
#     for species1_gene, species2_gene in zip(species1_gene_names, species2_gene_names):
#         if (species1_gene in adata_ref.var_names) and (species2_gene in adata_target.var_names):
#             valid_pairs.append((species1_gene, species2_gene))

#     species1_genes_in_adata = [pair[0] for pair in valid_pairs]
#     species2_genes_in_adata = [pair[1] for pair in valid_pairs]

#     print(f"Number of homologous pairs found: {len(valid_pairs)}")
#     print(f"Number of {species1} genes found: {len(species1_genes_in_adata)}")
#     print(f"Number of {species2} genes found: {len(species2_genes_in_adata)}")

#     species1_counter = Counter(species1_genes_in_adata)
#     species1_total_duplicates = sum(count - 1 for count in species1_counter.values() if count > 1)
#     print(f"Total number of duplicated entries in species1 ({species1}): {species1_total_duplicates}")

#     species2_counter = Counter(species2_genes_in_adata)
#     species2_total_duplicates = sum(count - 1 for count in species2_counter.values() if count > 1)
#     print(f"Total number of duplicated entries in species2 ({species2}): {species2_total_duplicates}")

#     # extract the homo and nonhomo adata
#     adata_ref_homo = adata_ref[:, species1_genes_in_adata].copy()
#     nonhomo_genes_ref = ~adata_ref.var_names.isin(species1_genes_in_adata)
#     adata_ref_nonhomo = adata_ref[:, nonhomo_genes_ref].copy()
#     ref_X_homo, ref_X_nonhomo = adata_ref_homo.X, adata_ref_nonhomo.X

#     adata_target_homo = adata_target[:, species2_genes_in_adata].copy()
#     nonhomo_genes_target = ~adata_target.var_names.isin(species2_genes_in_adata)
#     adata_target_nonhomo = adata_target[:, nonhomo_genes_target].copy()
#     target_X_homo, target_X_nonhomo = adata_target_homo.X, adata_target_nonhomo.X

#     # preprocess the raw counts
#     adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = preprocess(args, adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo)

#     # Convert to dense if needed
#     if hasattr(ref_X_homo, 'toarray'):
#         ref_X_homo = ref_X_homo.toarray()
#     if hasattr(ref_X_nonhomo, 'toarray'):
#         ref_X_nonhomo = ref_X_nonhomo.toarray()
#     if hasattr(target_X_homo, 'toarray'):
#         target_X_homo = target_X_homo.toarray()
#     if hasattr(target_X_nonhomo, 'toarray'):
#         target_X_nonhomo = target_X_nonhomo.toarray()

#     # Extract training features
#     ref_pos = adata_ref.obsm[loc_name_ref]
#     ref_y_raw = adata_ref.obs[celltype_name_ref].str.lower()
#     ref_cell_types = np.sort(ref_y_raw.unique()).tolist()
#     if region_name_ref:
#         ref_slices = adata_ref.obs[region_name_ref]
#         ref_edges = get_edge_index_standard_region(ref_pos, ref_slices, distance_thres_ref)
#     else:
#         ref_edges = get_edge_index_standard(ref_pos, distance_thres_ref)

#     ref_cell_type_dict = {ct: i for i, ct in enumerate(ref_cell_types)}
#     inverse_dict_ref = {i: ct for ct, i in ref_cell_type_dict.items()}
#     ref_y = np.array([ref_cell_type_dict[ct] for ct in ref_y_raw])

#     # Extract test features
#     target_pos = adata_target.obsm[loc_name_target]
#     target_y_raw = adata_target.obs[celltype_name_target].str.lower()
#     target_cell_types = np.sort(target_y_raw.unique()).tolist()
#     if region_name_target:
#         target_slices = adata_target.obs[region_name_target]
#         target_edges = get_edge_index_standard_region(target_pos, target_slices, distance_thres_target)
#     else:
#         target_edges = get_edge_index_standard(target_pos, distance_thres_target)

#     target_cell_type_dict = {ct: i for i, ct in enumerate(target_cell_types)}
#     inverse_dict_target = {i: ct for ct, i in target_cell_type_dict.items()}
#     target_y_c = np.array([target_cell_type_dict[ct] for ct in target_y_raw])

#     return ref_X_homo, ref_X_nonhomo, ref_y, ref_edges, inverse_dict_ref, target_X_homo, target_X_nonhomo, target_y_c, target_edges, inverse_dict_target

def extract_and_preprocess_data(args, adata_ref, adata_target, species1, species2, homo_genes_df):
    # rename cell types to lower case to keep consistency
    adata_ref.obs = adata_ref.obs.copy()
    adata_target.obs = adata_target.obs.copy()

    # extract the homologous genes
    species1_gene_names = homo_genes_df['Gene name']
    species2_gene_names = homo_genes_df[f'{species2} gene name']

    valid_pairs = [
        (g1, g2) for g1, g2 in zip(species1_gene_names, species2_gene_names)
        if g1 in adata_ref.var_names and g2 in adata_target.var_names
    ]

    species1_genes_in_adata = [pair[0] for pair in valid_pairs]
    species2_genes_in_adata = [pair[1] for pair in valid_pairs]

    print(f"Number of homologous pairs found: {len(valid_pairs)}")
    print(f"Number of {species1} genes found: {len(species1_genes_in_adata)}")
    print(f"Number of {species2} genes found: {len(species2_genes_in_adata)}")

    # Count duplicates
    from collections import Counter
    species1_counter = Counter(species1_genes_in_adata)
    species2_counter = Counter(species2_genes_in_adata)
    print(f"Total duplicates in {species1}: {sum(c-1 for c in species1_counter.values() if c > 1)}")
    print(f"Total duplicates in {species2}: {sum(c-1 for c in species2_counter.values() if c > 1)}")

    # Subset and preprocess
    adata_ref_homo = adata_ref[:, species1_genes_in_adata].copy()
    adata_ref_homo = adata_ref_homo[:, ~adata_ref_homo.var_names.duplicated(keep='first')] # remove the duplicate variables
    adata_ref_nonhomo = adata_ref[:, ~adata_ref.var_names.isin(species1_genes_in_adata)].copy()
    adata_target_homo = adata_target[:, species2_genes_in_adata].copy()
    adata_target_homo = adata_target_homo[:, ~adata_target_homo.var_names.duplicated(keep='first')] # remove the duplicate variables
    adata_target_nonhomo = adata_target[:, ~adata_target.var_names.isin(species2_genes_in_adata)].copy()

    # Preprocess (assumes preprocess function modifies AnnData objects in-place or returns updated copies)
    return preprocess(args, adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo, valid_pairs)

def load_dataset(args, adata_ref_homo, adata_ref_nonhomo, 
                adata_target_homo, adata_target_nonhomo,
                distance_thres_ref, distance_thres_target,
                celltype_name_ref, celltype_name_target,
                loc_name_ref, loc_name_target,
                region_name_ref=None, region_name_target=None):
    
    # Preprocess and extract features
    # adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo = extract_and_preprocess_data(
    #     args, adata_ref, adata_target, species1, species2, homo_genes_df
    # )

    # Convert to dense if needed
    ref_X_homo = adata_ref_homo.X.toarray() if hasattr(adata_ref_homo.X, 'toarray') else adata_ref_homo.X
    ref_X_nonhomo = adata_ref_nonhomo.X.toarray() if hasattr(adata_ref_nonhomo.X, 'toarray') else adata_ref_nonhomo.X
    target_X_homo = adata_target_homo.X.toarray() if hasattr(adata_target_homo.X, 'toarray') else adata_target_homo.X
    target_X_nonhomo = adata_target_nonhomo.X.toarray() if hasattr(adata_target_nonhomo.X, 'toarray') else adata_target_nonhomo.X

    # Reference labels and edges
    ref_pos = adata_ref_homo.obsm[loc_name_ref]
    ref_y_raw = adata_ref_homo.obs[celltype_name_ref].str.lower()
    ref_cell_types = np.sort(ref_y_raw.unique()).tolist()
    ref_edges = get_edge_index_standard_region(ref_pos, adata_ref_homo.obs[region_name_ref], distance_thres_ref) if region_name_ref else get_edge_index_standard(ref_pos, distance_thres_ref)
    ref_cell_type_dict = {ct: i for i, ct in enumerate(ref_cell_types)}
    inverse_dict_ref = {i: ct for ct, i in ref_cell_type_dict.items()}
    ref_y = np.array([ref_cell_type_dict[ct] for ct in ref_y_raw])

    # Target labels and edges
    target_pos = adata_target_homo.obsm[loc_name_target]
    target_y_raw = adata_target_homo.obs[celltype_name_target].str.lower()
    target_cell_types = np.sort(target_y_raw.unique()).tolist()
    target_edges = get_edge_index_standard_region(target_pos, adata_target_homo.obs[region_name_target], distance_thres_target) if region_name_target else get_edge_index_standard(target_pos, distance_thres_target)
    target_cell_type_dict = {ct: i for i, ct in enumerate(target_cell_types)}
    inverse_dict_target = {i: ct for ct, i in target_cell_type_dict.items()}
    target_y_c = np.array([target_cell_type_dict[ct] for ct in target_y_raw])

    return (ref_X_homo, ref_X_nonhomo, ref_y, ref_edges, inverse_dict_ref,
            target_X_homo, target_X_nonhomo, target_y_c, target_edges, inverse_dict_target)


class GraphDataset(InMemoryDataset):
    def __init__(self, ref_X_homo=None, ref_X_nonhomo=None, ref_y=None,
                 target_X_homo=None, target_X_nonhomo=None, target_y=None,
                 ref_edges=None, target_edges=None,
                 ref_data=None, target_data=None):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root)
        if ref_data and target_data:
            self.ref_data = ref_data
            self.target_data = target_data
        else:
            self.ref_data = Data(homo_x=torch.FloatTensor(ref_X_homo),
                                nonhomo_x=torch.FloatTensor(ref_X_nonhomo),
                                edge_index=torch.LongTensor(ref_edges).T,
                                y=torch.LongTensor(ref_y))
            self.ref_data.x = torch.cat([self.ref_data.homo_x, self.ref_data.nonhomo_x], dim=1)
            
            self.target_data = Data(homo_x=torch.FloatTensor(target_X_homo),
                                    nonhomo_x=torch.FloatTensor(target_X_nonhomo),
                                    edge_index=torch.LongTensor(target_edges).T,
                                    y=torch.LongTensor(target_y))
            self.target_data.x = torch.cat([self.target_data.homo_x, self.target_data.nonhomo_x], dim=1)
            

    @classmethod
    def from_saved(cls, ref_path='ref_data.pt', target_path='target_data.pt'):
        ref_data = torch.load(ref_path)
        target_data = torch.load(target_path)
        return cls(ref_data=ref_data, target_data=target_data)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.ref_data, self.target_data