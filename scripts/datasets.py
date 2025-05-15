import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess
import sklearn.neighbors
import scipy.sparse as sp

def get_edge_index_standard(pos: np.ndarray, distance_thres: float):
    """
    Construct edge index based on pairwise Euclidean distance threshold.
    pos: (N, D) numpy array of coordinates
    distance_thres: connect if dist < threshold
    Returns: list of [i, j] edges
    """
    dists = pairwise_distances(pos)
    mask = (dists < distance_thres)
    np.fill_diagonal(mask, False)
    edges = np.transpose(np.nonzero(mask))
    return edges.tolist()


def get_edge_index_standard_region(pos: np.ndarray, regions: np.ndarray, distance_thres: float):
    """
    Same as get_edge_index_standard but only connecting within the same region.
    regions: (N,) array of region labels
    """
    edges = []
    for reg in np.unique(regions):
        idx = np.where(regions == reg)[0]
        subpos = pos[idx]
        d = pairwise_distances(subpos)
        m = (d < distance_thres)
        np.fill_diagonal(m, False)
        subedges = np.transpose(np.nonzero(m))
        for i, j in subedges:
            edges.append([idx[i], idx[j]])
    return edges


def prepare_graph_data(adj):
    # adapted from STAGATE

    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)# self-loop
    #data =  adj.tocoo().data
    #adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    # adapted from STAGATE
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def get_edge_index_knn(adata, k):
    Cal_Spatial_Net(adata, k_cutoff=k, model='KNN')
    Spatial_Net = adata.uns['Spatial_Net']
    cells = adata.obs.index
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G_tf = prepare_graph_data(G)
    print(G_tf)
    return G_tf[0]


def filter_adata(adata: sc.AnnData) -> sc.AnnData:
    """
    Basic QC filtering: remove cells with high mitochondrial %, low genes, and genes in few cells.
    """
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    adata = adata[adata.obs['n_genes_by_counts'] > 200, :]
    adata = adata[adata.obs['pct_counts_mt'] < 5, :]
    adata = adata[:, adata.var['n_cells_by_counts'] > 3]
    return adata


def read_dataset(ref_path: str, target_path: str, species1: str, species2: str, skip_QC=False):
    """
    Load and QC filter two AnnData objects, assign deconvolution-based cell types.
    Returns: species1_adata, species2_adata, homo_genes_df
    """
    ad1 = sc.read_h5ad(ref_path)
    ad2 = sc.read_h5ad(target_path)
    if not skip_QC:
        ad1 = filter_adata(ad1)
        ad2 = filter_adata(ad2)


    # # assume deconv probs in obs columns 10:24 and 10:28 respectively
    # prob2 = ad2.obs.iloc[:, 10:24]
    # prob2['cell_type'] = prob2.idxmax(axis=1)
    # ad2.obs['cell_type'] = prob2['cell_type']
    # prob1 = ad1.obs.iloc[:, 10:28]
    # prob1['cell_type'] = prob1.idxmax(axis=1)
    # ad1.obs['cell_type'] = prob1['cell_type']

    tsv = f'/oscar/data/yma16/Project/Cross_species/00_Data/homolog_genes/{species1}_{species2}.tsv'
    homo_df = pd.read_csv(tsv, sep='\t')
    return ad1, ad2, homo_df


def extract_and_preprocess_data(args, ad1, ad2, species1: str, species2: str, homo_df: pd.DataFrame, homo_gene_id: str):
    """
    Subset AnnData to homologous vs non-homologous genes and preprocess.
    Returns: four AnnData: ref_homo, ref_nonhomo, tgt_homo, tgt_nonhomo
    plus valid_pairs list.
    """
    genes1 = homo_df[homo_gene_id]
    lowered = homo_gene_id[0].lower() + homo_gene_id[1:]
    genes2 = homo_df[f'{species2} {lowered}']
    pairs = [(g1, g2) for g1, g2 in zip(genes1, genes2)
             if g1 in ad1.var_names and g2 in ad2.var_names]
    ref_homo = ad1[:, [p[0] for p in pairs]].copy()
    ref_homo = ref_homo[:, ~ref_homo.var_names.duplicated()]
    ref_non = ad1[:, ~ad1.var_names.isin([p[0] for p in pairs])].copy()
    tgt_homo = ad2[:, [p[1] for p in pairs]].copy()
    tgt_homo = tgt_homo[:, ~tgt_homo.var_names.duplicated()]
    tgt_non = ad2[:, ~ad2.var_names.isin([p[1] for p in pairs])].copy()

    return preprocess(
        args, ref_homo, ref_non,
        tgt_homo, tgt_non, pairs
    )



def load_dataset(
    args,
    ad_ref_homo, ad_ref_non,
    ad_tgt_homo, ad_tgt_non,
    k_ref: int, k_tgt: int,
    ct_ref: str, ct_tgt: str,
    loc_ref: str, loc_tgt: str,
    region_ref: str = None, region_tgt: str = None,
    identity_graph: bool = False
):
    """
    Build PyG Data objects for reference and target graphs.
    If identity_graph=True, only self-loops will be created (no neighborhood edges).
    Returns GraphDataset, inverse label dicts.
    """
    # Data matrices
    Xrh = ad_ref_homo.X.toarray() if hasattr(ad_ref_homo.X, 'toarray') else ad_ref_homo.X
    Xrn = ad_ref_non.X.toarray() if hasattr(ad_ref_non.X, 'toarray') else ad_ref_non.X
    Xth = ad_tgt_homo.X.toarray() if hasattr(ad_tgt_homo.X, 'toarray') else ad_tgt_homo.X
    Xtn = ad_tgt_non.X.toarray() if hasattr(ad_tgt_non.X, 'toarray') else ad_tgt_non.X

    # Labels
    y1_raw = ad_ref_homo.obs[ct_ref].str.lower().values
    types1 = sorted(np.unique(y1_raw))
    d1 = {t: i for i, t in enumerate(types1)}
    y1 = np.array([d1[c] for c in y1_raw])

    y2_raw = ad_tgt_homo.obs[ct_tgt].str.lower().values
    types2 = sorted(np.unique(y2_raw))
    d2 = {t: i for i, t in enumerate(types2)}
    y2 = np.array([d2[c] for c in y2_raw])

    def build_identity_graph(adata):
        n = adata.obs.shape[0]
        return [[i, i] for i in range(n)]

    # Edge index construction
    def build_edges_distance(pos, regions, thres, region_flag):
        if region_flag:
            return get_edge_index_standard_region(pos, regions, thres)
        return get_edge_index_standard(pos, thres)

    def build_edges_knn(adata, k, regions, region_flag):
        if region_flag:
            # TODO
            return get_edge_index_standard_region(pos, regions, thres)
        return get_edge_index_knn(adata, k)
    
    if identity_graph:
        e1 = build_identity_graph(ad_ref_homo)
        e2 = build_identity_graph(ad_tgt_homo)
    else:
        e1 = build_edges_knn(
            ad_ref_homo,
            k_ref,
            ad_ref_homo.obs[region_ref].values if region_ref else None,
            region_ref is not None
        )

        e2 = build_edges_knn(
            ad_tgt_homo,
            k_tgt,
            ad_tgt_homo.obs[region_tgt].values if region_tgt else None,
            region_tgt is not None
        )

    # Build Data objects
    ref_data = Data(
        homo_x=torch.FloatTensor(Xrh),
        nonhomo_x=torch.FloatTensor(Xrn),
        edge_index=torch.LongTensor(e1).T.contiguous(),
        y=torch.LongTensor(y1)
    )
    ref_data.x = torch.cat([ref_data.homo_x, ref_data.nonhomo_x], dim=1)

    tgt_data = Data(
        homo_x=torch.FloatTensor(Xth),
        nonhomo_x=torch.FloatTensor(Xtn),
        edge_index=torch.LongTensor(e2).T.contiguous(),
        y=torch.LongTensor(y2)
    )
    tgt_data.x = torch.cat([tgt_data.homo_x, tgt_data.nonhomo_x], dim=1)

    inv_ref = {v: k for k, v in d1.items()}
    inv_tgt = {v: k for k, v in d2.items()}

    return GraphDataset(ref_data, tgt_data), inv_ref, inv_tgt

class GraphDataset(Dataset):
    def __init__(self, ref_data, target_data):
        self.ref_data    = ref_data
        self.target_data = target_data

    @classmethod
    def from_saved(cls, ref_path: str, target_path: str):
        ref = torch.load(ref_path)
        tgt = torch.load(target_path)
        return cls(ref, tgt)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.ref_data, self.target_data