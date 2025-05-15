import scanpy as sc
import numpy as np

def fix_var_index_conflict(adata):
    if adata.var.index.name in adata.var.columns:
        if not (adata.var.index.to_series().equals(adata.var[adata.var.index.name])):
            adata.var.index.name = None
        else:
            adata.var = adata.var.drop(columns=[adata.var.index.name])

def preprocess(args, adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo, valid_pairs, save_data=True, target_sum=1e4):
    print('Reference Homologous Gene:', adata_ref_homo.var.shape[0])
    print('Target Homologous Gene:', adata_target_homo.var.shape[0])
    print('Reference Non-Homologous Gene:', adata_ref_nonhomo.var.shape[0])
    print('Target Non-Homologous Gene:', adata_target_nonhomo.var.shape[0])

    # select hvg for homogenes of each dataset separately then take the intersect
    sc.pp.highly_variable_genes(adata_ref_homo, flavor='seurat_v3', n_top_genes=int(args.hvg))
    sc.pp.highly_variable_genes(adata_target_homo, flavor='seurat_v3', n_top_genes=int(args.hvg))

    # take a intersect of hvg genes of homolog genes
    # take intersect
    hvg_list_ref = np.array(adata_ref_homo.var[adata_ref_homo.var['highly_variable']].index.tolist())
    hvg_list_target = np.array(adata_target_homo.var[adata_target_homo.var['highly_variable']].index.tolist())
    hvg_intersect_ref = []
    hvg_intersect_target = []
    hvg_set_ref = set(hvg_list_ref)
    hvg_set_target = set(hvg_list_target)
    for ref_gene, target_gene in valid_pairs:
        if ref_gene in hvg_set_ref and target_gene in hvg_set_target:
            hvg_intersect_ref.append(ref_gene)
            hvg_intersect_target.append(target_gene)

    hvg_intersect_ref = np.array(hvg_intersect_ref)
    hvg_intersect_target = np.array(hvg_intersect_target)
    adata_ref_homo = adata_ref_homo[:, hvg_intersect_ref]
    adata_target_homo = adata_target_homo[:, hvg_intersect_target]
    print(f"Number of HVG intersect in ref_homo is {len(hvg_intersect_ref)}.", flush=True)
    print(f"Number of HVG intersect in target_homo is {len(hvg_intersect_target)}.", flush=True)

    # # normalization the ref homo
    # sc.pp.normalize_total(adata_ref_homo, target_sum=target_sum)
    # sc.pp.log1p(adata_ref_homo)
    # sc.pp.scale(adata_ref_homo, zero_center=True, max_value=None, copy=False)

    # # normalization the target homo 
    # sc.pp.normalize_total(adata_target_homo, target_sum=target_sum)
    # sc.pp.log1p(adata_target_homo)
    # sc.pp.scale(adata_target_homo, zero_center=True, max_value=None, copy=False)

    # select hvg for nonhomo genes of each dataset
    if adata_ref_nonhomo.n_vars > args.hvg:
        sc.pp.highly_variable_genes(adata_ref_nonhomo, flavor='seurat_v3', n_top_genes = 3000)
        sc.pp.highly_variable_genes(adata_target_nonhomo, flavor='seurat_v3', n_top_genes = 3000)
        hvg_list_ref_nonhomo = np.array(adata_ref_nonhomo.var[adata_ref_nonhomo.var['highly_variable']].index.tolist())
        hvg_list_target_nonhomo = np.array(adata_target_nonhomo.var[adata_target_nonhomo.var['highly_variable']].index.tolist())
        adata_ref_nonhomo = adata_ref_nonhomo[:, adata_ref_nonhomo.var_names.isin(hvg_list_ref_nonhomo)]
        adata_target_nonhomo = adata_target_nonhomo[:, adata_target_nonhomo.var_names.isin(hvg_list_target_nonhomo)]  


    # # normalization the ref nonhomo
    # sc.pp.normalize_total(adata_ref_nonhomo, target_sum=target_sum)
    # sc.pp.log1p(adata_ref_nonhomo)
    # sc.pp.scale(adata_ref_nonhomo, zero_center=True, max_value=None, copy=False)

    # # normalization the target nonhomo 
    # sc.pp.normalize_total(adata_target_nonhomo, target_sum=target_sum)
    # sc.pp.log1p(adata_target_nonhomo)
    # sc.pp.scale(adata_target_nonhomo, zero_center=True, max_value=None, copy=False)

    # save the dataset
    if save_data:
        fix_var_index_conflict(adata_ref_homo)
        fix_var_index_conflict(adata_ref_nonhomo)
        fix_var_index_conflict(adata_target_homo)
        fix_var_index_conflict(adata_target_nonhomo)
        
        # TODO: later modify it to save in the original dataset (or somewhere else) instead of the output files so that no need to repeat preprocessing for different runs
        adata_ref_homo.write_h5ad(f'{args.savedir}/adata_ref_homo_preprocessed.h5ad')
        adata_ref_nonhomo.write_h5ad(f'{args.savedir}/adata_ref_nonhomo_preprocessed.h5ad')
        adata_target_homo.write_h5ad(f'{args.savedir}/adata_target_homo_preprocessed.h5ad')
        adata_target_nonhomo.write_h5ad(f'{args.savedir}/adata_target_nonhomo_preprocessed.h5ad')

    return adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo

def load_preprocessed(savedir):
    adata_ref_homo = sc.read_h5ad(f'{savedir}/adata_ref_homo_preprocessed.h5ad')
    adata_ref_nonhomo = sc.read_h5ad(f'{savedir}/adata_ref_nonhomo_preprocessed.h5ad')
    adata_target_homo = sc.read_h5ad(f'{savedir}/adata_target_homo_preprocessed.h5ad')
    adata_target_nonhomo = sc.read_h5ad(f'{savedir}/adata_target_nonhomo_preprocessed.h5ad')
    return adata_ref_homo, adata_ref_nonhomo, adata_target_homo, adata_target_nonhomo