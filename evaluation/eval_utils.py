import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc
import pandas as pd

def jaccard_similarity(list1, list2):
    """
    Compute the Jaccard similarity between two lists.
    """
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_topk_de_gene_ids(ctrl, post, k=20):
    """
    Get the top k differentially expressed genes from the results.
    """
    # Get the top k differentially expressed genes
    diff = post - ctrl
    diff_genes_ids = np.argsort(np.abs(diff))[-k:]
    return diff_genes_ids

def average_of_perturbation_centroids(adata):
    pert_means = []
    pert_adata = adata[adata.obs['control'] == 0]
    for cond in pert_adata.obs['condition'].unique():
        adata_cond = pert_adata[pert_adata.obs['condition'] == cond]
        pert_mean = np.array(adata_cond.X.mean(axis=0))[0]
        pert_means.append(pert_mean)
    pert_means = np.array(pert_means)
    return np.mean(pert_means, axis=0)


def get_perturbation_shifts(adata, reference, top_20=False, control_col="control"):
    if control_col in adata.obs.columns:
        adata = adata[adata.obs[control_col] == 0]

    conditions = set(adata.obs["condition"].unique()) 
    shifts = {}

    for condition in conditions:
        adata_condition = adata[adata.obs["condition"] == condition]
        pert_shift = np.asarray(adata_condition.X.mean(axis=0)).ravel() - reference

        if top_20:
            top20_de_genes = adata.uns["top_non_dropout_de_20"][
                adata_condition.obs["condition_name"].values[0]
            ]
            top20_de_idxs = np.argwhere(np.isin(adata.var.index, top20_de_genes)).ravel()
            pert_shift = pert_shift[top20_de_idxs]

        shifts[condition] = pert_shift

    return shifts


def calculate_cosine_similarities(pert_shifts, reference):
    sims = cosine_similarity(pert_shifts, reference[None, :]).ravel()
    return sims

def calculate_pairwise_cosine_similarities(pert_shifts):
    sims = cosine_similarity(np.array(pert_shifts))
    return sims[np.triu_indices(len(sims), k=1)]

def average_of_perturbation_centroids(adata):
    pert_means = []
    pert_adata = adata[adata.obs['control'] == 0]
    for cond in pert_adata.obs['condition'].unique():
        adata_cond = pert_adata[pert_adata.obs['condition'] == cond]
        pert_mean = np.array(adata_cond.X.mean(axis=0))[0]
        pert_means.append(pert_mean)
    pert_means = np.array(pert_means)
    return np.mean(pert_means, axis=0)

def calculate_norms(pert_shifts):
    return np.linalg.norm(pert_shifts, axis=1)

def compute_shift_similarities(adata, avg_pert_centroids=True, control_mean=None):
    # 1. split pert / control
    pert_adata = adata[adata.obs["control"] == 0]

    if control_mean is None:
        control_adata = adata[adata.obs["control"] == 1]
        control_mean = np.asarray(control_adata.X.mean(axis=0)).ravel()

    # 2. compute pert mean
    if avg_pert_centroids:
        pert_mean = average_of_perturbation_centroids(pert_adata)
    else:
        pert_mean = np.asarray(pert_adata.X.mean(axis=0)).ravel()

    avg_shift = pert_mean - control_mean

    # 3. get shifts as dicts (name -> vector)
    shifts_ctl  = get_perturbation_shifts(pert_adata, reference=control_mean)
    shifts_pert = get_perturbation_shifts(pert_adata, reference=pert_mean)

    # 4. align perturbations by key (NO sorting needed)
    common = shifts_ctl.keys() & shifts_pert.keys()
    perts = list(common)

    v_ctl  = np.vstack([shifts_ctl[p]  for p in perts])
    v_pert = np.vstack([shifts_pert[p] for p in perts])

    # 5. compute metrics
    sims = {
        "avg_ctl": calculate_cosine_similarities(v_ctl, avg_shift),
        "avg_pert": calculate_cosine_similarities(v_pert, avg_shift),
    }

    norms = {
        "avg_ctl": calculate_norms(v_ctl),
        "avg_pert": calculate_norms(v_pert),
    }

    pairwise = {
        "avg_ctl": calculate_pairwise_cosine_similarities(v_ctl),
        "avg_pert": calculate_pairwise_cosine_similarities(v_pert),
    }

    # 6. build DataFrames with explicit name binding
    df = pd.DataFrame(sims)
    df_norm = pd.DataFrame(norms)

    df["pert_names"] = perts
    df_norm["pert_names"] = perts

    df = df.melt(id_vars="pert_names", var_name="variable", value_name="value")
    df_norm = df_norm.melt(id_vars="pert_names", var_name="variable", value_name="value")

    df_pair = pd.DataFrame(pairwise).melt(var_name="variable", value_name="value")

    return df, df_pair, df_norm, perts
