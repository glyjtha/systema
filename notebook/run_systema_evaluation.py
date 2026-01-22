"""
Script to run Systema evaluation metrics and create plots.
This script should be run within the notebook context or can be imported.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('/gpfs/home/juntongy/gloria/systema/systema/evaluation')
from eval_utils import average_of_perturbation_centroids, get_topk_de_gene_ids, compute_shift_similarities
from centroid_accuracy import calculate_centroid_accuracies
from pearson_delta_reference_metrics import pearson_delta_reference_metrics
from scipy.stats import pearsonr


def prepare_centroids(adata, pred, Xpred, PERT_COL="condition"):
    """Compute ground-truth and predicted centroids per condition."""
    print("Preparing data and computing centroids...")
    
    # Get predictions matrix
    if isinstance(Xpred, np.ndarray):
        if Xpred.shape[0] != len(pred.obs):
            print("Warning: Xpred shape mismatch. Using pred.X or pred.layers/obsm")
            Xpred = pred.X if hasattr(pred.X, 'shape') else np.array(pred.X)
    else:
        Xpred = np.array(Xpred)
    
    # Ensure predictions and ground truth are aligned by cell
    adata_aligned = adata.copy()
    if len(pred.obs_names) == len(adata.obs_names):
        pred_aligned = pred.copy()
    else:
        common_cells = adata.obs_names.intersection(pred.obs_names)
        if len(common_cells) > 0:
            adata_aligned = adata[common_cells].copy()
            pred_aligned = pred[common_cells].copy()
            if Xpred.shape[0] == len(pred.obs_names):
                cell_idx = [pred.obs_names.get_loc(c) for c in common_cells]
                Xpred = Xpred[cell_idx]
        else:
            pred_aligned = pred.copy()
            print("Warning: No matching cell IDs found. Using predictions as-is.")
    
    # Convert to dense if sparse
    if hasattr(adata_aligned.X, 'toarray'):
        adata_aligned.X = adata_aligned.X.toarray()
    elif hasattr(adata_aligned.X, 'todense'):
        adata_aligned.X = np.array(adata_aligned.X.todense())
    if hasattr(Xpred, 'toarray'):
        Xpred = Xpred.toarray()
    Xpred = np.array(Xpred)
    
    print(f"GT shape: {adata_aligned.shape}, Pred shape: {Xpred.shape}")
    
    # Compute centroids
    unique_conditions = sorted([c for c in adata_aligned.obs[PERT_COL].unique() if str(c).lower() != 'ctrl'])
    
    post_gt_df = pd.DataFrame(columns=adata_aligned.var_names)
    post_pred_df = pd.DataFrame(columns=adata_aligned.var_names)
    
    for condition in tqdm(unique_conditions, desc="Computing centroids"):
        # Ground truth centroid
        mask_gt = adata_aligned.obs[PERT_COL] == condition
        if mask_gt.sum() > 0:
            gt_centroid = np.array(adata_aligned[mask_gt].X.mean(axis=0)).ravel()
            post_gt_df.loc[condition] = gt_centroid
        
        # Predicted centroid - align by condition
        if len(pred_aligned.obs) == len(adata_aligned.obs):
            mask_pred = pred_aligned.obs[PERT_COL] == condition
        else:
            # Try to match by condition if cell alignment doesn't work
            mask_pred = pred_aligned.obs[PERT_COL] == condition
        
        if mask_pred.sum() > 0:
            if Xpred.shape[0] == len(pred_aligned.obs):
                pred_centroid = np.array(Xpred[mask_pred].mean(axis=0)).ravel()
            elif Xpred.shape[0] == len(mask_pred):
                pred_centroid = np.array(Xpred.mean(axis=0)).ravel()
            else:
                pred_centroid = np.array(pred_aligned[mask_pred].X.mean(axis=0)).ravel()
            post_pred_df.loc[condition] = pred_centroid
    
    print(f"GT centroids: {post_gt_df.shape}, Pred centroids: {post_pred_df.shape}")
    return post_gt_df, post_pred_df, adata_aligned, pred_aligned, Xpred


def quantify_systematic_variation(post_gt_df, post_pred_df, adata):
    """Quantify systematic variation using Systema's compute_shift_similarities."""
    print("Quantifying systematic variation...")
    
    # Use Systema's compute_shift_similarities if adata has control column
    # Otherwise compute manually using perturbation mean as reference
    if 'control' in adata.obs.columns:
        # Use Systema's function
        df, df_pair, df_norm, pert_names = compute_shift_similarities(adata, control_mean=None)
        # Extract pairwise similarities from GT (using avg_ctl or avg_pert reference)
        pairwise_sims_gt_flat = df_pair[df_pair['variable'] == 'avg_pert']['value'].values
        
        # For predictions, compute manually since we don't have pred in adata format
        pert_mean = post_gt_df.values.mean(axis=0)
        conditions_list = list(post_gt_df.index.intersection(post_pred_df.index))
        pert_shifts_pred = []
        for condition in conditions_list:
            if condition in post_pred_df.index:
                shift_pred = post_pred_df.loc[condition].values - pert_mean
                pert_shifts_pred.append(shift_pred)
        pert_shifts_pred = np.array(pert_shifts_pred)
        pairwise_sims_pred = cosine_similarity(pert_shifts_pred)
        n = len(conditions_list)
        pairwise_sims_pred_flat = pairwise_sims_pred[np.triu_indices(n, k=1)]
        
    else:
        # Manual computation: use perturbation mean as reference
        pert_mean = post_gt_df.values.mean(axis=0)
        conditions_list = list(post_gt_df.index.intersection(post_pred_df.index))
        pert_shifts_gt = []
        pert_shifts_pred = []
        
        for condition in conditions_list:
            if condition in post_gt_df.index:
                shift_gt = post_gt_df.loc[condition].values - pert_mean
                pert_shifts_gt.append(shift_gt)
            if condition in post_pred_df.index:
                shift_pred = post_pred_df.loc[condition].values - pert_mean
                pert_shifts_pred.append(shift_pred)
        
        pert_shifts_gt = np.array(pert_shifts_gt)
        pert_shifts_pred = np.array(pert_shifts_pred)
        
        # Compute pairwise cosine similarities
        pairwise_sims_gt = cosine_similarity(pert_shifts_gt)
        pairwise_sims_pred = cosine_similarity(pert_shifts_pred)
        
        # Extract upper triangle
        n = len(conditions_list)
        pairwise_sims_gt_flat = pairwise_sims_gt[np.triu_indices(n, k=1)]
        pairwise_sims_pred_flat = pairwise_sims_pred[np.triu_indices(n, k=1)]
    
    print(f"Systematic variation (mean pairwise similarity):")
    print(f"  GT: {pairwise_sims_gt_flat.mean():.3f} ± {pairwise_sims_gt_flat.std():.3f}")
    print(f"  Pred: {pairwise_sims_pred_flat.mean():.3f} ± {pairwise_sims_pred_flat.std():.3f}")
    
    return pairwise_sims_gt_flat, pairwise_sims_pred_flat, pert_mean


def compute_centroid_accuracy(post_gt_df, post_pred_df, method_name="model"):
    """Compute centroid accuracy metric."""
    print("Computing centroid accuracy...")
    
    # Prepare data in required format
    post_pred_multiindex = post_pred_df.copy()
    post_pred_multiindex.index = pd.MultiIndex.from_tuples(
        [(cond, method_name) for cond in post_pred_multiindex.index],
        names=['condition', 'method']
    )
    
    # Ensure same conditions and genes
    common_conditions = post_gt_df.index.intersection(post_pred_multiindex.index.get_level_values(0))
    post_gt_subset = post_gt_df.loc[common_conditions]
    post_pred_subset = post_pred_multiindex.loc[post_pred_multiindex.index.get_level_values(0).isin(common_conditions)]
    
    common_genes = post_gt_subset.columns.intersection(post_pred_subset.columns)
    post_gt_subset = post_gt_subset[common_genes]
    post_pred_subset = post_pred_subset[common_genes]
    
    # Calculate centroid accuracies
    centroid_acc_df = calculate_centroid_accuracies(post_pred_subset, post_gt_subset)
    print(f"Centroid accuracy: {centroid_acc_df[method_name].mean():.3f} ± {centroid_acc_df[method_name].std():.3f}")
    print(f"Results for {len(centroid_acc_df)} conditions")
    
    return centroid_acc_df, post_gt_subset, post_pred_subset, common_conditions


def compute_pearson_metrics(post_gt_subset, post_pred_subset, pert_mean, common_conditions, method_name="model", control_mean=None):
    """Compute PearsonΔ metrics following Systema evaluation."""
    print("Computing PearsonΔ metrics...")
    
    # If control_mean is not provided, we'll compute using pert_mean as reference
    # (which is the "allpert" version - delta from perturbation mean)
    reference = pert_mean  # Using perturbation mean as reference (PearsonΔ_allpert)
    
    pearson_metrics = []
    
    for condition in tqdm(common_conditions, desc="Computing PearsonΔ"):
        if condition not in post_gt_subset.index:
            continue
        if (condition, method_name) not in post_pred_subset.index:
            continue
        
        X_true = post_gt_subset.loc[condition].values
        X_pred = post_pred_subset.loc[(condition, method_name)].values
        
        # Get top 20 DE genes using the shift from reference
        top20_de_idxs = get_topk_de_gene_ids(reference, X_true, k=20)
        
        # Compute metrics with perturbation mean as reference (PearsonΔ_allpert)
        metrics = pearson_delta_reference_metrics(X_true, X_pred, reference, top20_de_idxs)
        metrics['condition'] = condition
        pearson_metrics.append(metrics)
    
    pearson_df = pd.DataFrame(pearson_metrics)
    print(f"\nPearsonΔ metrics (using perturbation mean as reference):")
    print(f"  PearsonΔ (all genes): {pearson_df['corr_all_allpert'].mean():.3f} ± {pearson_df['corr_all_allpert'].std():.3f}")
    print(f"  PearsonΔ20 (top 20 DE): {pearson_df['corr_20de_allpert'].mean():.3f} ± {pearson_df['corr_20de_allpert'].std():.3f}")
    
    return pearson_df


def create_plots(pairwise_sims_gt_flat, pairwise_sims_pred_flat, 
                 centroid_acc_df, pearson_df, method_name="model"):
    """Create evaluation plots."""
    print("Creating plots...")
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 100
    fontsize = 12
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Systematic variation
    axes[0, 0].hist(pairwise_sims_gt_flat, bins=30, alpha=0.7, label='Ground Truth', color='blue', edgecolor='black')
    axes[0, 0].hist(pairwise_sims_pred_flat, bins=30, alpha=0.7, label='Predictions', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Pairwise Cosine Similarity', fontsize=fontsize)
    axes[0, 0].set_ylabel('Frequency', fontsize=fontsize)
    axes[0, 0].set_title('Systematic Variation\n(Pairwise Perturbation Similarities)', fontsize=fontsize+1)
    axes[0, 0].legend(fontsize=fontsize-1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Centroid Accuracy
    centroid_acc_df_sorted = centroid_acc_df.sort_values(method_name, ascending=False)
    top_n = min(20, len(centroid_acc_df_sorted))
    axes[0, 1].barh(range(top_n), centroid_acc_df_sorted[method_name].values[:top_n], alpha=0.7)
    axes[0, 1].set_yticks(range(top_n))
    axes[0, 1].set_yticklabels(centroid_acc_df_sorted.index[:top_n], fontsize=8)
    axes[0, 1].set_xlabel('Centroid Accuracy', fontsize=fontsize)
    axes[0, 1].set_title(f'Centroid Accuracy (Top {top_n})\n(Mean: {centroid_acc_df[method_name].mean():.3f})', fontsize=fontsize+1)
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    axes[0, 1].set_xlim([0, 1])
    mean_acc = centroid_acc_df[method_name].mean()
    axes[0, 1].axvline(mean_acc, color='red', linestyle='--', linewidth=2)
    
    # Plot 3: PearsonΔ distribution
    axes[1, 0].hist(pearson_df['corr_all_allpert'].dropna(), bins=30, alpha=0.7, label='PearsonΔ (all)', color='blue', edgecolor='black')
    axes[1, 0].hist(pearson_df['corr_20de_allpert'].dropna(), bins=30, alpha=0.7, label='PearsonΔ20 (top 20 DE)', color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Correlation', fontsize=fontsize)
    axes[1, 0].set_ylabel('Frequency', fontsize=fontsize)
    axes[1, 0].set_title('PearsonΔ Metrics Distribution', fontsize=fontsize+1)
    axes[1, 0].legend(fontsize=fontsize-1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 4: Summary
    metrics_summary = {
        'Centroid\nAccuracy': centroid_acc_df[method_name].mean(),
        'PearsonΔ\n(all)': pearson_df['corr_all_allpert'].mean(),
        'PearsonΔ20\n(top 20)': pearson_df['corr_20de_allpert'].mean(),
        'Sys Var\n(GT)': pairwise_sims_gt_flat.mean(),
        'Sys Var\n(Pred)': pairwise_sims_pred_flat.mean(),
    }
    axes[1, 1].bar(range(len(metrics_summary)), list(metrics_summary.values()), 
                  color=['blue', 'green', 'orange', 'purple', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(metrics_summary)))
    axes[1, 1].set_xticklabels(list(metrics_summary.keys()), fontsize=fontsize-1, rotation=15, ha='right')
    axes[1, 1].set_ylabel('Score', fontsize=fontsize)
    axes[1, 1].set_title('Systema Metrics Summary', fontsize=fontsize+1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, (key, val) in enumerate(metrics_summary.items()):
        axes[1, 1].text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=fontsize-2)
    
    plt.tight_layout()
    plt.show()


def run_full_evaluation(adata, pred, Xpred, PERT_COL="condition", method_name="model"):
    """Run complete Systema evaluation pipeline."""
    # Step 1: Prepare centroids
    post_gt_df, post_pred_df, adata_aligned, pred_aligned, Xpred_aligned = prepare_centroids(
        adata, pred, Xpred, PERT_COL
    )
    
    # Step 2: Quantify systematic variation
    pairwise_sims_gt_flat, pairwise_sims_pred_flat, pert_mean = quantify_systematic_variation(
        post_gt_df, post_pred_df, adata_aligned
    )
    
    # Step 3: Compute centroid accuracy
    centroid_acc_df, post_gt_subset, post_pred_subset, common_conditions = compute_centroid_accuracy(
        post_gt_df, post_pred_df, method_name
    )
    
    # Step 4: Compute PearsonΔ metrics
    pearson_df = compute_pearson_metrics(
        post_gt_subset, post_pred_subset, pert_mean, common_conditions, method_name
    )
    
    # Step 5: Create plots
    create_plots(pairwise_sims_gt_flat, pairwise_sims_pred_flat, 
                 centroid_acc_df, pearson_df, method_name)
    
    # Print summary
    print("\n=== Final Summary ===")
    print(f"Number of conditions: {len(common_conditions)}")
    print(f"Centroid Accuracy: {centroid_acc_df[method_name].mean():.3f} ± {centroid_acc_df[method_name].std():.3f}")
    print(f"PearsonΔ (all): {pearson_df['corr_all_allpert'].mean():.3f} ± {pearson_df['corr_all_allpert'].std():.3f}")
    print(f"PearsonΔ20 (top 20 DE): {pearson_df['corr_20de_allpert'].mean():.3f} ± {pearson_df['corr_20de_allpert'].std():.3f}")
    print(f"Systematic Variation (GT): {pairwise_sims_gt_flat.mean():.3f} ± {pairwise_sims_gt_flat.std():.3f}")
    print(f"Systematic Variation (Pred): {pairwise_sims_pred_flat.mean():.3f} ± {pairwise_sims_pred_flat.std():.3f}")
    
    return {
        'centroid_acc_df': centroid_acc_df,
        'pearson_df': pearson_df,
        'pairwise_sims_gt': pairwise_sims_gt_flat,
        'pairwise_sims_pred': pairwise_sims_pred_flat,
        'post_gt_df': post_gt_df,
        'post_pred_df': post_pred_df
    }

