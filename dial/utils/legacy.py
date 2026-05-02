"""
Utility functions for logging, saving results, and visualization
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Lazy imports — matplotlib/seaborn only needed for plotting, not experiments
plt = None
sns = None

def _ensure_plotting():
    """Import matplotlib and seaborn on first use."""
    global plt, sns
    if plt is None:
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        plt = _plt
        sns = _sns


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def setup_logger(
    name: str = "DIAL",
    level: str = "INFO",
    log_to_file: bool = False,
    log_file: str = None,
    results_dir: str = None
) -> logging.Logger:
    """
    Setup logger with console and optional file output
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_file: Log file name
        results_dir: Directory to save log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_file:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            log_path = os.path.join(results_dir, log_file)
        else:
            log_path = log_file
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_results(
    stats: List[Dict],
    config: Dict,
    summary: Dict,
    results_dir: str,
    save_json: bool = True,
    save_csv: bool = True,
    validation_results: List[Dict] = None
):
    """
    Save experimental results to disk
    
    Args:
        stats: List of per-state statistics
        config: Experiment configuration
        summary: Summary statistics
        results_dir: Directory to save results
        save_json: Save detailed results as JSON
        save_csv: Save summary as CSV
        validation_results: Optional validation results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed stats as JSON
    if save_json:
        json_path = os.path.join(results_dir, f"detailed_stats_{timestamp}.json")
        output = {
            "config": config,
            "summary": summary,
            "stats": stats
        }
        if validation_results:
            output["validation"] = validation_results
        
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
    
    # Save summary as CSV
    if save_csv:
        csv_path = os.path.join(results_dir, f"summary_{timestamp}.csv")
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(csv_path, index=False)
        
        # Also save per-episode stats
        stats_path = os.path.join(results_dir, f"per_state_stats_{timestamp}.csv")
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(stats_path, index=False)
        
        if validation_results:
            val_path = os.path.join(results_dir, f"validation_{timestamp}.csv")
            val_df = pd.DataFrame(validation_results)
            val_df.to_csv(val_path, index=False)


def create_visualizations(
    stats: List[Dict],
    summary: Dict,
    results_dir: str,
    formats: List[str] = ["png"],
    validation_results: List[Dict] = None
):
    """
    Create visualization plots

    Args:
        stats: List of per-state statistics
        summary: Summary statistics
        results_dir: Directory to save plots
        formats: List of file formats (e.g., ["png", "pdf"])
        validation_results: Optional validation results
    """
    _ensure_plotting()
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sns.set_style("whitegrid")
    
    # 1. Inversion rate over episodes
    fig, ax = plt.subplots(figsize=(10, 6))
    df = pd.DataFrame(stats)
    
    if 'episode' in df.columns and 'inversion' in df.columns:
        episode_inv = df.groupby('episode')['inversion'].mean()
        ax.plot(episode_inv.index, episode_inv.values, marker='o', linewidth=2)
        ax.axhline(y=summary['inversion_rate'], color='r', linestyle='--', 
                   label=f'Overall Rate: {summary["inversion_rate"]:.3f}')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Inversion Rate', fontsize=12)
        ax.set_title('Inversion Rate Across Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for fmt in formats:
            plt.savefig(
                os.path.join(results_dir, f"inversion_rate_episodes_{timestamp}.{fmt}"),
                dpi=300, bbox_inches='tight'
            )
    plt.close()
    
    # 2. Retro gap distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'retro_gap' in df.columns:
        gaps = df['retro_gap'].dropna()
        ax.hist(gaps, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=gaps.mean(), color='r', linestyle='--', 
                   label=f'Mean: {gaps.mean():.3f}')
        ax.set_xlabel('Retrospective Gap', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Retrospective Gap', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for fmt in formats:
            plt.savefig(
                os.path.join(results_dir, f"retro_gap_dist_{timestamp}.{fmt}"),
                dpi=300, bbox_inches='tight'
            )
    plt.close()
    
    # 3. Inversion vs Gap scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'inversion' in df.columns and 'retro_gap' in df.columns:
        inv_df = df[df['inversion'] == True]
        non_inv_df = df[df['inversion'] == False]
        
        ax.scatter(range(len(non_inv_df)), non_inv_df['retro_gap'], 
                  alpha=0.5, s=30, label='No Inversion', color='blue')
        ax.scatter(range(len(inv_df)), inv_df['retro_gap'], 
                  alpha=0.5, s=30, label='Inversion', color='red')
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Retrospective Gap', fontsize=12)
        ax.set_title('Retrospective Gap: Inversion vs No Inversion', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for fmt in formats:
            plt.savefig(
                os.path.join(results_dir, f"inversion_gap_scatter_{timestamp}.{fmt}"),
                dpi=300, bbox_inches='tight'
            )
    plt.close()
    
    # 4. Validation results (if available)
    if validation_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        val_df = pd.DataFrame(validation_results)
        
        if 'success_diff' in val_df.columns:
            # Success rate difference
            axes[0].hist(val_df['success_diff'], bins=20, alpha=0.7, 
                        edgecolor='black', color='green')
            axes[0].axvline(x=val_df['success_diff'].mean(), color='r', 
                           linestyle='--', label=f'Mean: {val_df["success_diff"].mean():.3f}')
            axes[0].set_xlabel('Success Rate Difference (R - F)', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Success Rate: Retro vs Forward', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
        
        if 'return_diff' in val_df.columns:
            # Return difference
            axes[1].hist(val_df['return_diff'], bins=20, alpha=0.7, 
                        edgecolor='black', color='orange')
            axes[1].axvline(x=val_df['return_diff'].mean(), color='r', 
                           linestyle='--', label=f'Mean: {val_df["return_diff"].mean():.3f}')
            axes[1].set_xlabel('Return Difference (R - F)', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Return: Retro vs Forward', fontsize=13, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        for fmt in formats:
            plt.savefig(
                os.path.join(results_dir, f"validation_comparison_{timestamp}.{fmt}"),
                dpi=300, bbox_inches='tight'
            )
        plt.close()
    
    # 5. Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Samples", f"{summary['num_samples']}"],
        ["Inversion Rate (argmax)", f"{summary['inversion_rate']:.4f}"],
        ["Inversion Rate (margin-gated)", f"{summary.get('inversion_rate_margined', 0):.4f}"],
        ["Avg Retro Gap (All)", f"{summary.get('avg_retro_gap', 0):.4f}"],
        ["Avg Retro Gap (Inv Only)", f"{summary.get('avg_retro_gap_when_inversion', 0):.4f}"],
        ["Avg F Margin (top1-top2)", f"{summary.get('avg_f_margin', 0):.4f}"],
        ["Avg R Margin (top1-top2)", f"{summary.get('avg_r_margin', 0):.4f}"],
        ["Avg Rank Correlation (Spearman)", f"{summary.get('avg_rank_corr', 0):.4f}"],
    ]
    
    if validation_results:
        val_df = pd.DataFrame(validation_results)
        summary_data.extend([
            ["Validation Samples", f"{len(validation_results)}"],
            ["Avg Success Diff (R-F)", f"{val_df['success_diff'].mean():.4f}"],
            ["Avg Return Diff (R-F)", f"{val_df['return_diff'].mean():.4f}"]
        ])

    if 'avg_depth_inversion' in summary:
        summary_data.extend([
            ["Avg Depth (Inversion)", f"{summary['avg_depth_inversion']:.1f}"],
            ["Avg Depth (No Inversion)", f"{summary['avg_depth_no_inversion']:.1f}"],
        ])
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Experiment Summary', fontsize=14, fontweight='bold', pad=20)
    
    for fmt in formats:
        plt.savefig(
            os.path.join(results_dir, f"summary_table_{timestamp}.{fmt}"),
            dpi=300, bbox_inches='tight'
        )
    plt.close()

    # ================================================================
    # 6. Depth-based analysis: Inversion rate by depth bin
    # ================================================================
    if 'depth' in df.columns:
        depth_bins = [0, 10, 20, 40, 60, 100, 200]
        depth_labels = ['0-9', '10-19', '20-39', '40-59', '60-99', '100+']
        df['depth_bin'] = pd.cut(df['depth'], bins=depth_bins, labels=depth_labels, right=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Inversion rate by depth bin
        depth_stats = df.groupby('depth_bin', observed=True).agg(
            inv_rate=('inversion', 'mean'),
            count=('inversion', 'count'),
        ).reset_index()

        bars = axes[0].bar(depth_stats['depth_bin'].astype(str), depth_stats['inv_rate'],
                           color='#5B9BD5', edgecolor='black', alpha=0.85)
        # annotate counts on bars
        for bar, cnt in zip(bars, depth_stats['count']):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'n={cnt}', ha='center', va='bottom', fontsize=9)
        axes[0].set_xlabel('Depth (step within episode)', fontsize=12)
        axes[0].set_ylabel('Inversion Rate', fontsize=12)
        axes[0].set_title('Inversion Rate by Depth', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1.1)
        axes[0].axhline(y=summary['inversion_rate'], color='r', linestyle='--',
                        label=f'Overall: {summary["inversion_rate"]:.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: Distribution of depths (inversions vs non-inversions)
        inv_depths = df[df['inversion'] == True]['depth']
        non_inv_depths = df[df['inversion'] == False]['depth']
        max_depth = df['depth'].max()
        hist_bins = np.arange(0, max_depth + 10, 5)
        axes[1].hist(non_inv_depths, bins=hist_bins, alpha=0.6, label='No Inversion',
                     color='blue', edgecolor='black')
        axes[1].hist(inv_depths, bins=hist_bins, alpha=0.6, label='Inversion',
                     color='red', edgecolor='black')
        axes[1].set_xlabel('Depth (step)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Depth Distribution: Inversions vs Non-Inversions',
                          fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(os.path.join(results_dir, f"depth_inversion_analysis_{timestamp}.{fmt}"),
                        dpi=300, bbox_inches='tight')
        plt.close()

    # ================================================================
    # 7. Task-stage based analysis
    # ================================================================
    if 'task_stage' in df.columns:
        stage_names = {0: 'Stage 0\n(Find Key)', 1: 'Stage 1\n(Go to Door)', 2: 'Stage 2\n(Go to Goal)'}

        fig, ax = plt.subplots(figsize=(8, 6))
        stage_stats = df.groupby('task_stage').agg(
            inv_rate=('inversion', 'mean'),
            count=('inversion', 'count'),
        ).reset_index()

        x_labels = [stage_names.get(s, str(s)) for s in stage_stats['task_stage']]
        colors = ['#FF6347', '#FFD700', '#32CD32']
        bars = ax.bar(x_labels, stage_stats['inv_rate'],
                      color=[colors[i % 3] for i in range(len(stage_stats))],
                      edgecolor='black', alpha=0.85)
        for bar, cnt in zip(bars, stage_stats['count']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'n={cnt}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Inversion Rate', fontsize=12)
        ax.set_title('Inversion Rate by Task Stage', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=summary['inversion_rate'], color='gray', linestyle='--',
                   label=f'Overall: {summary["inversion_rate"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        for fmt in formats:
            plt.savefig(os.path.join(results_dir, f"stage_inversion_analysis_{timestamp}.{fmt}"),
                        dpi=300, bbox_inches='tight')
        plt.close()

    # ================================================================
    # 8. Depth-based success rate improvement when following inversion
    # ================================================================
    if validation_results:
        val_df = pd.DataFrame(validation_results)

        if 'depth' in val_df.columns and len(val_df) > 0:
            depth_bins_v = [0, 10, 20, 40, 60, 100, 200]
            depth_labels_v = ['0-9', '10-19', '20-39', '40-59', '60-99', '100+']
            val_df['depth_bin'] = pd.cut(val_df['depth'], bins=depth_bins_v,
                                         labels=depth_labels_v, right=False)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left: Success rate improvement by depth
            vd_stats = val_df.groupby('depth_bin', observed=True).agg(
                mean_success_diff=('success_diff', 'mean'),
                mean_return_diff=('return_diff', 'mean'),
                count=('success_diff', 'count'),
                mean_success_F=('success_F', 'mean'),
                mean_success_R=('success_R', 'mean'),
            ).reset_index()

            x = np.arange(len(vd_stats))
            width = 0.35
            bars1 = axes[0].bar(x - width/2, vd_stats['mean_success_F'], width,
                               label='Forward $a_F$', color='#5B9BD5', edgecolor='black')
            bars2 = axes[0].bar(x + width/2, vd_stats['mean_success_R'], width,
                               label='Retro $a_R$', color='#ED7D31', edgecolor='black')
            for bar, cnt in zip(bars2, vd_stats['count']):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                             f'n={cnt}', ha='center', va='bottom', fontsize=8)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(vd_stats['depth_bin'].astype(str))
            axes[0].set_xlabel('Depth (step)', fontsize=12)
            axes[0].set_ylabel('Success Rate', fontsize=12)
            axes[0].set_title('Success Rate by Depth\n(at Inversion Points)', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].set_ylim(0, 1.0)

            # Right: Success rate difference (R-F) by depth
            colors_diff = ['green' if v >= 0 else 'red' for v in vd_stats['mean_success_diff']]
            bars = axes[1].bar(vd_stats['depth_bin'].astype(str), vd_stats['mean_success_diff'],
                               color=colors_diff, edgecolor='black', alpha=0.85)
            for bar, diff, cnt in zip(bars, vd_stats['mean_success_diff'], vd_stats['count']):
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + (0.01 if diff >= 0 else -0.03),
                             f'{diff:+.2f}\nn={cnt}', ha='center',
                             va='bottom' if diff >= 0 else 'top', fontsize=8)
            axes[1].axhline(y=0, color='black', linewidth=0.8)
            axes[1].set_xlabel('Depth (step)', fontsize=12)
            axes[1].set_ylabel('Success Rate Diff (R - F)', fontsize=12)
            axes[1].set_title('Success Rate Improvement\nfrom Following Inversion', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            for fmt in formats:
                plt.savefig(os.path.join(results_dir, f"depth_success_improvement_{timestamp}.{fmt}"),
                            dpi=300, bbox_inches='tight')
            plt.close()

        # 9. Task-stage based success rate improvement
        if 'task_stage' in val_df.columns and len(val_df) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            stage_names_v = {0: 'Stage 0\n(Find Key)', 1: 'Stage 1\n(Go to Door)', 2: 'Stage 2\n(Go to Goal)'}

            vs_stats = val_df.groupby('task_stage').agg(
                mean_success_F=('success_F', 'mean'),
                mean_success_R=('success_R', 'mean'),
                mean_success_diff=('success_diff', 'mean'),
                mean_return_diff=('return_diff', 'mean'),
                count=('success_diff', 'count'),
            ).reset_index()

            x = np.arange(len(vs_stats))
            width = 0.3
            ax.bar(x - width/2, vs_stats['mean_success_F'], width,
                   label='Forward $a_F$', color='#5B9BD5', edgecolor='black')
            ax.bar(x + width/2, vs_stats['mean_success_R'], width,
                   label='Retro $a_R$', color='#ED7D31', edgecolor='black')

            labels = [stage_names_v.get(s, str(s)) for s in vs_stats['task_stage']]
            for i, (_, row) in enumerate(vs_stats.iterrows()):
                diff = row['mean_success_diff']
                sign = '+' if diff >= 0 else ''
                ax.text(i, max(row['mean_success_F'], row['mean_success_R']) + 0.04,
                        f'Δ={sign}{diff:.2f}\nn={int(row["count"])}',
                        ha='center', fontsize=9, fontweight='bold',
                        color='green' if diff >= 0 else 'red')

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.set_title('Success Rate Improvement by Task Stage\n(at Inversion Points)',
                         fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.0)

            plt.tight_layout()
            for fmt in formats:
                plt.savefig(os.path.join(results_dir, f"stage_success_improvement_{timestamp}.{fmt}"),
                            dpi=300, bbox_inches='tight')
            plt.close()


def summarize_stats(stats: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics from per-state measurements
    
    Args:
        stats: List of per-state statistics
        
    Returns:
        Dictionary of summary statistics
    """
    if not stats:
        return {
            "num_samples": 0,
            "inversion_rate": 0.0,
            "avg_retro_gap": 0.0,
            "avg_retro_gap_when_inversion": 0.0
        }
    
    df = pd.DataFrame(stats)
    
    inversions = df['inversion'].sum() if 'inversion' in df.columns else 0
    inversion_rate = inversions / len(stats)
    
    avg_gap = df['retro_gap'].mean() if 'retro_gap' in df.columns else 0.0
    
    inv_gaps = df[df['inversion'] == True]['retro_gap'] if 'inversion' in df.columns else []
    avg_gap_inv = inv_gaps.mean() if len(inv_gaps) > 0 else 0.0

    result = {
        "num_samples": len(stats),
        "inversion_rate": inversion_rate,
        "avg_retro_gap": avg_gap,
        "avg_retro_gap_when_inversion": avg_gap_inv,
        "num_inversions": int(inversions)
    }

    # Margin-gated inversion
    if 'inversion_margined' in df.columns:
        margined_inversions = df['inversion_margined'].sum()
        result['inversion_rate_margined'] = margined_inversions / len(stats)
        result['num_inversions_margined'] = int(margined_inversions)
    else:
        result['inversion_rate_margined'] = inversion_rate
        result['num_inversions_margined'] = int(inversions)

    # Margins and rank correlation
    if 'f_margin' in df.columns:
        result['avg_f_margin'] = float(df['f_margin'].mean())
    if 'r_margin' in df.columns:
        result['avg_r_margin'] = float(df['r_margin'].mean())
    if 'rank_corr' in df.columns:
        result['avg_rank_corr'] = float(df['rank_corr'].mean())

    # Depth-based analysis
    if 'depth' in df.columns:
        # Inversion rate by depth bin
        depth_bins = [0, 10, 20, 40, 60, 100, 200]
        depth_labels = ['0-9', '10-19', '20-39', '40-59', '60-99', '100+']
        df['depth_bin'] = pd.cut(df['depth'], bins=depth_bins, labels=depth_labels, right=False)
        depth_inv_rate = df.groupby('depth_bin', observed=True)['inversion'].mean().to_dict()
        depth_count = df.groupby('depth_bin', observed=True)['inversion'].count().to_dict()
        result['depth_inversion_rate'] = {str(k): float(v) for k, v in depth_inv_rate.items()}
        result['depth_sample_count'] = {str(k): int(v) for k, v in depth_count.items()}

        # Average depth where inversion occurs vs doesn't
        inv_df_d = df[df['inversion'] == True]
        non_inv_df_d = df[df['inversion'] == False]
        result['avg_depth_inversion'] = float(inv_df_d['depth'].mean()) if len(inv_df_d) > 0 else 0.0
        result['avg_depth_no_inversion'] = float(non_inv_df_d['depth'].mean()) if len(non_inv_df_d) > 0 else 0.0

    # Task stage analysis
    if 'task_stage' in df.columns:
        stage_inv_rate = df.groupby('task_stage')['inversion'].mean().to_dict()
        stage_count = df.groupby('task_stage')['inversion'].count().to_dict()
        stage_names = {0: 'find_key', 1: 'go_to_door', 2: 'go_to_goal'}
        result['stage_inversion_rate'] = {stage_names.get(k, str(k)): float(v) for k, v in stage_inv_rate.items()}
        result['stage_sample_count'] = {stage_names.get(k, str(k)): int(v) for k, v in stage_count.items()}

    return result
