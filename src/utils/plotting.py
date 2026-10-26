from tkinter import TRUE
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from configs.envs import CoordinationGame
from utils.saving import load_most_recent_experiment_results_for_env, load_all_experiment_results_for_env, get_all_available_envs_for_experiment
from utils.metrics import compute_xp_summary_statistics

from typing import Literal


from dataclasses import dataclass, asdict

@dataclass
class PlottingMetadata:
    epoch_indices: List[int]
    sp_means: List[float]
    sp_stds: List[float]
    sp_mins: List[float]
    xp_means: List[float]
    xp_stds: List[float]
    xp_mins: List[float]
    sp_std_errs: List[float]
    xp_std_errs: List[float]

def plot_xp_scores(
    xp_scores: np.ndarray, 
    include_legend: bool = True,
    color_range: Optional[Tuple[float, float]] = None, 
    agent_labels: Optional[List[str]] = None,
    include_title: bool = True,
    custom_title: Optional[str] = None,
    round_digits: int = 1,
    ) -> plt.Figure:
    # Calculate SP-scores (diagonal values)
    sp_scores = np.diag(xp_scores)
    sp_mean = np.mean(sp_scores)
    sp_std_err = np.std(sp_scores, ddof=1) / np.sqrt(len(sp_scores))
    
    # Calculate XP-scores (off-diagonal values)
    mask = ~np.eye(xp_scores.shape[0], dtype=bool)  # Create mask for off-diagonal
    xp_values = xp_scores[mask]
    xp_mean = np.mean(xp_values)
    xp_std_err = np.std(xp_values, ddof=1) / np.sqrt(len(xp_values))
    
    # Create the plot with limited color range
    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    if color_range is not None:
        vmin, vmax = color_range
        im = ax.imshow(xp_scores, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Color cells black if they are below vmin
        for i in range(xp_scores.shape[0]):
            for j in range(xp_scores.shape[1]):
                if xp_scores[i, j] < vmin:
                    # Add a black rectangle over this cell
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                             facecolor='black', edgecolor='none'))
    else:
        im = ax.imshow(xp_scores, cmap='viridis')
    
    # Add text annotations showing the actual values
    for i in range(xp_scores.shape[0]):
        for j in range(xp_scores.shape[1]):
            value = xp_scores[i, j]
            # Use white text for dark cells, black text for light cells
            # Always use white text for cells that are colored black (below vmin)
            if color_range is not None and value < color_range[0]:
                text_color = 'white'
            else:
                text_color = 'white' if value < (xp_scores.min() + xp_scores.max()) / 2 else 'black'
            ax.text(j, i, f'{value:.{round_digits}f}', ha='center', va='center', 
                   color=text_color, fontsize=12, fontweight='bold')
    
    if include_legend:
        # Add colorbar with label
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('XP-scores', rotation=270, labelpad=20)
    
    # Set title with statistics
    if include_title:
        if custom_title is not None:
            ax.set_title(custom_title, fontsize=18, pad=10)
        else:
            ax.set_title(f'SP-scores: {sp_mean:.2f} ± {sp_std_err:.2f} | '
                     f'XP-scores: {xp_mean:.2f} ± {xp_std_err:.2f}',
                 fontsize=18, pad=10)



    if agent_labels is not None:
        ax.set_xticks(range(len(agent_labels)))
        ax.set_yticks(range(len(agent_labels)))
        ax.set_xticklabels(agent_labels, rotation=45, ha='right')
        ax.set_yticklabels(agent_labels)
    else:
        ax.set_xticks(range(xp_scores.shape[1]))
        ax.set_yticks(range(xp_scores.shape[0]))
    
    # Always set tick parameters to ensure fontsize is 18 regardless of agent labels
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_xlabel('Agent 1', fontsize=18)
    ax.set_ylabel('Agent 0', fontsize=18)
    
    fig.tight_layout()
    
    # Close the figure to prevent automatic display in Jupyter
    plt.close(fig)
    
    return fig


def load_plotting_meatadata(env_name_file_safe: str, experiment_name: str, use_epoch_indices: bool = True):
        # Load the starting point
    experiments = [load_most_recent_experiment_results_for_env(f"coordination_failure_starting_point", mode="code", env_name_file_safe=env_name_file_safe)]

    # Load the meta-loop experiments
    experiments.extend(load_all_experiment_results_for_env(experiment_name, mode="code", env_name_file_safe=env_name_file_safe))

    xp_summary_statistics = [compute_xp_summary_statistics(experiment.xp_matrix) for experiment in experiments]

    # Extract data for plotting
    # iterations = list(range(1, len(xp_summary_statistics) + 1))
    if use_epoch_indices:
        epoch_indices = [0] + [experiment.additional_info["epoch_index"] for experiment in experiments[1:]]
    else:
        epoch_indices = list(range(1, len(xp_summary_statistics) + 1))
        
    sp_means = [stats.sp_scores_mean for stats in xp_summary_statistics]
    sp_stds = [stats.sp_scores_std for stats in xp_summary_statistics]
    sp_mins = [stats.sp_scores_min for stats in xp_summary_statistics]
    xp_means = [stats.xp_scores_mean for stats in xp_summary_statistics]
    xp_stds = [stats.xp_scores_std for stats in xp_summary_statistics]
    xp_mins = [stats.xp_scores_min for stats in xp_summary_statistics]
    sp_std_errs = [stats.sp_scores_std_err for stats in xp_summary_statistics]
    xp_std_errs = [stats.xp_scores_std_err for stats in xp_summary_statistics]

    return PlottingMetadata(epoch_indices, sp_means, sp_stds, sp_mins, xp_means, xp_stds, xp_mins, sp_std_errs, xp_std_errs)


def plot_all_envs_one_metric_meta_loop_scores(
    experiment_name: str,
    metric: Literal["sp", "xp"],
    plot_std_err: bool = False,
    normalize_jointly: bool = True,
    ):
    # Get all the available environments
    available_envs = get_all_available_envs_for_experiment(experiment_name, "code")

    results_dict = {}

    metric_dict = {}
    episode_indices = {}
    std_err_dict = {}
    min_dict = {}

    for env_name_file_safe in available_envs:
        results_dict[env_name_file_safe] = load_plotting_meatadata(env_name_file_safe, experiment_name, use_epoch_indices=True)

    # Normalize the scores
    for env in available_envs:
        if normalize_jointly:
            min_val = min(np.array(results_dict[env].sp_means).min(), np.array(results_dict[env].xp_means).min())
            max_val = max(np.array(results_dict[env].sp_means).max(), np.array(results_dict[env].xp_means).max())
        else:
            min_val = np.array(asdict(results_dict[env]).get(f"{metric}_means")).min()
            max_val = np.array(asdict(results_dict[env]).get(f"{metric}_means")).max()

        metric_dict[env] = (np.array(asdict(results_dict[env]).get(f"{metric}_means")) - min_val) / (max_val - min_val)
        episode_indices[env] = asdict(results_dict[env]).get("epoch_indices")
        std_err_dict[env] = asdict(results_dict[env]).get(f"{metric}_std_errs") / (max_val - min_val)
        min_dict[env] = (np.array(asdict(results_dict[env]).get(f"{metric}_mins")) - min_val) / (max_val - min_val)

    max_episode_index = max([episode_indices[env][-1] for env in available_envs])

    for env in available_envs:
        if episode_indices[env][-1] != max_episode_index:
            metric_dict[env] = np.append(metric_dict[env], metric_dict[env][-1])
            std_err_dict[env] = np.append(std_err_dict[env], std_err_dict[env][-1])
            min_dict[env] = np.append(min_dict[env], min_dict[env][-1])
            episode_indices[env].append(max_episode_index)
    

    print(metric_dict)

    def break_long_strings(string: str, max_length: int = 18) -> str:
        if len(string) > max_length:
            # look for first underscore after max_length
            idx = string.find("_", max_length)
            if idx != -1:  # found underscore
                return string[:idx] + '\n' + break_long_strings(string[idx+1:], max_length)
            else:  # no underscore found, just hard break
                return string[:max_length] + '\n' + break_long_strings(string[max_length:], max_length)
        else:
            return string

    
    # Plot the scores
    plt.figure(figsize=(10, 6), dpi=600)
    
    # Use traditional matplotlib colors (blue, orange, red, green, etc.)
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [default_colors[i % len(default_colors)] for i in range(len(available_envs))]
    
    for i, env in enumerate(available_envs):
        color = colors[i]
        
        # Wrap long environment names for legend
        wrapped_env = break_long_strings(env)
        
        # Plot the mean line
        plt.plot(episode_indices[env], metric_dict[env], label=wrapped_env, linewidth=2, color=color)
        # Plot the min line as dotted with same color (no label for legend)
        plt.plot(episode_indices[env], min_dict[env], linestyle='--', alpha=0.7, color=color)
        # Add shading manually
        if plot_std_err:
            plt.fill_between(episode_indices[env], metric_dict[env] - std_err_dict[env], metric_dict[env] + std_err_dict[env], alpha=0.2, color=color)

    # Set axis labels with appropriate metric type
    metric_label = "SP" if metric == "sp" else "XP"
    plt.xlabel('Step', fontsize=20)
    plt.ylabel(f'Normalised {metric_label} scores', fontsize=20)
    
    # Set tick parameters for fontsize
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Force x-axis to show only integers
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    #plt.ylim(0, 1)
    plt.legend(fontsize=16)
    plt.show()



def lineplot_meta_loop_scores(env_name_file_safe: str, experiment_name: str, use_epoch_indices: bool = False):
    """
    Plots the SP and XP scores across the different iterations of the meta-loop. Currently we always assume that a starting point was used.

    Args:
        game: The specific coordination game that was played.
        experiment_suffix: The suffix of the experiment under which it was saved (This should be some kind of timestamp e.g. 20250801_090450).
    """

    plotting_metadata = load_plotting_meatadata(env_name_file_safe, experiment_name, use_epoch_indices)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot SP scores with error bars
    plt.errorbar(plotting_metadata.epoch_indices, plotting_metadata.sp_means, yerr=plotting_metadata.sp_stds, 
                label='SP Scores (Mean)', marker='o', capsize=5, capthick=2, color='#1f77b4')
    
    # Plot XP scores with error bars
    plt.errorbar(plotting_metadata.epoch_indices, plotting_metadata.xp_means, yerr=plotting_metadata.xp_stds, 
                label='XP Scores (Mean)', marker='s', capsize=5, capthick=2, color='orange')
    
    # Plot SP minimum values as dotted line
    plt.plot(plotting_metadata.epoch_indices, plotting_metadata.sp_mins, 
             label='SP Scores (Min)', linestyle='--', marker='o', alpha=0.7, color='#1f77b4')
    
    # Plot XP minimum values as dotted line
    plt.plot(plotting_metadata.epoch_indices, plotting_metadata.xp_mins, 
             label='XP Scores (Min)', linestyle='--', marker='s', alpha=0.7, color='orange')
    
    # Customize the plot
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f'SP and XP Scores Across Meta-Loop Iterations\n({env_name_file_safe})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":

    #/workspaces/zsc_llm/data/code/all_games_but_permuted_levers_gpt5+mini
    
    # Original line plot
    #lineplot_meta_loop_scores("cat_and_dog", "all_games_but_permuted_levers_gpt5+mini", use_epoch_indices=True)

    plot_all_envs_one_metric_meta_loop_scores("navigation_games__meta_training", "sp", plot_std_err=True, normalize_jointly=True)

    # Combined ridgeline plot (SP and XP on separate rows)
    #fig2 = plot_combined_ridgeline_meta_loop_scores(game, experiment_suffix)
    #fig2.show()
    
    # Overlapping ridgeline plot (SP and XP on same height, overlapping)
    
    # Other example experiments (commented out)
    #plot_meta_loop_scores(CoordinationGame.AD_HOC_CONVENTIONS, "20250801_094442")
    #plot_meta_loop_scores(CoordinationGame.CAT_AND_DOG, "20250801_153744")
    #plot_meta_loop_scores(CoordinationGame.BRITTLE_BRIDGE, "20250802_103210")
    #plot_meta_loop_scores(CoordinationGame.AD_HOC_CONVENTIONS, "20250801_151750")
