import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import List
from PIL import Image
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.font_manager as fm
from matplotlib import gridspec, ticker
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from typing import Tuple

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def plot_single_dataset_pdf(
    data:pd.DataFrame,
    x_label:str,
    y_label:str,
    density:bool,
    output_path:str,
    color="#2178B5",
    figsize=(10, 8),
    x_axis_st:str='',
    y_axis_st:str='',
    xlim=None
    ):
    # Create figure with fixed size using add_axes for precise control
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.13, 0.1, 0.85, 0.85])
    
    # Plot histogram
    clean_data = data.dropna()
    ax.hist(clean_data, bins=100, density=density, color=color, edgecolor='black', linewidth=0.5)
    
    # Set x-axis tick labels
    if x_axis_st=='int':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    elif x_axis_st=='float':
        x_ticks = ax.get_xticks()
        x_maxv  = np.max(np.abs(x_ticks)) if len(x_ticks) else 0.0
        x_exp   = 0 if x_maxv==0 else int(np.floor(np.log10(x_maxv)))
        x_exp   = 0 if -1 <= x_exp <= 1 else x_exp
        x_scale = 10.0 ** x_exp
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/x_scale:.2f}"))
        ax.yaxis.offsetText.set_visible(False)
        if x_exp != 0:
            ax.text(-0.10, 1.01, rf"($\times 10^{{{x_exp}}}$)", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    
    # Set y-axis tick labels
    if y_axis_st=='int':
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
    elif y_axis_st=='float':
        y_ticks = ax.get_yticks()
        y_maxv  = np.max(np.abs(y_ticks)) if len(y_ticks) else 0.0
        y_exp   = 0 if y_maxv==0 else int(np.floor(np.log10(y_maxv)))
        y_exp   = 0 if -1 <= y_exp <= 1 else y_exp
        y_scale = 10.0 ** y_exp
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y/y_scale:.2f}"))
        ax.yaxis.offsetText.set_visible(False)
        if y_exp != 0:
            ax.text(-0.10, 1.01, rf"($\times 10^{{{y_exp}}}$)", transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_xlabel(x_label, fontsize=24, labelpad=10)
    ax.set_ylabel(y_label, fontsize=24, labelpad=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=4, direction='out', colors='black')
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    
def create_scatter_plot(
    df:pd.DataFrame,
    x_col:str,
    y_col:str,
    output_path:str,
    title:str=None,
    add_diagonal:bool=True,
    show_corr:bool=True,
    figsize:Tuple[int, int]=(10, 8),
    xlim:Tuple[int, int]=None,
    ylim:Tuple[int, int]=None
    ):
    
    # Create figure with fixed size using add_axes for precise control
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.13, 0.1, 0.85, 0.85])

    # Create scatter plot with larger, darker points
    ax.scatter(df[x_col], df[y_col], alpha=0.8, s=20, color='#1f77b4')

    if add_diagonal:
        lims = [min(df[x_col].min(), df[y_col].min()),
                max(df[x_col].max(), df[y_col].max())]
        ax.plot(lims, lims, 'r--', alpha=0.3)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set labels with consistent font size
    ax.set_xlabel(x_col, fontsize=24, labelpad=10)
    ax.set_ylabel(y_col, fontsize=24, labelpad=10)

    if title:
        ax.set_title(title, fontsize=22, pad=20)

    if show_corr:
        corr = df[x_col].corr(df[y_col])
        ax.text(0.95, 0.05, f'R={corr:.2f}', transform=ax.transAxes, va='bottom', ha='right', fontsize=20)

    # Style consistency with plot_single_dataset_pdf
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=4, direction='out', colors='black')
    ax.grid(True, alpha=0.3)

    # Save with consistent DPI
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def create_boxplot(
    df:pd.DataFrame,
    x_col:str,
    y_col:str,
    x_name:str,
    y_name:str,
    x_lim:List[float],
    y_lim:List[float],
    save_path:str,
    hue:str=None,
    figsize=(10, 8)
    ):
    # Create figure with fixed size using add_axes for precise control
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.13, 0.1, 0.85, 0.87])
    
    # Prepare data for box plot
    positions, data_to_plot, colors = [], [], []
    
    if hue is None:
        for attach, sub_attach in df.groupby(x_col):
            vals = sub_attach[y_col].dropna()
            if not vals.empty:
                positions.append(attach)
                data_to_plot.append(vals)
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.12, patch_artist=True, showfliers=False)
        
        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels([int(p) for p in positions])
        
    else:
        # Save the center position of each max_point_att_num
        group_centers = []  
        for attach, sub_attach in df.groupby(x_col):
            mean_val = sub_attach[hue].mean()
            group_positions = []  # This group's box positions
            for frags, sub_frags in sub_attach.groupby(hue):
                vals = sub_frags[y_col].dropna()
                if not vals.empty:
                    pos = attach + (frags - mean_val) * 0.1
                    positions.append(pos)
                    group_positions.append(pos)
                    data_to_plot.append(vals)
                    colors.append(frags)
            # Calculate the center position of the boxes in this group
            if group_positions:
                group_centers.append((attach, np.mean(group_positions)))
        
        norm = plt.cm.colors.Normalize(vmin=df[hue].min(), vmax=df[hue].max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.1, patch_artist=True, showfliers=False)
        
        for patch, color_val in zip(bp['boxes'], colors):
            patch.set_facecolor(sm.to_rgba(color_val))
            patch.set_alpha(0.7)
        
        # One x-axis tick per max_point_att_num group (at the center position)
        ax.set_xticks([center for _, center in group_centers])
        ax.set_xticklabels([int(attach) for attach, _ in group_centers])
    
    # Adjust median line style
    for median in bp['medians']:
        median.set_linewidth(3)
        median.set_color('red')
        
    # Remove grid
    ax.grid(False)
    
    # Set labels with consistent font size
    ax.set_xlabel(x_name, fontsize=24, labelpad=10)
    ax.set_ylabel(y_name, fontsize=24, labelpad=10)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
        
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
        
    ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=4, direction='out', colors='black')
    plt.savefig(save_path, dpi=300)
    plt.show()