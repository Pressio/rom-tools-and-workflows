"""
Visualization script for hybrid MFUQ results.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def plot_hybrid_mfuq_results(work_directory, save_plots=True, show_plots=True):
    """
    Create comprehensive visualization of hybrid MFUQ results.
    
    Args:
        work_directory: Path to working directory containing visualization_data.npz
        save_plots: Whether to save plots to disk
        show_plots: Whether to display plots interactively
    """
    # Load data
    data_path = os.path.join(work_directory, 'visualization_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}")
    
    data = np.load(data_path)
    
    # Debug: print all keys
    print("Available data keys:")
    for key in sorted(data.keys()):
        print(f"  {key}: shape = {data[key].shape}, dtype = {data[key].dtype}")
    print()
    
    # Extract basic info
    n_aux = int(data['n_aux'])
    s_star = data['s_star']
    rom_basis_optimal = int(round(s_star[-1]))
    
    # Create output directory for plots
    plot_dir = os.path.join(work_directory, 'plots')
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Create plots
    print("Creating correlation and cost plots...")
    fig1 = plot_all_correlations(data, n_aux)
    if save_plots:
        fig1.savefig(os.path.join(plot_dir, 'all_correlations.png'), 
                    dpi=300, bbox_inches='tight')
    
    print("Creating cost functions plot...")
    fig2 = plot_all_costs(data, n_aux)
    if save_plots:
        fig2.savefig(os.path.join(plot_dir, 'all_costs.png'),
                    dpi=300, bbox_inches='tight')
    
    print("Creating variance reduction plots...")
    fig3 = plot_variance_reduction(data, n_aux)
    if save_plots:
        fig3.savefig(os.path.join(plot_dir, 'variance_reduction.png'),
                    dpi=300, bbox_inches='tight')
    
    print("Creating optimal allocation summary...")
    fig4 = plot_optimal_allocation(data, n_aux, rom_basis_optimal)
    if save_plots:
        fig4.savefig(os.path.join(plot_dir, 'optimal_allocation.png'),
                    dpi=300, bbox_inches='tight')
    
    if save_plots:
        print(f"\nPlots saved to {plot_dir}/")
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')


def plot_all_correlations(data, n_aux):
    """Plot all correlation functions on a single figure."""
    ss = data['ss'][0]  # ROM basis sizes
    pp = data['pp'][0]  # Pilot basis sizes
    
    # Determine number of correlation types
    n_fom_aux = n_aux  # FOM-aux correlations
    n_aux_aux = n_aux * (n_aux - 1) // 2 if n_aux > 1 else 0  # aux-aux pairs
    n_fom_rom = 1  # FOM-ROM
    n_aux_rom = n_aux  # aux-ROM
    
    total_corrs = n_fom_aux + n_aux_aux + n_fom_rom + n_aux_rom
    
    # Create figure with subplots
    n_cols = 3
    n_rows = (total_corrs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4.5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Color scheme
    color_constant = 'purple'
    color_variable = 'blue'
    
    # Plot FOM-aux correlations (constant)
    for i in range(n_aux):
        ax = axes[plot_idx]
        
        # Get surrogate values
        rho_vals = data[f'rho_fom_aux{i}_vals']
        
        # Get pilot value (scalar)
        rho_pilot_val = float(data[f'rho_fom_aux{i}_pilot'])
        
        # Plot
        ax.plot(ss, rho_vals, color=color_constant, linewidth=2.5, label='Surrogate')
        ax.scatter(pp, np.full_like(pp, rho_pilot_val, dtype=float), 
                  color='red', s=100, marker='o', zorder=5, label='Pilot Data', edgecolors='darkred', linewidths=1.5)
        ax.axhline(y=rho_pilot_val, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        
        ax.set_xlabel('ROM Basis Size', fontsize=11)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_title(f'ρ(FOM, Aux{i}) [Constant]', fontsize=12, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        plot_idx += 1
    
    # Plot aux-aux correlations (constant, if n_aux > 1)
    if n_aux > 1:
        for i in range(n_aux):
            for j in range(i):
                ax = axes[plot_idx]
                
                # Get surrogate values
                rho_vals = data[f'rho_aux{j}_aux{i}_vals']
                
                # Get pilot value (scalar)
                rho_pilot_val = float(data[f'rho_aux{j}_aux{i}_pilot'])
                
                # Plot
                ax.plot(ss, rho_vals, color=color_constant, linewidth=2.5, label='Surrogate')
                ax.scatter(pp, np.full_like(pp, rho_pilot_val, dtype=float),
                          color='red', s=100, marker='o', zorder=5, label='Pilot Data', edgecolors='darkred', linewidths=1.5)
                ax.axhline(y=rho_pilot_val, color='gray', linestyle='--', alpha=0.4, linewidth=1)
                
                ax.set_xlabel('ROM Basis Size', fontsize=11)
                ax.set_ylabel('Correlation', fontsize=11)
                ax.set_title(f'ρ(Aux{j}, Aux{i}) [Constant]', fontsize=12, fontweight='bold')
                ax.set_ylim([-0.05, 1.05])
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                
                plot_idx += 1
    
    # Plot FOM-ROM correlation (variable)
    ax = axes[plot_idx]
    
    # Get surrogate values
    rho_fom_rom = data['rho_fom_rom_vals']
    
    # Get pilot values (array)
    rho_fom_rom_pilot = data['fom_rom_corrs_pilot']
    
    # Plot
    ax.plot(ss, rho_fom_rom, color=color_variable, linewidth=2.5, label='Surrogate')
    ax.scatter(pp, rho_fom_rom_pilot, color='red', s=100, marker='o', zorder=5, 
              label='Pilot Data', edgecolors='darkred', linewidths=1.5)
    
    ax.set_xlabel('ROM Basis Size', fontsize=11)
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('ρ(FOM, ROM) [Variable]', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plot_idx += 1
    
    # Plot aux-ROM correlations (variable)
    for i in range(n_aux):
        ax = axes[plot_idx]
        
        # Get surrogate values
        rho_vals = data[f'rho_aux{i}_rom_vals']
        
        # Get pilot values (array)
        rho_pilot = data[f'rho_aux{i}_rom_pilot']
        
        # Plot
        ax.plot(ss, rho_vals, color=color_variable, linewidth=2.5, label='Surrogate')
        ax.scatter(pp, rho_pilot, color='red', s=100, marker='o', zorder=5,
                  label='Pilot Data', edgecolors='darkred', linewidths=1.5)
        
        ax.set_xlabel('ROM Basis Size', fontsize=11)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_title(f'ρ(Aux{i}, ROM) [Variable]', fontsize=12, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        plot_idx += 1
    
    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle('Correlation Functions: Surrogate vs Pilot Data', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def plot_all_costs(data, n_aux):
    """Plot all cost functions on a single figure."""
    ss = data['ss'][0]  # ROM basis sizes
    pp = data['pp'][0]  # Pilot basis sizes
    
    # Number of cost functions
    n_costs = n_aux + 1  # aux costs + ROM cost
    
    # Create figure
    fig, axes = plt.subplots(1, n_costs, figsize=(6*n_costs, 5))
    if n_costs == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Color scheme
    color_constant = 'green'
    color_variable = 'darkgreen'
    
    # Plot auxiliary costs (constant)
    for i in range(n_aux):
        ax = axes[plot_idx]
        
        # Get surrogate values
        cost_vals = data[f'cost_aux{i}_vals']
        
        # Get pilot value (scalar)
        cost_pilot_val = float(data[f'cost_aux{i}_pilot'])
        
        # Plot
        ax.plot(ss, cost_vals, color=color_constant, linewidth=2.5, label='Surrogate')
        ax.scatter(pp, np.full_like(pp, cost_pilot_val, dtype=float),
                  color='red', s=100, marker='o', zorder=5, label='Pilot Data', 
                  edgecolors='darkred', linewidths=1.5)
        ax.axhline(y=cost_pilot_val, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        
        ax.set_xlabel('ROM Basis Size', fontsize=11)
        ax.set_ylabel('Relative Cost', fontsize=11)
        ax.set_title(f'Cost(Aux{i}) [Constant]', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(cost_vals) * 1.15])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        plot_idx += 1
    
    # Plot ROM cost (variable)
    ax = axes[plot_idx]
    
    # Get surrogate values
    cost_rom = data['cost_rom_vals']
    
    # Get pilot values (array)
    cost_rom_pilot = data['normalized_rom_times_pilot']
    
    # Plot
    ax.plot(ss, cost_rom, color=color_variable, linewidth=2.5, label='Surrogate')
    ax.scatter(pp, cost_rom_pilot, color='red', s=100, marker='o', zorder=5,
              label='Pilot Data', edgecolors='darkred', linewidths=1.5)
    
    ax.set_xlabel('ROM Basis Size', fontsize=11)
    ax.set_ylabel('Relative Cost', fontsize=11)
    ax.set_title('Cost(ROM) [Variable]', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(cost_rom) * 1.15])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    # Add overall title
    fig.suptitle('Cost Functions: Surrogate vs Pilot Data',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def plot_variance_reduction(data, n_aux):
    """Plot variance reduction as a function of budget."""
    xx = data['xx']  # Budget values
    
    fMFs = data['fMFs']  # MF variance ratios (surrogate)
    fMFs_ex = data['fMFs_ex']  # MF variance ratios (exact)
    fISs = data['fISs']  # IS variance ratios (surrogate)
    fISs_ex = data['fISs_ex']  # IS variance ratios (exact)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MF plot
    ax = axes[0]
    ax.plot(xx, fMFs, 'b-o', linewidth=2, markersize=8, label='Surrogate', alpha=0.7)
    ax.plot(xx, fMFs_ex, 'r--s', linewidth=2, markersize=8, label='Exact (Trained ROM)', alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Standard MC')
    
    ax.set_xlabel('Computational Budget (HF equivalent)', fontsize=12)
    ax.set_ylabel('Variance Ratio', fontsize=12)
    ax.set_title('Multifidelity (MF) Variance Reduction', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    # IS plot
    ax = axes[1]
    ax.plot(xx, fISs, 'b-o', linewidth=2, markersize=8, label='Surrogate', alpha=0.7)
    ax.plot(xx, fISs_ex, 'r--s', linewidth=2, markersize=8, label='Exact (Trained ROM)', alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5, label='Standard MC')
    
    ax.set_xlabel('Computational Budget (HF equivalent)', fontsize=12)
    ax.set_ylabel('Variance Ratio', fontsize=12)
    ax.set_title('Importance Sampling (IS) Variance Reduction', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_optimal_allocation(data, n_aux, rom_basis_optimal):
    """Plot optimal allocation details."""
    s_star = data['s_star']
    xx = data['xx']
    
    fISs = data['fISs']
    fISs_ex = data['fISs_ex']
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Allocation summary text
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    
    allocation_text = "Optimal Allocation (at minimum budget)\n"
    allocation_text += "=" * 60 + "\n\n"
    allocation_text += f"N (HF samples):                {s_star[0]:.2f}\n"
    for i in range(n_aux):
        allocation_text += f"r_aux{i} (oversample ratio):      {s_star[i+1]:.3f}\n"
    allocation_text += f"r_ROM (oversample ratio):      {s_star[n_aux+1]:.3f}\n"
    allocation_text += f"s_ROM (basis size):            {rom_basis_optimal}\n\n"
    
    # Compute effective samples
    N = s_star[0]
    allocation_text += "Effective Sample Counts:\n"
    allocation_text += f"  HF samples:                  {N:.1f}\n"
    for i in range(n_aux):
        allocation_text += f"  Aux{i} samples:                 {N * s_star[i+1]:.1f}\n"
    allocation_text += f"  ROM samples:                 {N * s_star[n_aux+1]:.1f}\n"
    
    ax_text.text(0.1, 0.5, allocation_text, fontsize=12, family='monospace',
                verticalalignment='center')
    
    # Variance reduction comparison
    ax1 = fig.add_subplot(gs[1, :])
    
    width = 0.35
    x_pos = np.arange(len(xx))
    
    ax1.bar(x_pos - width/2, fISs, width, label='Surrogate', alpha=0.7, color='blue')
    ax1.bar(x_pos + width/2, fISs_ex, width, label='Exact', alpha=0.7, color='red')
    
    ax1.set_xlabel('Budget Level', fontsize=12)
    ax1.set_ylabel('Variance Ratio (IS)', fontsize=12)
    ax1.set_title('Variance Reduction Comparison: Surrogate vs Exact', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{b:.0f}' for b in xx])
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Speedup factors
    ax2 = fig.add_subplot(gs[2, 0])
    
    speedup_surrogate = 1.0 / np.array(fISs)
    speedup_exact = 1.0 / np.array(fISs_ex)
    
    ax2.plot(xx, speedup_surrogate, 'b-o', linewidth=2, markersize=8, 
            label='Surrogate', alpha=0.7)
    ax2.plot(xx, speedup_exact, 'r--s', linewidth=2, markersize=8,
            label='Exact', alpha=0.7)
    ax2.set_xlabel('Computational Budget', fontsize=12)
    ax2.set_ylabel('Variance Reduction Factor', fontsize=12)
    ax2.set_title('Speedup vs Standard MC', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Relative error
    ax3 = fig.add_subplot(gs[2, 1])
    
    relative_error = np.abs(fISs - fISs_ex) / fISs_ex * 100
    
    ax3.plot(xx, relative_error, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Computational Budget', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    ax3.set_title('Surrogate Error vs Exact', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax3.legend()
    
    return fig


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        work_dir = sys.argv[1]
    else:
        work_dir = "./work"  # Default directory
    
    plot_hybrid_mfuq_results(work_dir, save_plots=True, show_plots=True)