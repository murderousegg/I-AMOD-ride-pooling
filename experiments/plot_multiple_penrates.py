import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.fontsize": 12,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.labelsize": 13,
})

mode_styles = {
    "totCost": {'linestyle': '-', 'label': 'avg all users'},
    "IAMoDCosts": {'linestyle': '--', 'label': 'I-AMoD'},
    "privateCosts": {'linestyle': ':', 'label': 'Private'},
}

flow_order = ["private_flow", "rp_flow", "pt_flow", "bike_flow", "ped_flow"]
flow_colors = {
    "private_flow": "tab:purple",
    "rp_flow": "tab:blue",
    "pt_flow": "tab:orange",
    "bike_flow": "tab:green",
    "ped_flow": "tab:red",
}

flow_labels = {
    "private_flow": "Private",
    "rp_flow": "Ride-pooling",
    "pt_flow": "Public Transit",
    "bike_flow": "Biking",
    "ped_flow": "Pedestrian"
}

def plot_penrate_comparison_grid():
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 9), sharex=False, gridspec_kw={'width_ratios': [1, 1.8]})
    plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Add vertical padding


    phi_values = [1, 2, 3]
    file_paths = [
        "results/penrate/penrate_15000_1/penrate_metrics_0.99.csv",
        "results/penrate/penrate_30000_2/penrate_metrics_0.99.csv",
        "results/penrate/penrate_45000_3/penrate_metrics_0.99.csv",
    ]

    for row_idx, (phi, path) in enumerate(zip(phi_values, file_paths)):
        df = pd.read_csv(path)

        # -- Line plot (left column) --
        ax_line = axs[row_idx, 0]
        x = np.linspace(0.01, 0.99, len(df["totCost"]))

        for col, style in mode_styles.items():
            ax_line.plot(x, [v * 60 for v in df[col]], 
                         label=style['label'], linestyle=style['linestyle'], linewidth=2)

        ax_line.set_title(fr"$\phi = {phi}$", fontsize=15)
        if row_idx == 2:
            ax_line.set_xlabel(r"Penetration Rate (\%)")
        ax_line.set_ylabel(r"Avg Travel Time ($\mathrm{min}$)")
        ax_line.grid(alpha=0.3)
        if row_idx == 0:
            ax_line.legend(loc='upper right')

        # -- Stacked bar plot (right column) --
        ax_bar = axs[row_idx, 1]
        x_vals = np.linspace(0.01, 0.99, len(df))  # same as in line plot
        bottom = np.zeros(len(df))
        
        for flow_type in flow_order:
            heights = df[flow_type].values/10000
            ax_bar.bar(x_vals, heights, bottom=bottom, \
                       color=flow_colors[flow_type], label=flow_labels[flow_type] if row_idx == 0 else "",\
                          edgecolor='black', linewidth=0.5, width = 0.2)
            bottom += heights

        ax_bar.set_xlim(-0.1, 1.1)
        if row_idx == 2:
            ax_bar.set_xlabel("Penetration Rate (%)")
        ax_bar.set_ylabel(r"Modal share ($\times10^4$ $\mathrm{h}$)")
        ax_bar.set_title(fr"$\phi = {phi}$", fontsize=15)
        if row_idx == 0:
            ax_bar.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig("results/penrate/penRate_3x2_subplots.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    plot_penrate_comparison_grid()
