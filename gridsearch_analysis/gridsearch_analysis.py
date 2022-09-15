import pandas as pd
import numpy as np

# making plots look good
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.serif'] = 'Arial'
matplotlib.rcParams.update({'font.size': 14, 'legend.handleheight':1, 'hatch.linewidth': 1.0,
                           'lines.markersize':4, 'lines.linewidth':1.5,'xtick.labelsize':14})

cm = 1/2.54
H = 14.56
W = 9


METRIC_NAMES = {
    # "ValidationMetric": "(mae(snr)+mae(c50)+(1-fscore(vad))/3",
    "ValidationMetric": r'\frac{(1-fscore(VAD))+NMAE(SNR)+NMAE(C50)}{3}',
    "c50ValMetric": "C50 MAE",
    "snrValMetric": "SNR MAE",
    "vadValMetric": "VAD Fscore"
}
METRICS = [
    "ValidationMetric",
    "c50ValMetric",
    "snrValMetric",
    "vadValMetric"
]
PARAMS = [
    "dropout",
    "duration",
    "batch_size",
    "hidden_size",
    "num_layers"
]
PARAMS_NAMES = {
    "dropout": "Dropout",
    "duration": "Duration (s)",
    "batch_size": "Batch size",
    "hidden_size": "LSTM Hidden size",
    "num_layers": "Number of LSTM layers"
}

# load csv with all the data
data = pd.read_csv("gridsearch_models_data.csv", index_col=0)
data.head()


def scattered_boxplot(
    data: pd.DataFrame,
    param_to_monitor: str = "dropout",
    metric: str = 'ValidationMetric',
    figure_path: str = None
) -> None:
    
    possible_values = np.sort(data[param_to_monitor].unique())
    if param_to_monitor == "vadValMetric":
        coef = 100
    else:
        coef = 1

    values = dict()
    for value in possible_values:
        values[value] = data[data[param_to_monitor] == value][metric].reset_index(drop=True) * coef
    # Rearrange the data for plotting
    df = pd.DataFrame(values)

    default_arch = data[data["name"] == 'dur_2_bs_32_lstm_hs_128_lstm_nl_2_dropout_0']
    val_default_arch = default_arch[metric] * coef

    vals, names, xs = [],[],[]
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))
        # adds jitter to the data points - can be adjusted
    
    fig, ax = plt.subplots(1,1, figsize=(H*cm,W*cm), constrained_layout=True)
    ax.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    markers = ['.', '^', 'x', '*']
    for x, val, c, m in zip(xs, vals, palette, markers):
        ax.scatter(x, val, alpha=0.4, color=c, marker=m, s=60)

    # y axis
    plt.xlabel(param_to_monitor.replace('_', ' ').capitalize(), fontweight='normal', fontsize=14)
    plt.ylabel(METRIC_NAMES[metric], fontweight='normal', fontsize=14)

    # default architecture
    ax.axhline(y=float(val_default_arch), color='k', linestyle='--', alpha=0.7, linewidth=3, label='Default arch.')
    ax.legend(bbox_to_anchor=(0.31, 1.15), loc=2, borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    if not figure_path:
        figure_path = f"figures/{param_to_monitor}_{metric}.png"
    plt.savefig(figure_path, bbox_inches="tight")


scattered_boxplot(data, metric="ValidationMetric")