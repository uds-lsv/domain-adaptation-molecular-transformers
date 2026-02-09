import warnings

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.stats import sem, t
from statsmodels.stats.libqsturng import psturng, qsturng
import pingouin as pg


def make_figure(n_datasets, n_cols, width=10, height_ratio=4):
    # Create a subplot grid
    n_rows = int(np.ceil(n_datasets / n_cols))  # Define number of rows (customized based on the specified number of columns)
    figure, subplots = plt.subplots(n_rows, n_cols, figsize=(width, height_ratio * n_rows))  # Create a grid of subplots
    if isinstance(subplots, np.ndarray):
        subplots = subplots.flatten()  # Flatten the axes for easy iteration
    return figure, subplots

def get_censored(col, censor_percent=5):
    if len(col) < 10:
        return col
    elif len(col) < 100:
        censor_percent = 20
    value_counts = col.value_counts(normalize=True) * 100

    # Identify values exceeding the threshold
    values_to_replace = value_counts[value_counts > censor_percent].index

    # Remove rows with these values
    return col.apply(lambda x: np.nan if x in values_to_replace else x)


def rm_tukey_hsd(df, metric, group_col, fold_col, alpha=0.05, sort=False, direction_dict=None):
    """
    Perform repeated measures Tukey HSD test on the given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric (str): The metric column name to perform the test on.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the test. Default is 0.05.
    sort (bool): Whether to sort the output tables. Default is False.

    Returns:
    tuple: A tuple containing:
        - result_tab (pd.DataFrame): DataFrame with pairwise comparisons and adjusted p-values.
        - df_means (pd.DataFrame): DataFrame with mean values for each group.
        - df_means_diff (pd.DataFrame): DataFrame with mean differences between groups.
        - pc (pd.DataFrame): DataFrame with adjusted p-values for pairwise comparisons.
    """
    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == 'maximize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=False)
        elif direction_dict[metric] == 'minimize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=True)
        else:
            raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
    else:
        df_means = df.groupby(group_col, observed=False).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero encountered in scalar divide')
        aov = pg.rm_anova(dv=metric, within=group_col, subject=fold_col , data=df, detailed=True, correction=True)
    mse = aov.loc[1, 'MS']
    df_resid = aov.loc[1, 'DF']

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons),
                              columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)

    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'].astype('string') + ' - ' + result_tab['group2'].astype('string')

    return result_tab, df_means, df_means_diff, pc


figure_extension = 'pdf'
sns.set_style('white')
fontsize = 24

ncols = 2
fig_width = 18
fig_height_ratio = 9


def make_figure(n_datasets, n_cols, width=10, height_ratio=4):
    # Create a subplot grid
    n_rows = int(
        np.ceil(n_datasets / n_cols))  # Define number of rows (customized based on the specified number of columns)
    figure, subplots = plt.subplots(n_rows, n_cols, figsize=(width, height_ratio * n_rows))  # Create a grid of subplots
    subplots = subplots.flatten()  # Flatten the axes for easy iteration
    return figure, subplots


def ttest_ci(sample, confidence=0.95):
    # Parameters
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_sem = sem(sample)  # Standard error of the mean

    # t-critical value for two-tailed CI
    t_crit = t.ppf((1 + confidence) / 2, df=n - 1)

    # Confidence interval
    margin = t_crit * sample_sem
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    return sample_mean, (ci_lower, ci_upper)


def format_ticks(value, tick_number=2):
    return f"{value:.2f}"


def plot_sns_heatmap(significance_mat, mean_diff_mat, fig_ax, apply_mask=False, higher_is_better=True):
    # Define thresholds and colors
    thresholds = [0, 0.001, 0.01, 0.05, 1]
    colors = ['#005a32', '#238b45', '#a1d99b', '#fbd7d4']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(thresholds, cmap.N)

    if apply_mask:
        mask = np.zeros_like(significance_mat, dtype=bool)
        np.fill_diagonal(mask, True)
    else:
        mask = None

    # Construct annotation matrix (text inside each cell)
    annot_mat = np.empty_like(significance_mat, dtype=object)
    for i in range(significance_mat.shape[0]):
        for j in range(significance_mat.shape[1]):
            if i == j:
                annot_mat[i, j] = ""
                continue
            p_val = significance_mat.iloc[i, j]
            if p_val < 0.05:
                diff = mean_diff_mat.iloc[i, j]
                is_row_better = (diff > 0) if higher_is_better else (diff < 0)
                arrow = '\n←' if is_row_better else '↓'
                annot_mat[i, j] = f"{p_val:.3f}{arrow}"
            else:
                annot_mat[i, j] = f"{p_val:.3f}"

    # Plot heatmap with annotation matrix
    heatmap = sns.heatmap(significance_mat, ax=fig_ax, cmap=cmap, norm=norm,
                          annot=annot_mat, fmt="",  # annotation already formatted
                          cbar_kws={'ticks': [0.0005, 0.005, 0.03, 0.5]}, annot_kws={'fontsize': 14},
                          linewidths=0.6, linecolor='black',
                          clip_on=False, mask=mask)

    # Customize colorbar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0.0005, 0.005, 0.03, 0.5])
    colorbar.set_ticklabels(['p < 0.001', 'p < 0.01', 'p < 0.05', 'NS'])
    colorbar.ax.tick_params(labelsize=14)
