import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_plot_mean_rmse(csv_path, name_model="SVD1", imputing='fill with 0'):
    """
    Draws a line plot of mean RMSE vs r_component.

    Parameters:
    csv_path (str): Path to the CSV file containing 'r_component' and 'mean_rmse' columns.
    """
    df_mean = pd.read_csv(csv_path)
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_mean, x='r_component', y='mean_rmse', hue="lmb", marker='o', linewidth=2.5, palette="tab10")

    plt.title(f'Method of imputation: {imputing}', fontsize=12)
    plt.suptitle(f'Mean RMSE vs Number of Components for {name_model}', y=0.95, fontsize=18)
    plt.xlabel('Number of Components (r_component)', fontsize=14)
    plt.ylabel('Mean RMSE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def draw_plot_mean_rmse_SGD_lambda(df):
    """
    Draws a line plot of mean RMSE vs r_component.

    Parameters:
    csv_path (str): Path to the CSV file containing 'r_component' and 'mean_rmse' columns.
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='r_component', y='mean_rmse', hue="lmb", marker='o', linewidth=2.5, palette="colorblind")

    # plt.title(r'Mean RMSE of SGD model in dependence on choice of parameter $\lambda$ and $r$', fontsize=14)
    plt.suptitle(r'Mean RMSE of SGD model in dependence on choice of parameter $\lambda$ and $r$', fontsize=18)
    plt.xlabel('Number of Components (r_component)', fontsize=14)
    plt.ylabel('Mean RMSE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(title=r"$\lambda$")
    plt.tight_layout()
    plt.show()


def draw_boxplots_rmse(csv_path, components_to_plot=None):
    """
    Draws boxplots of RMSEs for different r_components across folds.

    Parameters:
    csv_path (str): Path to the CSV file containing 'r_component' and multiple 'rmse_foldX' columns.
    components_to_plot (list, optional): List of r_component values to display. If None, display all.
    """
    df_folds = pd.read_csv(csv_path)
    df_long = df_folds.melt(id_vars='r_component',
                            value_vars=['rmse_fold1', 'rmse_fold2', 'rmse_fold3', 'rmse_fold4', 'rmse_fold5'],
                            var_name='Fold',
                            value_name='RMSE')

    if components_to_plot is not None:
        df_long = df_long[df_long['r_component'].isin(components_to_plot)]

    #sns.set(style="whitegrid")

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_long, x='r_component', y='RMSE')

    plt.title('Boxplot of RMSE across 5 Folds vs Number of Components', fontsize=16)
    plt.xlabel('Number of Components (r_component)', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def draw_plot_mean_rmse2(csv_paths, model_labels=None):
    """
    Draws a line plot of mean RMSE vs r_component for one or multiple models.

    Parameters:
    csv_paths (str or list): Path(s) to the CSV file(s) containing 'r_component' and 'mean_rmse'.
    model_labels (list, optional): List of labels for the models. Must match the number of csv_paths.
    """
    # Ensure csv_paths is a list
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    if model_labels is not None and len(model_labels) != len(csv_paths):
        raise ValueError("Number of model labels must match the number of CSV paths provided.")

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot each model
    for idx, path in enumerate(csv_paths):
        df_mean = pd.read_csv(path)
        label = model_labels[idx] if model_labels else f'Model {idx + 1}'

        sns.lineplot(data=df_mean, x='r_component', y='mean_rmse', marker='o', linewidth=2.5, label=label)

    # Titles and labels
    plt.title('Mean RMSE vs Number of Components (Multiple Models)', fontsize=16)
    plt.xlabel('Number of Components (r_component)', fontsize=14)
    plt.ylabel('Mean RMSE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Models', fontsize=12, title_fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# zeros:
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SGD_lambda.csv", name_model='SGD', imputing=' - ')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD2_mean.csv", name_model='SVD2', imputing='fill with mean')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD2_mean_user.csv", name_model='SVD2', imputing='fill with user mean')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD1.csv", name_model='SVD2', imputing='fill with using PCA')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_NMF.csv", name_model='NMF')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD2.csv", name_model='NMF')


# means:
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_NMF_mean.csv", name_model='NMF', imputing='fill with mean')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_NMF_mean.csv", name_model='NMF', imputing='fill with mean')

# # means_user:
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD1_mean_user.csv", name_model='SVD1', imputing='fill with user mean')
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_NMF_mean_user.csv", name_model='NMF', imputing='fill with user mean')

# knn:
# draw_plot_mean_rmse("../data/grid_search_AvgRMSE_SVD1_knn.csv")

#
# draw_boxplots_rmse("../data/grid_search_FoldsRMSE_SGD_all.csv")
# draw_boxplots_rmse("../data/grid_search_FoldsRMSE_SVD1_mean.csv", components_to_plot=[4, 10, 16, 20, 26, 30, 50])
# draw_boxplots_rmse("../data/grid_search_FoldsRMSE_NMF_mean.csv", components_to_plot=[4, 10, 16, 20, 26, 30, 50])
# draw_boxplots_rmse("../data/grid_search_FoldsRMSE_SVD1_mean.csv", components_to_plot=[4, 10, 16, 20, 26, 30, 50])

df_SGD = pd.read_csv("../data/grid_search_AvgRMSE_SGD_all.csv")
df_SVD1_0 = pd.read_csv("../data/grid_search_AvgRMSE_SVD1_pca.csv")
df_NMF_0 = pd.read_csv("../data/grid_search_AvgRMSE_NMF_pca.csv")
df_SVD2_0 = pd.read_csv("../data/grid_search_AvgRMSE_SVD2_pca.csv")


def compare_methods(list_of_df):
    list_of_best_RMSE = []

    for df in list_of_df:
        list_of_best_RMSE.append(df.loc[df['mean_rmse'] == min(df['mean_rmse'])])

    df_all = pd.concat(list_of_best_RMSE, ignore_index=True)
    df_all.insert(0, 'model', ['SVD1', 'NMF', 'SDV2'])
    return df_all


df0 = pd.read_csv("../data/grid_search_AvgRMSE_SGD_lambda.csv")
df1 = pd.read_csv("../data/grid_search_AvgRMSE_SGD_all.csv")
df2 = df1[['lmb', 'r_component', 'mean_rmse']].loc[df1['r_component'].isin([1, 5, 10, 25])]

df_full = pd.concat([df2, df0])
print(df_full)

# draw_plot_mean_rmse_SGD_lambda(df_full)


print(df_full.groupby(["lmb", "r_component"]).min())
print(df_full.loc[df_full.groupby("lmb")["mean_rmse"].idxmin()])

