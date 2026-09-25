# %%
import pandas as pd
import pickle as pkl
import umap
import numpy as np
from rdkit import Chem

with open("dfs_morgan_fps.pkl", "rb") as f:
    dfs_fps = pkl.load(f)

df_plot = pd.DataFrame()

for name, df in dfs_fps.items():
    df_best_per_step = df.loc[df.groupby("Step")["total_score"].idxmax()].reset_index(drop=True)
    df_best_per_step["run"] = name
    df_plot = pd.concat([df_plot, df_best_per_step], axis=0)
df_plot = df_plot.reset_index(drop=True)

def get_umap(df):
    
    umap_model = umap.UMAP(metric = "jaccard",
                        n_neighbors = 25,
                        n_components = 2,
                        low_memory = False,
                        min_dist = 0.001)
    all_fgrps = np.vstack(df['morgan_fps'])
    X_embedded = umap_model.fit_transform(np.asarray(all_fgrps))

    df['umap_x'] = X_embedded[:, 0]
    df['umap_y'] = X_embedded[:, 1]
    return df

df_plot = get_umap(df_plot)


with open("dfs_morgan_fps_umap.pkl", "wb+") as f:
    pkl.dump(df_plot, f)