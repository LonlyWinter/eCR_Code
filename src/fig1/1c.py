# %%
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as mpm
from matplotlib.font_manager import fontManager
import numpy as np
import pandas as pd
import seaborn as sns
import os


fontManager.addfont("/mnt/runtask/CDesk/src/Tasks/arial.ttf")
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]

# %%
dir_base = "/mnt/liudong/task/20240116_HT/result.sort/fig1/1c"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%
file_data = "/mnt/liudong/task/20240116_HT/result.sort/fig1/1c/1c.data.20240408.xlsx"
# %%
df = pd.read_excel(file_data, index_col=0)
# %%
df.fillna(0.0, inplace=True)
# %%
df.columns
# %%
cols_dict = dict(map(
    lambda d: (
        d,
        [d, "{}.1".format(d), "{}.2".format(d)]
    ),
    filter(
        lambda d: "." not in d,
        df.columns
    )
))
# %%
for k in cols_dict:
    df[k] = np.log10(df[cols_dict[k]].mean(axis=1)+1)
# %%
df = df[list(cols_dict.keys())]
# %%
vmax = np.log10(999)
vmax1 = np.log10(99)
vmax2 = np.log10(9)
vmin = 0
cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("white", "red"))
norm = mpl.colors.Normalize(
    vmin=vmin,
    vmax=vmax
)
# %%
fig = plt.figure(figsize=(3.5, 3))
grid = plt.GridSpec(
    1,
    7,
    hspace=0,
    wspace=2,
    figure=fig,
)
grid_temp = grid[:, :-1]
ax = fig.add_subplot(
    grid_temp,
    xticklabels=[],
    yticklabels=[],
)
# df_temp = df_now.pivot(index="x", columns="y", values="value")
# df_temp.fillna(0.0, inplace=True)
sns.heatmap(
    df,
    norm=norm,
    cmap=cmap,
    # legend=None,
    # s=120,
    square=True,
    cbar=False,
    ax=ax
)
for i in range(len(df.columns)):
    ax.axvline(
        x=i,
        linestyle="-",
        linewidth=0.3,
        color="#F0F0F0"
    )
for i in range(len(df.index)):
    ax.axhline(
        y=i,
        linestyle="-",
        linewidth=0.3,
        color="#F0F0F0"
    )
ax.spines[:].set_visible(True)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([
    "{}$^{{BiD}}$".format(d[:-3]) for d in df.columns
], rotation=60, va="top", ha="center")
ax.set_yticklabels([
    "{}".format(d) for d in df.index
])

grid_temp = grid[:, -1]
ax_color = fig.add_subplot(
    grid_temp,
    xticklabels=[],
    yticklabels=[],
)
plt.colorbar(
    mpm.ScalarMappable(
        norm=norm,
        cmap=cmap
    ),
    cax=ax_color,
)
ax_color.set_yticks(
    [vmin, vmax2, vmax1, vmax],
    ["0", "10", "100", "1000"],
)
plt.savefig(
    os.path.join(dir_base, "1c.20240408.pdf"),
    bbox_inches="tight",
    dpi=200
)
plt.show()
plt.close()
# %%
