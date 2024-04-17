# %%
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mpt
import matplotlib.patches as patches
from matplotlib.font_manager import fontManager
import pandas as pd
import os


fontManager.addfont("/mnt/runtask/CDesk/src/Tasks/arial.ttf")
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]


def read_genes(file_gene: str):
    with open(file_gene) as f:
        genes_temp = set(filter(
            lambda d: d != '',
            map(
                lambda dd: dd.strip(),
                f
            )
        ))
    return genes_temp


def write_gene(file_path: str, genes: set):
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(set(genes)))

# %%
dir_base = "/mnt/liudong/task/20240116_HT/result.sort/fig2/2b"
file_data = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230906.MS/Result-X101SC23084594_Z01_J001_B1_43/4.DiffExprAnalysis/all_diff_prot.xls"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%
df = pd.read_table(file_data)
# %%
ts = ("HT_M_1.vs.HT_M_3", "HT_M_2.vs.HT_M_4")
data_res = dict()
data_res["WT"] = set(df[((df["{}.UP.DOWN".format(ts[0])] == "Significant") & (df["{}.log2FC".format(ts[0])] < 0)) & ((df["{}.UP.DOWN".format(ts[1])] == "Significant") & (df["{}.log2FC".format(ts[1])] < 0))]["Gene"].dropna().unique())
data_res["N70"] = set(df[((df["{}.UP.DOWN".format(ts[0])] == "Significant") & (df["{}.log2FC".format(ts[0])] > 0)) & ((df["{}.UP.DOWN".format(ts[1])] == "Significant") & (df["{}.log2FC".format(ts[1])] > 0))]["Gene"].dropna().unique())
data_res["Common"] = set(df[(df["{}.UP.DOWN".format(ts[0])] == "Nonsignificant") & (df["{}.UP.DOWN".format(ts[1])] == "Nonsignificant")]["Gene"].dropna().unique())
# %%
print("\n".join(map(lambda d: "{} {}".format(d[0], len(d[1])), data_res.items())))
# %%
for k in data_res:
    write_gene(
        file_path=os.path.join(dir_base, "{}.txt".format(k)),
        genes=data_res[k]
    )
# %%
genes_baf = read_genes("/mnt/liudong/task/20230407_MSP2300090/BAF_genes.txt")
write_gene(
    file_path=os.path.join(dir_base, "baf.txt"),
    genes=data_res["N70"] & genes_baf
)
# %%

def check_int(data: int, label: str):
    assert isinstance(data, int) and (data >= 0), "{} not positive int !".format(label)

def venn2(area1: int, area2: int, area_cross: int, labels: list, num_ext: list, title: str, margin: float = 0.5):
    
    check_int(area1, "area1")
    check_int(area2, "area2")
    check_int(area_cross, "area_cross")

    w_ori = 2
    h_ori = 1.5
    colors = ["#92b2af", "#cc9b8b"]
    alpha = [0.5, 0.5]

    assert len(labels) == 2, "labels need 2 length !"
    label_colors = ["black", "black"]
    label_sizes = [11, 11]

    nums = [area1, area_cross, area2]
    num_colors = ["black", "black", "black"]
    num_sizes = [11, 11, 11]


    # 去除margin的区域画图
    w = w_ori - (margin * 2)
    h = h_ori - (margin * 2)

    area1 *= 2
    # area2 *= 0.9
    # area1 = np.log2(area1 + 1)
    # area2 = np.log2(area2 + 1)
    # area_cross = np.log2(area_cross + 1)
    area_cross *= 0.4
    # 直径
    w1 = (area1 + area_cross) / (area1 + area2 + area_cross) / 2 * w
    w2 = (area2 + area_cross) / (area1 + area2 + area_cross) / 2 * w
    wc = area_cross / (area1 + area2 + area_cross) * w

    w1, w2, wc = round(w1, 2), round(w2, 2), round(wc, 2)

    fig = plt.figure(figsize=(w_ori, h_ori))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.xaxis.set_major_formatter(mpt.NullFormatter())
    ax.yaxis.set_major_formatter(mpt.NullFormatter())
    ax.yaxis.set_tick_params(left=False)
    ax.xaxis.set_tick_params(bottom=False)
    [ax.spines[i].set_visible(False) for i in ("top", "bottom", "left", "right")]

    ax.set_xlim((0, w_ori))
    ax.set_ylim((0, h_ori))
    
    h_center = h / 2 + margin

    # 画圆
    for x, r, i in zip(
        [
            w1 + margin,
            w1 * 2 - wc + w2 + margin,
        ],
        [
            w1,
            w2,
        ],
        range(2)
    ):
        c = patches.Circle(
            (x, h_center),
            r,
            facecolor=colors[i],
            alpha=alpha[i],
        )
        ax.add_patch(c)

    # 画圆内的数字
    for x, i, e in zip(
        [
            (w1 * 2 - wc) / 2 + margin,
            w1 * 2 + margin - (wc / 2),
            (w2 * 2 - wc) / 2 + (w1 * 2) + margin
        ],
        range(3),
        num_ext,
    ):
        ax.text(
            x=x,
            y=h_center,
            s="{}{}".format(nums[i], e),
            fontdict={
                "size": num_sizes[i],
                "color": num_colors[i],
            },
            ha="center",
            va="center"
        )
    
    # 画圆外的标签
    for x, y, i in zip(
        [
            w1 / 5 + margin,
            w1 * 2 + w2 * 2 - wc - w2 / 5 + margin
        ],
        [
            w1 * 2 + margin * 0.7,
            w1 * 2 + margin * 0.7,
        ],
        range(2)
    ):
        ax.text(
            x=x,
            y=y,
            s=str(labels[i]),
            fontdict={
                "size": label_sizes[i],
                "color": label_colors[i],
            },
            ha="right" if i == 0 else "left",
            va="top"
        )
    ax.set_title(title)

    # plt.show()
    plt.savefig(
        "{}.venn.pdf".format(title.replace(" ", ".")),
        dpi=200,
        bbox_inches='tight'
    )
    plt.close()

# %%
venn2(
    area1=len(data_res["WT"]),
    area2=len(data_res["N70"]),
    area_cross=len(data_res["Common"]),
    labels=["WT", "N70"],
    num_ext=["", "", ""],
    title="MS"
)
# %%
