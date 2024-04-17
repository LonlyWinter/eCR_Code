# %%
import warnings
warnings.filterwarnings('ignore')

import concurrent.futures as fu
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mpt
from matplotlib.font_manager import fontManager
import matplotlib
from PyPDF2 import PdfFileMerger
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import threading
import traceback
import logging
import re
import os

fontManager.addfont("/mnt/runtask/CDesk/src/Tasks/arial.ttf")
matplotlib.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')


def cmd_logging(pipe_now: subprocess.Popen, log_type: str):
    global logger
    while True:
        buff = pipe_now.stdout.readline()
        if (buff == '') and (pipe_now.poll() != None):
            break
        buff = buff.strip()
        if buff == '':
            continue
        getattr(logger, log_type)(buff)


def exec_cmd(cmd):
    global logger
    if isinstance(cmd, (list, tuple)):
        cmd = ' '.join(cmd)
    logger.info('exec cmd: {}'.format(cmd))
    pipe = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True
    )
    logging_info = threading.Thread(
        target=cmd_logging,
        args=(pipe, 'info')
    )
    logging_error = threading.Thread(
        target=cmd_logging,
        args=(pipe, 'warning')
    )
    logging_info.start()
    logging_error.start()
    logging_info.join()
    logging_error.join()


def pdf_merge(pdf_dir: str, pdf_end: str, pdf_result: str):
    file_res = os.path.basename(pdf_result)
    pdf_list = list(map(
        lambda d: os.path.join(pdf_dir, d),
        filter(
            lambda d: d.endswith(pdf_end) and (d != file_res),
            os.listdir(pdf_dir)
        )
    ))
    pdf_list.sort(key=lambda d: int(os.path.basename(d).split(".", 1)[0][7:]))
    if len(pdf_list) == 0:
        return
    pdf_res = PdfFileMerger()
    for i in pdf_list:
        pdf_res.append(i)
    pdf_res.write(pdf_result)


def write_gene(file_path: str, genes: set):
    with open(file_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(set(genes)))


def multi_task(func_list: tuple, arg_list: tuple[dict], task_num: int = 16):
    task_all = list()
    task_data = tuple(zip(func_list, arg_list))
    bar = tqdm(total=len(task_data))
    
    def task_done(f: fu.Future):
        global logger
        nonlocal bar

        bar.update(1)
        # 线程异常处理
        e = f.exception()
        if e:
            logger.error(traceback.format_exc())

    with fu.ThreadPoolExecutor(max_workers=task_num) as executor:
        for f, args in task_data:
            # 提交任务
            task_temp = executor.submit(f, **args)
            # 任务回调
            task_temp.add_done_callback(task_done)
            task_all.append(task_temp)
        fu.wait(task_all)
    bar.close()
# %%

dir_base = "/mnt/liudong/task/20240116_HT/result.sort/fig3/3a.3b.3c"
dir_cluster = os.path.join(dir_base, "fpkm_cluster")
os.chdir(dir_base)
cluster_change = dict(zip(
    ["2", "10", "11", "12", "3", "4", "6", "1", "5", "8", "7", "9"],
    map(str, range(1, 13)),
))
# %%
def read_single(file_name: str):
    global dir_cluster, cluster_change

    df_now = pd.read_table(os.path.join(dir_cluster, file_name), index_col=0)
    for i in df_now.columns:
        df_now[i] = np.log2(df_now[i]+1)
        
    df_now = df_now.T
    for i in df_now.columns:
        df_now[i] = df_now[i] - df_now[i].mean()
        # df_now[i] = df_now[i] / df_now[i].std()
    df_now = df_now.T
    
    df_now["cluster"] = os.path.splitext(file_name)[0]
    df_now["cluster"] = "Cluster{}".format(cluster_change[os.path.splitext(file_name)[0][7:]])
    return df_now
# %%
df = pd.concat(map(
    read_single,
    os.listdir(dir_cluster)
))
# %%
df.set_index("cluster", drop=True, inplace=True)
# %%
df['order'] = df.index.map(lambda d: int(d[7:]))
df.sort_values("order", inplace=True, ascending=False)
df.drop("order", inplace=True, axis=1)
# %%
pat_d = re.compile("\d+")
def sort_func(d: str):
    global pat_d
    t = "".join(pat_d.findall(d))
    if len(t) == 0:
        if d == "MEF":
            return -100
        elif d == "ESC":
            return 100000
        return ord(d[0])
    else:
        return int(t)
# %%
cols = list(df.columns)
cols.sort(key=sort_func)
# %%
df = df[cols]
# %%
df.to_csv("RNA.cluster.final.txt", sep="\t")
# %%


















# %%
def read_single2(file_name: str):
    global dir_cluster, cluster_change, cols

    df_now = pd.read_table(os.path.join(dir_cluster, file_name), index_col=0)
    df_now["cluster"] = os.path.splitext(file_name)[0]
    df_now["cluster"] = "Cluster{}".format(cluster_change[os.path.splitext(file_name)[0][7:]])
    return df_now
# %%
df = pd.concat(map(
    read_single2,
    os.listdir(dir_cluster)
))
cols_now = ["cluster"] + cols
df = df[cols_now]
df.to_csv(
    "fpkm.cluster.final.txt",
    sep="\t"
)
dir_cluster_final = "{}_final".format(dir_cluster)
dir_gene_final = "{}_genes_final".format(dir_cluster[:-8])
os.makedirs(dir_cluster_final, exist_ok=True)
os.makedirs(dir_gene_final, exist_ok=True)
for cluster, df_temp in df.groupby("cluster"):
    df_temp.drop("cluster", axis=1).to_csv(
        os.path.join(dir_cluster_final, "{}.txt".format(cluster)),
        sep="\t"
    )
    write_gene(
        os.path.join(dir_gene_final, "{}.txt".format(cluster)),
        df_temp.index.unique()
    )
# %%
dir_gene_enrich_final = dir_gene_final.replace("fpkm_genes", "genes_enrich")
os.makedirs(dir_gene_enrich_final, exist_ok=True)
# %%
def enrich_single(file_gene: str, dir_result: str):
    cmd = [
        "/usr/bin/Rscript-4.2",
        os.path.join(os.path.dirname(__file__), "enrich.r"),
        file_single,
        os.path.join(dir_result, os.path.basename(file_single)[:-4]),
        "SYMBOL",
        "Mm",
        "mmu",
        "{}.log".format(os.path.join(dir_result, os.path.basename(file_gene)[:-4])),
        "2>&1"
    ]
    exec_cmd(" ".join(cmd))
# %%
func_list = list()
args_list = list()
for file_single in os.listdir(dir_gene_final):
    func_list.append(enrich_single)
    args_list.append(dict(
        file_gene=os.path.join(dir_gene_final, file_single),
        dir_result=dir_gene_enrich_final
    ))
# %%
multi_task(
    func_list=func_list,
    arg_list=args_list,
    task_num=32
)


# %%














# %%
df = pd.read_table("RNA.cluster.final.txt", index_col=0)
# %%
def heatmap(df: pd.DataFrame, title_name: str, res_file: str):
    cutoff_max = 3
    cutoff_min = -3
    x_n = len(df.columns)
    fig = plt.figure(figsize=(x_n*0.4, 12)) 
    grid = plt.GridSpec(20, 24, hspace=0.2, wspace=0) 
    x_ax = fig.add_subplot(grid[-1:, 5:19], yticklabels=[], xticklabels=[])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("#16557A","white","red"))
    y_now = 0
    for cs, iis, ii in zip(
        [
            ("Cluster1", ),
            tuple(["Cluster{}".format(i) for i in range(2, 11)]),
            ("Cluster11", "Cluster12"),
        ],
        [4, 9, 5],
        range(3)
    ):
        main_ax = fig.add_subplot(grid[y_now:y_now+iis, 1:])
        y_ax = fig.add_subplot(grid[y_now:y_now+iis, 0], xticklabels=[], sharey=main_ax)
        y_now += iis
        if ii == 2:
            kws = dict(
                cbar_ax=x_ax,
                cbar_kws={
                    "orientation": "horizontal",
                }
            )
        else:
            kws = dict(
                cbar=False
            )
            title_name = ""
        df_temp = df[df.index.isin(cs)]
        sns.heatmap(
            df_temp,
            cmap=cmap,
            vmax=cutoff_max,
            vmin=cutoff_min,
            ax=main_ax,
            rasterized=True,
            **kws,
        )
        labels = dict(enumerate(df_temp.index))
        main_ax.set_title(title_name)
        x_label_all = [i.get_text() for i in main_ax.xaxis.get_ticklabels()]
        _ = main_ax.set_ylabel("")
        if ii == 2:
            main_ax.xaxis.set_ticklabels(ticklabels=[
                i.split("_",1)[1] if "_" in i else i for i in x_label_all
            ], rotation=0)
            _ = main_ax.yaxis.set_tick_params(left=False)
            x_label_head_all = list()
            for i in x_label_all:
                if "_" in i:
                    i = i.split("_",1)[0]
                else:
                    continue
                if (len(x_label_head_all) > 0) and (x_label_head_all[-1] == i):
                    continue
                x_label_head_all.append(i)
            _ = main_ax.set_xlabel("{}{}".format(
                " "*5,
                (" "*45).join(x_label_head_all)
            ))
        else:
            _ = main_ax.set_xticks([])
            _ = main_ax.xaxis.set_tick_params(bottom=False)
        y_min, y_max = main_ax.get_ylim()
        i = len(x_label_all) // 2
        main_ax.plot(
            [i, i],
            [y_min, y_max],
            color="white",
            linewidth=2,
            zorder=100,
        )
        side = 1
        main_ax.plot(
            [side, side],
            [y_min, y_max],
            color="white",
            linewidth=2,
            zorder=100,
        )
        main_ax.plot(
            [len(x_label_all) - side, len(x_label_all) - side],
            [y_min, y_max],
            color="white",
            linewidth=2,
            zorder=100,
        )
        main_ax.spines[:].set_visible(True)
        label_dict = dict()
        label_max = 0
        for lp in labels:
            lt = labels[lp]
            if lt not in label_dict:
                label_dict[lt] = list()
            label_dict[lt].append(lp)
            label_max = max(label_max, lp)

        x_min, x_max = main_ax.get_xlim()
        for i in label_dict:
            imin = min(label_dict[i])
            main_ax.plot(
                [x_min, x_max],
                [imin, imin],
                color="black",
                linestyle="--",
                linewidth=0.5,
            )
            imax = max(label_dict[i])
            main_ax.plot(
                [x_min, x_max],
                [imax, imax],
                color="black",
                linestyle="--",
                linewidth=0.5,
            )
        label_dict_labels = dict(map(
            lambda d: ((max(label_dict[d]) - min(label_dict[d]))/2 + min(label_dict[d]), d),
            label_dict
        ))
        g = min(map(lambda d: len(d), label_dict.values())) / 4
        l = g
        label_dict_labels_posi = list()
        label_dict_labels_lab = list()
        for d in label_dict_labels.keys():
            t = l+(d-l)*2
            if l > t:
                continue
            d_t = [l, t]
            # print(d_t)
            l = t + g*2
            if (d_t[1]-d_t[0]) <= g:
                continue
            label_dict_labels_posi.append(d)
            label_dict_labels_lab.append(label_dict_labels[d])
        y_ax.yaxis.set_ticks([])
        for p, l in zip(label_dict_labels_posi, label_dict_labels_lab):
            y_ax.text(0.5, p, l.replace("Cluster", "C"), ha="right", va="center")
        _ = y_ax.yaxis.set_tick_params(left=False)
        y_ax.axis("off")
        y_ax.invert_yaxis()
    x_ax.spines[:].set_visible(True)
    x_ax.set_xticks(
        [cutoff_min, 0, cutoff_max],
        ["{:.1f}".format(cutoff_min), "0", "{:.1f}".format(cutoff_max)]
    )
    # x_ax.set_xlabel("log2(fpkm + 1)")
    x_ax.set_xlabel("Centered expression in log scale")
    
    # res_file_png = res_file.replace(".pdf", ".png")
    # plt.savefig(res_file_png, dpi=200, bbox_inches='tight')
    plt.savefig(res_file, dpi=5000, bbox_inches='tight')
    plt.close()

heatmap(
    df=df,
    title_name="RNA cluster",
    res_file="RNA.cluster.final.20230907.pdf",
)
# %%











# %%
cols = list(df.columns)
dir_cluster_final = "{}_final".format(dir_cluster)
dir_gene_final = "{}_genes_final_now".format(dir_cluster[:-8])
# %%
markers = ["H", "D", "o", "s", "v", "h", "p", "^", "*", "x", "8", "d"]
# %%
# %%
pat_d = re.compile("\d+")
def sort_func(d: str):
    global pat_d
    t = "".join(pat_d.findall(d))
    if len(t) == 0:
        if d == "MEF":
            return -100
        elif d == "ESC":
            return 100000
        return ord(d[0])
    else:
        return int(t)


def line_plot2(ax, marker: str, data: list, x: str, y: str, series: str, title: str):
    """ 画线图    
    """
    data = deepcopy(data)
    for i in range(len(data)):
        data[i][y] = np.log2(np.asarray(data[i][y]) + 1)
    colors = {
        "NanogN70.Oct4": "#C7A609",
        "Nanog.Oct4": "#16557A"
    }
    colors_head = {
        "MEF": "#8B7E75",
        "ESC": "#0F231F",
    }

    y_min = 1000
    y_max = 0
    for i in range(len(data)):
        # y_min = min(data[i][y], y_min)
        # y_max = max(data[i][y], y_max)
        y_min = min(data[i][y].mean(), y_min)
        y_max = max(data[i][y].mean(), y_max)
    y_gap = (y_max - y_min) * 0.1
    y_min -= y_gap
    y_max += y_gap

    data_now = list()
    for i in range(len(data)):
        for j in data[i][y]:
            data_now.append({
                x: data[i][x],
                series: data[i][series],
                y: j,
            })
    


    df = pd.DataFrame(data_now)
    df_uniq = df[df[series] == "uniq"]
    sx = dict()
    for xt, df_temp in df_uniq.groupby(x):
        sx[xt] = (
            np.std(df_temp[y].values) / np.sqrt(len(df_temp[y].values)) * 2,
            df_temp[y].mean(),
        )
    def plot_head_tail(k: str):
        nonlocal ax, sx, colors_head, marker
        ax.scatter(
            [k],
            [sx[k][1]],
            color=colors_head[k],
            marker=marker,
            s=10,
        )
        ax.plot(
            [k, k],
            [sx[k][1] - sx[k][0], sx[k][1] + sx[k][0]],
            color=colors_head[k],
            linewidth=0.5,
        )
    
    plot_head_tail("MEF")
    sns.lineplot(
        df[df[series] != "uniq"],
        x=x,
        y=y,
        hue=series,
        palette=colors,
        linewidth=0.5,
        ax=ax
    )

    for k, df_temp in df[df[series] != "uniq"].groupby(series):
        xs = list()
        ys = list()
        for i in df_temp[x].unique():
            xs.append(i)
            ys.append(df_temp[df_temp[x] == i][y].mean())
        ax.scatter(
            xs,
            ys,
            color=colors[k],
            marker=marker,
            s=10,
        )
    plot_head_tail("ESC")
    # cn = int(title.split("(", 1)[0][1:])
    # if (cn < 10) and (cn > 1):
    #     yt = (y_max - y_min) / 2 + y_min
    # else:
    #     yt = sx["ESC"][1]
    ax.text(
        9.8,
        sx["ESC"][1],
        s=title,
        ha="left",
        va="center"
    )



fig = plt.figure(figsize=(3.5, 8)) 
grid = plt.GridSpec(20, 24, hspace=0.5, wspace=0) 
y_now = 0
for cs, iis, ii in zip(
    [
        ("Cluster1", ),
        tuple(["Cluster{}".format(i) for i in range(2, 11)]),
        ("Cluster11", "Cluster12"),
    ],
    [4, 9, 5],
    range(3)
):
    main_ax = fig.add_subplot(grid[y_now:y_now+iis, :])
    y_now += iis

    ni = 0
    for file_single in cs:
        file_single = os.path.join(os.path.join(dir_cluster_final, "{}.txt".format(file_single)))
        df = pd.read_table(file_single)
        data_line = list()
        for s in cols:
            ss = s.rsplit("_", 1)
            if len(ss) == 2:
                t, d = ss
            else:
                t = "uniq"
                d = ss[0]
            m = df[s].to_list()
            data_line.append({
                "day": d,
                "type": t,
                "value": m
            })
        # print(data_line)

        data_line.sort(key=lambda d: sort_func(d["day"]))

        line_plot2(
            main_ax,
            markers[ni],
            data_line,
            x="day",
            y="value",
            series="type",
            title="C{} (n={})".format(os.path.basename(file_single)[7:-4], len(df)),
        )
        ni += 1

    if ii == 2:
        pass
    else:
        _ = main_ax.xaxis.set_tick_params(bottom=False)
        _ = main_ax.set_xticks([])

    main_ax.spines["top"].set_visible(False)
    main_ax.spines["right"].set_visible(False)
    main_ax.yaxis.set_major_locator(mpt.MultipleLocator(2.0))
    main_ax.set_ylabel("log2(fpkm+1)")
    main_ax.set_xlabel("")
    main_ax.set_title("")
    main_ax.set_ylim(0, iis+0.5 if iis == 9 else iis)
    main_ax.legend().remove()

# legend_t = ax.legend()
# legend_t_handlers = legend_t.legendHandles[:2]
# legend_t_texts = [i.get_text() for i in legend_t.texts[:2]]
# ax.legend(
#     legend_t_handlers,
#     legend_t_texts
# )
# sns.move_legend(
#     ax, "upper center",
#     bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False,
# )
plt.savefig(
    "RNA.cluster.final.20230907.line.pdf",
    dpi=1000,
    bbox_inches='tight'
)
# plt.show()
plt.close()
# %%





# %%
def single(file_go: str, term_need: set, width: float, height: float, xlim: float, unit: float, title: str, file_res: str):

    df = pd.read_table(file_go)
    df.reset_index(inplace=True)
    df.rename({
        "level_0": "GOID",
        "level_1": "GOTerm"
    }, axis=1, inplace=True)
    df = df[df["GOTerm"].isin(term_need)]
    df["-log10(p.adjust)"] = -np.log10(df["p.adjust"])
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot()
    sns.barplot(
        df,
        y="GOTerm",
        x="-log10(p.adjust)",
        ax=ax,
        color="#d87f69"
    )
    ax.xaxis.set_major_locator(mpt.MultipleLocator(unit))
    for i, (_, se) in enumerate(df.iterrows()):
        # gene_ov_temp = list(set(se["genes"].split(",")) & read_genes(file_tf))
        # gene_ov_temp.sort()
        ax.text(
            x=0.05,
            y=i,
            # s=" ".join(gene_ov_temp),
            s=se["GOTerm"],
            va="center",
            ha="left",
            c="black"
        )
    ax.set_yticks([])
    ax.yaxis.set_tick_params(left=False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(0, xlim)
    plt.savefig(
        file_res,
        dpi=500,
        bbox_inches='tight'
    )
    # plt.show()
    plt.close()

single(
    file_go=os.path.join(dir_base, "genes_enrich_final", "Cluster1.GOTerms.txt"),
    term_need=set((
        "pattern specification process",
        "anterior/posterior pattern specification",
        "regionalization",
        "neuron projection guidance",
        "axon guidance",
        "stem cell differentiation",
        "morphogenesis of embryonic epithelium",
        "central nervous system neuron differentiation",
        "hindbrain development",
    )),
    width=5.0,
    xlim=15,
    height=3.0,
    unit=5.0,
    title="GO terms of Cluster1",
    file_res=os.path.join(dir_base, "SelectGOTerm.Cluster1.pdf")
)
single(
    file_go=os.path.join(dir_base, "genes_enrich_final", "Cluster12.GOTerms.txt"),
    term_need=set((
        "ossification",
        "ameboidal-type cell migration",
        "extracellular matrix organization",
        "extracellular structure organization",
        "external encapsulating structure organization",
        "cellular response to transforming growth factor beta stimulus",
        "response to transforming growth factor beta",
        "cell-substrate adhesion",
        "regulation of cellular response to growth factor stimulus",
    )),
    width=5.0,
    xlim=15,
    height=3.0,
    unit=5.0,
    title="GO terms of Cluster12",
    file_res=os.path.join(dir_base, "SelectGOTerm.Cluster12.pdf")
)
# %%
