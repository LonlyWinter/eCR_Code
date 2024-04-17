# %%
import warnings
warnings.filterwarnings('ignore')

# from sklearn.decomposition import PCA
# from sklearn import preprocessing
import concurrent.futures as fu
# from pyg2plot import Plot, JS
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mpt
import matplotlib.cm as mpm
from itertools import product
from matplotlib.patches import PathPatch
from matplotlib.font_manager import fontManager
# from scipy import stats
# from modin import pandas as mpd
from itertools import combinations
from PyPDF2 import PdfFileMerger
from tqdm import tqdm
from math import ceil
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import threading
import traceback
import time
import logging
import random
import json
import sys
import re
import os


fontManager.addfont("/mnt/liudong/task/20230731_HT_RNA_all/result/arial.ttf")
mpl.rc('pdf', fonttype=42)
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
    pdf_list.sort()
    if len(pdf_list) == 0:
        return
    pdf_res = PdfFileMerger()
    for i in pdf_list:
        pdf_res.append(i)
    pdf_res.write(pdf_result)


def region_num(file_path: str):
    num = 0
    with open(file_path) as f:
        for line in f:
            if line.strip() == '':
                continue
            num += 1
    return num


def region_len(file_path: str):
    num = 0
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t', 3)
            num += (int(line[2]) - int(line[1]) + 1)
    return num


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
type_beds = ("1101", "1100", "0101", "0100")
dir_beds = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230829.EachOther.heatmap2/Brg1New_Flag/bed_types_extend5k"
file_atac = "/mnt/liudong/task/20230203_HT_NanogOct4_RNAseq/result/20230406_heatmap/e5.fail.all/merge_beds/cluster.num.0bp.merge.ov.bed"
dir_cuttag_peak = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/peak"
dir_cuttag_bw = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/bw"
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230904.AndN70SensitiveWithTwoRegions"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%












# %%
dir_bed_base = os.path.join(dir_base, "base")
os.makedirs(dir_bed_base, exist_ok=True)
# %%
df = pd.read_table(file_atac)
# %%
class_beds = list(set(map(
    lambda d: d.split("_", 1)[1],
    filter(
        lambda dd: dd.startswith("Oct4"),
        df.columns
    )
)))
class_beds = list(filter(
    lambda d: (int(d[1:]) > 1) and (int(d[1:]) < 8),
    class_beds
))
# %%
cols_need = ["chr", "start", "end"]
for d in class_beds:
    cols_temp = ["{}_{}".format(i, d) for i in ("Oct4Nanog", "Oct4NanogN70")]
    for i in cols_temp:
        df[i] = (df[i] > 0).astype(int)
    for t, df_temp in df.groupby(cols_temp):
        file_res_temp = os.path.join(dir_bed_base, "{}.{}.bed".format(d, "".join(map(str, t))))
        if os.path.exists(file_res_temp):
            continue
        df_temp[cols_need].to_csv(
            file_res_temp,
            index=False,
            header=False,
            sep="\t"
        )
# %%
class_beds = list(map(
    lambda d: d[:-4],
    os.listdir(dir_bed_base)
))
class_beds.sort()
class_beds = tuple(class_beds)
# %%
file_data = os.path.join(dir_base, "data.json")
# %%
dir_ov_res = os.path.join(dir_base, "beds_ov")
os.makedirs(dir_ov_res, exist_ok=True)
# %%














# %%
# 看每个type(0101等)在每类bed(CO1等)内的比例
# %%
data = dict()
for t, c in product(type_beds, class_beds):
    file_type_bed_temp = os.path.join(dir_beds, "{}.extend5000bp.bed".format(t))
    file_class_bed_temp = os.path.join(dir_bed_base, "{}.bed".format(c))
    file_res_temp = os.path.join(dir_ov_res, "{}.{}.bed".format(c, t))
    # if not os.path.exists(file_res_temp):
    exec_cmd("intersectBed -wa -a {} -b {} | sort | uniq > {}".format(
        file_type_bed_temp,
        file_class_bed_temp,
        file_res_temp
    ))
    if t not in data:
        data[t] = dict()
    if c not in data[t]:
        data[t][c] = dict()
    data[t][c] = {
        "type": region_num(file_type_bed_temp),
        "class": region_num(file_class_bed_temp),
        "ov": region_num(file_res_temp),
    }
    data[t][c]["ov.percent"] = round(data[t][c]["ov"] / data[t][c]["type"] * 100, 1)
# %%
with open(file_data, "w", encoding="UTF-8") as f:
    json.dump(data, f, indent=4)
# %%
















# %%
# 画饼图
# %%
with open(file_data) as f:
    data = json.load(f)
# %%
dir_bar = os.path.join(dir_base, "plot_bar")
os.makedirs(dir_bar, exist_ok=True)
# %%
# colors = dict(zip(
#     ["Other"] + list(type_beds),
#     "#008792\n#A14C94\n#15A08C\n#8B7E75\n#1E7CAF".split("\n")
# ))
colors = "#15A08C\n#15B58F\n#A14C94\n#A15B9F".split("\n")
# %%
for class_i in class_beds:
    data_temp = dict()
    for k in ("0100", "0101", "1100", "1101"):
        data_temp[k] = data[k][class_i]
    data_temp = list(data_temp.items())
    # data_temp.sort(key=lambda d: d[0])
    # data_temp.sort(key=lambda d: int(d[0][3]), reverse=True)
    # data_sum = sum(map(lambda d: d[1], data_temp))
    # data_temp.append((
    #     "Other",
    #     100 - data_sum
    # ))
    fig = plt.figure(figsize=(3, 1.5))
    grid = plt.GridSpec(
        4,
        1,
        hspace=0,
        wspace=0,
        figure=fig,
    )
    ax_list = list()
    x_max = max(map(lambda d: d[1]["ov.percent"], data_temp)) * 1.3
    for index_i, index in enumerate(((0, 2), (2, 4))):
        grid_temp = grid[index[0]:index[1], :]
        ax = fig.add_subplot(grid_temp)
        sns.barplot(
            y=list(map(lambda d: d[0], data_temp[index[0]:index[1]])),
            x=list(map(lambda d: d[1]["ov.percent"], data_temp[index[0]:index[1]])),
            ax=ax,
            palette=colors[index[0]:index[1]]
        )
        x_test = max(map(lambda d: d[1]["ov.percent"], data_temp)) * 0.05
        for ii, i in enumerate(data_temp[index[0]:index[1]]):
            if i[1]["ov"] == 0:
                continue
            ax.text(
                x=x_test,
                y=ii,
                s="{} / {}".format(i[1]["ov"], i[1]["type"]),
                va="center",
                ha="left",
                color="black"
            )
        ax.plot(
            [x_max*0.8, x_max*0.8],
            [-0.2, 1.2],
            linewidth=1.0,
            color="black"
        )
        ax.text(
            x=x_max*0.82,
            y=0.5,
            s="{:.1f}x".format(
                data_temp[index[1]-1][1]["ov.percent"] / data_temp[index[0]][1]["ov.percent"]
            ),
            va="center",
            ha="left",
            color="black"
        )

        ax.set_xlim(0, x_max)
        ax_list.append(ax)
    
    ax_list[0].spines["bottom"].set_visible(False)
    # ax_list[0].spines["bottom"].set_linestyle()
    ax_list[1].spines["top"].set_visible(False)
    ax_list[0].axhline(
        y=-0.7,
        linestyle=":",
        dashes=(3, 2),
        color="black",
        linewidth=1.0
    )
    ax_list[0].set_ylim(-0.7, 1.7)
    ax_list[1].set_ylim(-0.7, 1.7)

    ax_list[0].xaxis.set_tick_params(bottom=False)
    ax_list[0].set_xticklabels([])
    ax_list[1].set_xlabel("Percentage of overlap regions(%)")
    class_i = class_i.split(".", 1)
    ax_list[0].set_title("{} {}".format(
        class_i[0],
        {"00": "NoOpen", "11": "BothOpen", "01": "N70SpecOpen", "10": "WTSpecOpen"}[class_i[1]]
    ))
    plt.savefig(
        os.path.join(dir_bar, "{}.{}.bar.pdf".format(*class_i)),
        dpi=200,
        bbox_inches="tight"
    )
    # plt.show()
    plt.close()
# %%


















# %%
# 计算EnrichmentScore
# %%

palette="#008792\n#A14C94\n#15A08C\n#87C232\n#8B7E75\n#1E7CAF\n#46489A\n#0F231F\n#1187CD\n#b7704f\n#596032\n#2570a1\n#281f1d\n#de773f\n#525f42\n#2585a6\n#2f271d\n#c99979\n#5f5d46\n#1b315e\n#1d1626\n#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75".split("\n")


def enrich(file_bed_a: str, file_bed_b: str, dir_result: str, title: str, species: str):
    if os.path.exists(os.path.join(dir_result, "{}.txt".format(title))):
        return
    r_str = """
library(regioneR)

setwd("{4}")

file_bed_a <- "{0}"
file_bed_b <- "{1}"

A <- toGRanges(
    file_bed_a,
    genome = "{2}"
)
B <- toGRanges(
    file_bed_b,
    genome = "{2}"
)
pt <- overlapPermTest(
    A = A,
    B = B,
    ntimes = 500,
    genome = "{2}",
    mc.cores = 16
)

res <- c(
    filea=file_bed_a,
    fileb=file_bed_b,
    pval=pt$numOverlaps$pval,
    zscore=pt$numOverlaps$zscore
)
write.table(
    res,
    "{3}.txt",
    col.names = FALSE,
    quote = FALSE,
    sep = "\t"
)
""".format(file_bed_a, file_bed_b, species, title, dir_result)
    file_r = os.path.join(dir_result, "{}.r".format(title))
    with open(file_r, "w") as f:
        f.write(r_str)
    exec_cmd("Rscript {}".format(file_r))


def enrich_main(file_bed_dict: dict, dict_type_name: str):
    global palette, dir_base, dir_beds, type_beds
    
    dir_enrich_now = os.path.join(dir_base, dict_type_name)
    os.makedirs(dir_enrich_now, exist_ok=True)
    dir_enrich_base = os.path.join(dir_base, "base")
    os.makedirs(dir_enrich_base, exist_ok=True)

    for bed_name in file_bed_dict:
        file_bed_temp = os.path.join(dir_enrich_now, "{}.bed".format(bed_name))
        if not os.path.exists(file_bed_temp):
            exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
                file_bed_dict[bed_name],
                file_bed_temp
            ))
        file_bed_dict[bed_name] = file_bed_temp
    
    bed_dict = {
        c: os.path.join(dir_bed_base, "{}.bed".format(c)) for c in class_beds
    }
    for bed_name in bed_dict:
        file_bed_temp = os.path.join(dir_enrich_base, "{}.bed".format(bed_name))
        if not os.path.exists(file_bed_temp):
            exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
                bed_dict[bed_name],
                file_bed_temp
            ))
        bed_dict[bed_name] = file_bed_temp


    func_list = list()
    args_list = list()
    file_list = list()
    for bed_name_a in bed_dict:
        for bed_name_b in file_bed_dict:
            func_list.append(enrich)
            args_list.append(dict(
                file_bed_a=bed_dict[bed_name_a],
                file_bed_b=file_bed_dict[bed_name_b],
                dir_result=dir_enrich_now,
                title="{}_vs_{}".format(bed_name_a, bed_name_b),
                species="mm10"
            ))
            file_list.append("{}_vs_{}.txt".format(bed_name_a, bed_name_b))
    multi_task(func_list, args_list, 16)
    
    if not os.path.exists(os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name))):
        data_res = list()
        for file_single in file_list:
            if not os.path.exists(os.path.join(dir_enrich_now, file_single)):
                continue
            type_name_a, type_name_b = file_single[:-4].split("_vs_", 1)
            with open(os.path.join(dir_enrich_now, file_single)) as f:
                data_temp = {
                    "a": type_name_a,
                    "b": type_name_b
                }
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    line = line.split("\t", 1)
                    data_temp[line[0]] = line[1]
                data_res.append(data_temp)
        df = pd.DataFrame(data_res)
        df["pval"] = df["pval"].astype(float)
        df["-log10(pval)"] = -np.log10(df["pval"])
        df["zscore"] = df["zscore"].astype(float, errors="ignore")
        df["zscore"].fillna(0.0, inplace=True)

        def get_ratio(se: pd.Series):
            file_temp = "{}.{}.tmp.{}.bed".format(
                os.path.basename(se["filea"]),
                os.path.basename(se["fileb"]),
                random.randint(100000, 999999)
            )
            exec_cmd("intersectBed -wa -a {} -b {} | sort | uniq > {}".format(
                se["filea"],
                se['fileb'],
                file_temp
            ))
            se["ratio"] = round(region_num(file_temp) / region_num(se["filea"]) * 100, 1)
            exec_cmd("rm -rf {}".format(file_temp))
            return se

        df = df.apply(get_ratio, axis=1)
        df.to_csv(
            os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name)),
            sep="\t",
            index=False
        )
    else:
        df = pd.read_table(os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name)), dtype={"a": str})
        df["zscore"] = df["zscore"].astype(float)
    df = df[df["b"].isin(tuple(file_bed_dict.keys()))]
    df["b_index"] = df["b"].map(lambda d: tuple(file_bed_dict.keys()).index(d))
    df.sort_values(["a", "b_index"], inplace=True)

    data_tuple = (10, 30, 50, 100)
    size_tuple = (10, 100, 200, 300, 400)

    def get_size(d: float):
        nonlocal data_tuple, size_tuple
        for i in range(len(data_tuple)):
            if d <= data_tuple[i]:
                return size_tuple[i]
        return size_tuple[-1]

    df["zscore_size"] = df["zscore"].map(get_size)

    fig = plt.figure(figsize=(len(df["a"].unique()) * 0.5, len(df["b"].unique()) * 0.7 + 1))
    grid = plt.GridSpec(20, 7, hspace=0.1, wspace=0.1)
    # vv = max(df["ratio"].abs().max(), df["ratio"].abs().max()) * 0.8
    # vmin = 0
    # vmax = vv
    vmin = 0
    vmax = 10
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("MistyRose", "red"))
    norm = mpl.colors.Normalize(
        vmin=vmin,
        vmax=vmax
    )

    ax_color = fig.add_subplot(
        grid[-10, 1:-1], 
        xticklabels=[],
        yticklabels=[]
    )
    plt.colorbar(
        mpm.ScalarMappable(
            norm=norm,
            cmap=cmap
        ),
        cax=ax_color,
        orientation='horizontal',
        label="% of targets",
    )
    ax_color.set_xticks(
        [vmin, vmax],
        ["{:.1f}".format(vmin), "{:.1f}".format(vmax)]
    )
    ax_color.spines[:].set_visible(True)

    ax_size = fig.add_subplot(
        grid[-4:, :], 
        xticklabels=[],
        yticklabels=[]
    )
    ss = list(zip(
        size_tuple,
        ["$\leq{}$        ".format(data_tuple[0])] + list(map(
            lambda d: "~{}".format(d),
            data_tuple[1:]
        )) + ["$\geq{}$".format(data_tuple[-1])]
    ))
    ax_size.scatter(
        [(i/len(ss))**1.4 for i in range(len(ss))],
        [0 for _ in range(len(ss))],
        s=list(map(lambda d: d[0], ss)),
        color="#ff9694"
    )
    for i, si in enumerate(ss):
        ax_size.text(
            x=(i/len(ss))**1.4,
            y=-1.5,
            s=si[1],
            ha="center",
            va="top"
        )
    ax_size.set_xlim(-0.1, 0.9)
    ax_size.set_ylim(-2, 1.5)
    ax_size.set_xlabel("EnrichmentZscore")
    ax_size.set_xticks([])
    ax_size.set_yticks([])
    ax_size.spines[:].set_visible(False)


    ax = fig.add_subplot(
        grid[:-13, :],
    )
    ax.scatter(
        df["a"].to_numpy(),
        df["b"].to_numpy(),
        s=df["zscore_size"].to_numpy(),
        c=df["ratio"].to_numpy(),
        cmap=cmap,
        norm=norm
    )
    ax.set_title("Enrich")
    # ax.set_ylabel("Enrichment Zscore")
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0] - 0.5, xlim[1] + 0.5))
    ylim = ax.get_ylim()
    ax.set_ylim((ylim[0] - 0.5, ylim[1] + 0.5))

    plt.savefig(
        os.path.join(dir_base, "enrich.{}.scatter.pdf".format(dict_type_name)),
        bbox_inches="tight",
        dpi=200
    )
    # plt.show()
    plt.close()


    
    fig = plt.figure(figsize=(len(df["a"].unique()) * len(df["b"].unique()) * 0.2, 2))
    ax = fig.add_subplot()
    sns.barplot(
        df,
        x="a",
        y="zscore",
        hue="b",
        ax=ax,
        palette=palette
    )
    ax.set_title("Enrich")
    ax.set_ylabel("Enrichment Zscore")
    ax.set_xlabel("")
    legend_t = ax.legend()
    ax.legend(
        legend_t.legendHandles,
        list(map(
            lambda d: d.get_text(),
            legend_t.texts
        )),
        bbox_to_anchor=(1.0, 0.5),
        loc="center left",
        frameon=False,
    )
    plt.savefig(
        os.path.join(dir_base, "enrich.{}.bar.pdf".format(dict_type_name)),
        bbox_inches="tight",
        dpi=200
    )
    # plt.show()
    plt.close()

# %%
file_bed_dict = {
    type_bed: os.path.join(dir_beds, "{}.bed".format(type_bed)) for type_bed in type_beds
}
# %%
enrich_main(
    file_bed_dict=file_bed_dict,
    dict_type_name="ATAC"
)
# %%
