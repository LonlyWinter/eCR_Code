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
from matplotlib.patches import PathPatch
from matplotlib.font_manager import fontManager
from itertools import product, combinations, chain
# from scipy import stats
# from modin import pandas as mpd
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
import sys
import re
import os

fontManager.addfont('/mnt/liudong/task/20230731_HT_RNA_all/result/arial.ttf')
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ['Arial']

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')


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
    while True:
        buff = pipe.stdout.readline()
        if (buff == '') and (pipe.poll() != None):
            break
        buff = buff.strip()
        if buff == '':
            continue
        logger.info(buff)


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


def multi_task(func_list: list, arg_list: list[dict], task_num: int = None):
    task_all = list()
    task_data = tuple(zip(func_list, arg_list))
    bar = tqdm(total=len(task_data))
    if task_num is None:
        task_num = len(func_list)
    
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



def calc_bed_signal(bed_file: str, bw_file: str, file_res: str, tmp_dir: str = None, force: bool = False):
    """ 根据bed计算信号值
    """

    assert os.path.exists(bed_file), "Bed file not exists: {}".format(bed_file)
    assert os.path.exists(bw_file), "Bw file not exists: {}".format(bw_file)

    signal_dir = os.path.dirname(file_res)

    if tmp_dir is None:
        tmp_dir = os.path.dirname(signal_dir)

    os.makedirs(signal_dir, exist_ok=True)

    # 已运行过就不再运行
    if os.path.exists(file_res) and (not force):
        with open(file_res) as f:
            res_data = np.asarray(list(map(
                lambda dd: float(dd.strip()),
                f
            )))
        return res_data
    
    tmp_signal_file = os.path.join(tmp_dir, os.path.basename(bw_file).replace(".bw", ".signal.{}.txt".format(
        random.randint(100000, 999999)
    )))
    tmp_bed_file = os.path.join(tmp_dir, os.path.basename(bw_file).replace(".bw", ".calc.{}.bed".format(
        random.randint(100000, 999999)
    )))

    exec_cmd("cat -n " + bed_file + """ | awk -F "\t" '{print $2"\t"$3"\t"$4"\t"$1}' > """ + tmp_bed_file)
    
    exec_cmd(" ".join(["bigWigAverageOverBed", bw_file, tmp_bed_file, tmp_signal_file]))
    exec_cmd("""cat """+tmp_signal_file+""" | sort -k1,1n | awk -F "\t" '{print $5}' > """ + file_res)
    
    with open(file_res) as f:
        res_data = np.asarray(list(map(
            lambda dd: float(dd.strip()),
            f
        )))

    exec_cmd(" ".join(["rm", "-rf", tmp_signal_file]))
    exec_cmd(" ".join(["rm", "-rf", tmp_bed_file]))
    return res_data




# %%
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230822.Enrich"
dir_cooc = "/mnt/liudong/task/20230203_HT_NanogOct4_RNAseq/result/20230625.ATACTimeFail/cluster_fail"
dir_motif = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.N70SensitiveSites/motif_beds"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%

palette="#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75\n#1E7CAF\n#46489A\n#0F231F\n#1187CD\n#b7704f\n#596032\n#2570a1\n#281f1d\n#de773f\n#525f42\n#2585a6\n#2f271d\n#c99979\n#5f5d46\n#1b315e\n#1d1626\n#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75".split("\n")

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
    global palette, dir_base, dir_cooc
    
    # dir_enrich_now = os.path.join(dir_base, dict_type_name)
    # os.makedirs(dir_enrich_now, exist_ok=True)
    # dir_enrich_base = os.path.join(dir_base, "base")
    # os.makedirs(dir_enrich_base, exist_ok=True)

    # for bed_name in file_bed_dict:
    #     file_bed_temp = os.path.join(dir_enrich_now, "{}.bed".format(bed_name))
    #     if not os.path.exists(file_bed_temp):
    #         exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
    #             file_bed_dict[bed_name],
    #             file_bed_temp
    #         ))
    #     file_bed_dict[bed_name] = file_bed_temp
    
    # bed_dict = list(map(
    #     lambda d: (d[13:-4], os.path.join(dir_cooc, d)),
    #     os.listdir(dir_cooc)
    # ))
    # bed_dict.sort(key=lambda d: d[0], reverse=True)
    # bed_dict = dict(bed_dict)
    # for bed_name in bed_dict:
    #     file_bed_temp = os.path.join(dir_enrich_base, "{}.bed".format(bed_name))
    #     if not os.path.exists(file_bed_temp):
    #         exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
    #             bed_dict[bed_name],
    #             file_bed_temp
    #         ))
    #     bed_dict[bed_name] = file_bed_temp


    # func_list = list()
    # args_list = list()
    # file_list = list()
    # for bed_name_a in bed_dict:
    #     for bed_name_b in file_bed_dict:
    #         func_list.append(enrich)
    #         args_list.append(dict(
    #             file_bed_a=bed_dict[bed_name_a],
    #             file_bed_b=file_bed_dict[bed_name_b],
    #             dir_result=dir_enrich_now,
    #             title="{}_vs_{}".format(bed_name_a, bed_name_b),
    #             species="mm10"
    #         ))
    #         file_list.append("{}_vs_{}.txt".format(bed_name_a, bed_name_b))
    # multi_task(func_list, args_list, 12)
    
    if not os.path.exists(os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name))):
        # data_res = list()
        # for file_single in file_list:
        #     if not os.path.exists(os.path.join(dir_enrich_now, file_single)):
        #         continue
        #     type_name_a, type_name_b = file_single[:-4].split("_vs_", 1)
        #     with open(os.path.join(dir_enrich_now, file_single)) as f:
        #         data_temp = {
        #             "a": type_name_a,
        #             "b": type_name_b
        #         }
        #         for line in f:
        #             line = line.strip()
        #             if line == "":
        #                 continue
        #             line = line.split("\t", 1)
        #             data_temp[line[0]] = line[1]
        #         data_res.append(data_temp)
        # df = pd.DataFrame(data_res)
        # df["pval"] = df["pval"].astype(float)
        # df["-log10(pval)"] = -np.log10(df["pval"])
        # df["zscore"] = df["zscore"].astype(float)

        # def get_ratio(se: pd.Series):
        #     file_temp = "{}.{}.tmp.{}.bed".format(
        #         os.path.basename(se["filea"]),
        #         os.path.basename(se["fileb"]),
        #         random.randint(100000, 999999)
        #     )
        #     exec_cmd("intersectBed -wa -a {} -b {} | sort | uniq > {}".format(
        #         se["filea"],
        #         se['fileb'],
        #         file_temp
        #     ))
        #     se["ratio"] = round(region_num(file_temp) / region_num(se["filea"]) * 100, 1)
        #     exec_cmd("rm -rf {}".format(file_temp))
        #     return se

        # df = df.apply(get_ratio, axis=1)
        # df.to_csv(
        #     os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name)),
        #     sep="\t",
        #     index=False
        # )
        pass
    else:
        df = pd.read_table(os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name)))
        df["zscore"] = df["zscore"].astype(float)
    df = df[df["b"].isin(tuple(file_bed_dict.keys()))]
    df["b_index"] = df["b"].map(lambda d: tuple(file_bed_dict.keys()).index(d))
    df.sort_values(["a", "b_index"], inplace=True)
    df = df[~df["a"].isin(("CO9", "OC1"))]
    df["OCCO"] = df["a"].map(lambda d: d[:2])
    df_all = df.copy()

    def get_size(d: float):
        if d < 0:
            return 10
        elif d < 5:
            return 100
        elif d < 10:
            return 200
        elif d < 20:
            return 350
        else:
            return 500

    df_all["zscore_size"] = df_all["zscore"].map(get_size)

    for cooc_type, df in df_all.groupby("OCCO"):
        fig = plt.figure(figsize=(len(df["a"].unique()) * 0.5, len(df["b"].unique()) * 0.7))
        grid = plt.GridSpec(20, 7, hspace=0.1, wspace=0.1)
        # vv = max(df["ratio"].abs().max(), df["ratio"].abs().max()) * 0.8
        # vmin = 0
        # vmax = vv
        vmin = 0
        vmax = 80
        cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("lightblue", "red"))
        norm = mpl.colors.Normalize(
            vmin=vmin,
            vmax=vmax
        )

        ax_color = fig.add_subplot(
            grid[-3:-2, :3], 
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
            grid[-4:, 4:], 
            xticklabels=[],
            yticklabels=[]
        )
        # ax_size = plt.Axes()
        
        # si, sa = df["zscore"].min(), df["zscore"].max()
        # si = 10
        # sa = 100
        # ss = [
        #     si,
        #     (sa - si) / 4 + si,
        #     (sa - si) / 4 * 3 + si,
        #     sa,
        # ]
        # ax_size.scatter(
        #     [i/len(ss) for i in range(len(ss))],
        #     [0 for _ in range(len(ss))],
        #     s=ss,
        # )
        # for i, si in enumerate(ss):
        #     ax_size.text(
        #         x=i/len(ss),
        #         y=-1,
        #         s="{:.0f}".format(si),
        #         ha="center",
        #         va="top"
        #     )
        ss = [
            (10, "0"),
            (100, ""),
            (200, ""),
            (350, ""),
            (500, "20"),
        ]
        ax_size.scatter(
            [(i/len(ss))**1.4 for i in range(len(ss))],
            [0 for _ in range(len(ss))],
            s=list(map(lambda d: d[0], ss)),
        )
        for i, si in enumerate(ss):
            ax_size.text(
                x=(i/len(ss))**1.4,
                y=-1,
                s=si[1],
                ha="center",
                va="top"
            )
        ax_size.set_xlim(-0.1, 0.9)
        ax_size.set_ylim(-2, 1.5)
        ax_size.set_xlabel("EnrichmentZscore")
        # ax_size.set_xticklabels([
        #     "{:.1f}".format(si),
        #     "{:.1f}".format((sa - si) / 2 + si),
        #     "{:.1f}".format(sa),
        # ])
        ax_size.set_xticks([])
        ax_size.set_yticks([])
        ax_size.spines[:].set_visible(False)


        ax = fig.add_subplot(
            grid[:-6, :],
        )
        ax.scatter(
            df["a"].to_numpy(),
            df["b"].to_numpy(),
            s=df["zscore_size"].to_numpy(),
            c=df["ratio"].to_numpy(),
            cmap=cmap,
            norm=norm
        )
        ax.set_title("N70-Sensitive Enrich")
        # ax.set_ylabel("Enrichment Zscore")
        ax.set_ylabel("")
        ax.set_xlabel("")
        
        xlim = ax.get_xlim()
        ax.set_xlim((xlim[0] - 0.5, xlim[1] + 0.5))
        ylim = ax.get_ylim()
        ax.set_ylim((ylim[0] - 0.5, ylim[1] + 0.5))

        plt.savefig(
            os.path.join(dir_base, "enrich.zscore.{}.{}.pdf".format(dict_type_name, cooc_type)),
            bbox_inches="tight",
            dpi=200
        )
        # plt.show()
        plt.close()
# %%
file_bed_dict = dict(map(
    lambda d: (d[:-4], os.path.join(dir_motif, d)),
    os.listdir(dir_motif)
))
# %%
motif3s = (
    "Nrf2",
    "Bach1",
    "Bach2",
    "Fos",
    "Fra2",
    "OCT4_SOX2_TCF_NANOG",
    "Oct4",
    "Sox2",
    "Nanog",
)
# %%
enrich_main(
    file_bed_dict={
        i: file_bed_dict[i] for i in motif3s
    },
    dict_type_name="MotifAll"
)