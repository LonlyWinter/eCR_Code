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
    file_go="/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230904.GeneRegions/genes_ov.tf/Cluster1.ov.COAll.genes.GOTerms.txt",
    term_need=set((
        "blastocyst formation",
        "stem cell population maintenance",
        "embryonic placenta development",
        "blastocyst development"
    )),
    width=7.0,
    xlim=7.9,
    height=1.2,
    unit=2.0,
    title="GO terms of Cluster1 ov CO",
    file_res=os.path.join(dir_base, "3m.pdf")
)
# %%
# 20230904.GeneRegion.py
gene_rna_cluster1 = read_genes(file_rna_cluster_gene["Cluster1"])
gene_co = read_genes(file_gene_dict["CO"].replace(".tfs", ""))
gene_common = gene_rna_cluster1 & gene_co
gene_common_tfs = gene_common & gene_tfs
write_gene(os.path.join(dir_result, "Cluster1.ov.COAll.genes.txt"), gene_common)
venn2(
    area1=len(gene_rna_cluster1 - gene_common),
    area2=len(gene_co - gene_common),
    area_cross=len(gene_common),
    num_ext=["", "\n({})".format(len(gene_common_tfs)), ""],
    labels=["Cluster1", "CO.All"],
    title="Cluster1 ov CO.All"
)