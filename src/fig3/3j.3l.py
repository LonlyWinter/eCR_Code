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
from itertools import product
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
# %%
dir_result = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230904.GeneRegions"
os.chdir(dir_result)
# %%



















# %%

def plot_enrich(enrich_file_all: list, enrich_type: str, result_file: str, title: str, term_need: tuple, padj: float = 0.05, gene_min: int = 1):
    
    if len(enrich_file_all) == 0:
        print(os.path.basename(result_file), enrich_type, len(enrich_file_all), "no data: no file")
        return

    data = list()
    sample_no = list()
    for file_name in enrich_file_all:
        i = os.path.basename(file_name).rsplit(".", 3)[0].split("_", 1)[1]
        if not os.path.exists(file_name):
            sample_no.append(i)
            continue
        df_temp = pd.read_table(file_name)
        df_temp['p.adjust'].fillna(1.0, inplace=True)
        # df_temp = df_temp[df_temp['p.adjust'] < padj]
        df_temp = df_temp[df_temp.index.map(lambda d: d[1] in term_need)]
        for term_temp, se in df_temp.iterrows():
            if len(se["genes"].split(",")) < gene_min:
                continue
            y_t = term_temp[1]
            data.append({
                "sample": i,
                "term": y_t,
                "p.adjust": se["p.adjust"],
                "GeneRatio": eval(se["GeneRatio"])
            })
        if len(df_temp) == 0:
            sample_no.append(i)
        del df_temp
    
    df_data = pd.DataFrame(data)

    df_data.sort_values("sample", inplace=True)
    
    term_order = list(term_need)

    fig = plt.figure(figsize=(len(df_data['sample'].unique())*0.2+3, len(df_data['term'].unique())*0.5))
    grid = plt.GridSpec(20, 7, hspace=0, wspace=0)
    ax = fig.add_subplot(
        grid[:-5, :], 
        xticklabels=[],
        yticklabels=[]
    )
    ax_color1 = fig.add_subplot(
        grid[-1:, 2:3], 
        xticklabels=[],
        yticklabels=[]
    )
    ax_color2 = fig.add_subplot(
        grid[-1:, 3:-2], 
        xticklabels=[],
        yticklabels=[]
    )

    pval_lim = -np.log10(0.05)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("MistyRose", "red", "darkred"))

    df_data = df_data.pivot(index="term", columns="sample", values="p.adjust")
    df_data.fillna(1.0, inplace=True)
    for i in sample_no:
        df_data[i] = 1.0
    cols = list(df_data.columns)
    cols.sort()
    df_data = df_data.loc[term_order, cols]

    df_data = -np.log10(df_data)
    
    vmin=pval_lim
    vmax=3
    
    norm1 = mpl.colors.Normalize(
        vmin=0,
        vmax=pval_lim
    )
    norm2 = mpl.colors.Normalize(
        vmin=pval_lim,
        vmax=3
    )
    
    sns.heatmap(
        df_data,
        mask=df_data <= pval_lim,
        cmap=cmap,
        square=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=False
        # cbar_ax=ax_color,
        # cbar_kws={
        #     "orientation": "horizontal",
        # }
    )
    # ax = plt.Axes()
    ax.spines[:].set_visible(True)
    _ = ax.xaxis.set_ticklabels(
        [i.get_text() for i in ax.xaxis.get_ticklabels()],
        rotation=90
    )
    _ = ax.set_xlabel("")
    _ = ax.set_ylabel("")
    _ = ax.set_title(title)

    plt.colorbar(
        mpm.ScalarMappable(
            norm=norm1,
            cmap=mpl.colors.LinearSegmentedColormap.from_list("www", ("white", "white"))
        ),
        cax=ax_color1,
        orientation='horizontal',
    )
    _ = ax_color1.set_xlim(0, pval_lim)
    _ = ax_color1.spines[:].set_visible(True)
    _ = ax_color1.set_xticks(
        [0, pval_lim],
        ["$1$", ""]
    )
    plt.colorbar(
        mpm.ScalarMappable(
            norm=norm2,
            cmap=cmap
        ),
        cax=ax_color2,
        orientation='horizontal',
    )
    _ = ax_color2.spines[:].set_visible(True)
    _ = ax_color2.set_xlim(pval_lim, 3)
    _ = ax_color2.set_xticks(
        [pval_lim, 3],
        ["$0.05$", "$<10^{-3}$"]
    )
    _ = ax_color2.set_xlabel("adjust p-value")

    plt.savefig(
        result_file,
        dpi=500,
        bbox_inches='tight'
    )
    plt.close()


# %%
dir_enrich = os.path.join(dir_result, "fail_gene_all")
# %%
enrich_file_all = list(filter(
    lambda d: d.endswith("GOTerms.txt") and (d[13:15] == "CO") and (int(d[15]) < 6),
    os.listdir(dir_enrich)
))
enrich_file_all.sort()
enrich_file_all = list(map(
    lambda d: os.path.join(dir_enrich, d),
    enrich_file_all
))
# %%
# %%
# file_terms = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions.20230901/fail_gene_Promoter.GOTerms.txt"
# term_need = set(pd.read_table(file_terms)["GOTerm"].unique())
term_need = (
    "negative regulation of apoptotic signaling pathway",
    "response to leukemia inhibitory factor",
    "cell cycle G1/S phase transition",
    "mesenchymal cell differentiation",
    "stem cell differentiation",
    "developmental maturation",
    "somatic stem cell population maintenance",
)
plot_enrich(
    enrich_file_all=enrich_file_all,
    enrich_type="GO",
    title="N70-Sensitive GO Enrich",
    result_file=os.path.join(dir_result, "fail_gene_CO.pdf"),
    term_need=term_need
)
# %%



















# %%
file_term_now = os.path.join(dir_result, "genes_ov.tf", "Cluster1.ov.COAll.genes.GOTerms.txt")
# %%
df = pd.read_table(file_term_now)
# %%
df.reset_index(inplace=True)
# %%
df.rename({
    "level_0": "GOID",
    "level_1": "GOTerm"
}, axis=1, inplace=True)
# %%
file_tf = "/mnt/liudong/data/Genome/mm10/mm10.tf.txt"
term_need = (
    "blastocyst formation",
    "stem cell population maintenance",
    "embryonic placenta development",
    "blastocyst development"
)
# %%
df = df[df["GOTerm"].isin(term_need)]
# %%
df["-log10(p.adjust)"] = -np.log10(df["p.adjust"])
# %%
fig = plt.figure(figsize=(6.5, 1.5))
ax = fig.add_subplot()
sns.barplot(
    df,
    y="GOTerm",
    x="-log10(p.adjust)",
    ax=ax,
    color="#d87f69"
)
ax.xaxis.set_major_locator(mpt.MultipleLocator(2))
for i, (_, se) in enumerate(df.iterrows()):
    print(se["GOTerm"], se["genes"])
    gene_ov_temp = list(set(se["genes"].split(",")) & read_genes(file_tf))
    gene_ov_temp.sort()
    ax.text(
        x=0.05,
        y=i,
        s=" ".join(gene_ov_temp),
        va="center",
        ha="left",
        c="white"
    )
ax.set_ylabel("")
plt.savefig(
    os.path.join(dir_result, "Cluster1.ov.CO.SelectGOTerm.pdf"),
    dpi=500,
    bbox_inches='tight'
)
plt.show()
plt.close()
# %%
