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
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.N70SensitiveSites"
dir_cooc = "/mnt/liudong/task/20230203_HT_NanogOct4_RNAseq/result/20230625.ATACTimeFail/cluster_fail"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%





















# %%
# Enhancer/Nanog Motif富集在N70-Sensitive Region
# %%
dir_motifs = os.path.join(dir_base, "motif_beds")
os.makedirs(dir_motifs, exist_ok=True)
# %%
file_homer_motif = "/mnt/liudong/task/20221103_huangtao_rowdata/result/homer.KnownMotifs.mm10.191105.bed"
file_motif = dict()
# %%
for i in ("Nanog", "Oct4", "OCT4-SOX2-TCF-NANOG", "Sox2", "Klf4", "Pitx1", "Tgif1", "Hoxa13",
    "Ptf1a", "Twist2", "Lhx1", "Zac1", "Bcl6", "Ap4", "Ronin", "Nrf2", "Six4", "Bach1", "Sp1", "Zfp809", "Srebp2",
    "Rfx2", "Zfp57", "Brn2", "Bach2", "Rfx1", "Sp5", "Zic", "Snail1", "Hoxc9", "Fra2", "Fos", "Sox17", "Tbx6", "Tcf21",
    "Tbet", "Gata4", "Tbr1", "Tcf12", "Tcf3", "Tcf7", "TCF4"):
    name = i.replace("-", "_")
    file_motif[name] = os.path.join(dir_motifs, "{}.bed".format(name))
    if not os.path.exists(file_motif[name]):
        exec_cmd("grep \"{}(\" {} | cut -f 1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
            i,
            file_homer_motif,
            file_motif[name]
        ))
# %%
palette="#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75\n#1E7CAF\n#46489A\n#0F231F\n#1187CD\n#b7704f\n#596032\n#2570a1\n#281f1d\n#de773f\n#525f42\n#2585a6\n#2f271d\n#c99979\n#5f5d46\n#1b315e\n#1d1626\n#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75".split("\n")
file_genome_len = "/mnt/zhaochengchen/Data/mm10/mm10.chrom.sizes"
genome_len = 0
with open(file_genome_len) as f:
    for line in f:
        line = line.strip()
        if line == "":
            continue
        line = line.split("\t",1)
        if len(line[0]) > 5:
            continue
        genome_len += int(line[1])
# %%
def enrich_score(file_bed_now: str, file_bed_background: str, file_bed_ov: str):
    global genome_len

    assert os.path.exists(file_bed_now), "Now bed file not exists !"
    assert os.path.exists(file_bed_background), "Background bed file not exists !"
    num_now = region_num(file_bed_now)
    len_background = region_len(file_bed_background)
    num_ov = region_num(file_bed_ov)
    return (num_ov / num_now) / (len_background / genome_len)
# %%
def enrich(file_region_change: str, type_change: str, file_bed_dict: dict, dir_enrich_now: str):
    num_bed_dict = dict()
    for bed_name in file_bed_dict:
        file_bed_temp = os.path.join(dir_enrich_now, "{}.ov{}.bed".format(bed_name, type_change))
        if not os.path.exists(file_bed_temp):
            exec_cmd("intersectBed -wa -a {} -b {} | sort -k1,1 -k2,2n | uniq > {}".format(
                file_region_change,
                file_bed_dict[bed_name],
                file_bed_temp
            ))
        num_bed_dict[bed_name] = region_num(file_bed_temp)
    
    num_bed_dict["Change_all"] = region_num(file_region_change)
    
    for bed_name in file_bed_dict:
        num_bed_dict["{}_All".format(bed_name)] = region_num(file_bed_dict[bed_name])
    
    with open(os.path.join(dir_enrich_now, "num{}.json".format(type_change)), "w", encoding="UTF-8") as f:
        json.dump(num_bed_dict, f, indent=4)
    
    score_bed_dict = dict()
    for bed_name in file_bed_dict:
        score_bed_dict[bed_name] = enrich_score(
            file_bed_now=file_region_change,
            file_bed_background=file_bed_dict[bed_name],
            file_bed_ov=file_bed_dict[bed_name].replace(".bed", ".ov{}.bed".format(type_change))
        )
    
    with open(os.path.join(dir_enrich_now, "enrich{}.json".format(type_change)), "w", encoding="UTF-8") as f:
        json.dump(score_bed_dict, f, indent=4)
# %%
def enrich_main(file_bed_dict: dict, dict_type_name: str):
    global palette, dir_base
    
    dir_enrich_now = os.path.join(dir_base, dict_type_name)
    os.makedirs(dir_enrich_now, exist_ok=True)

    for bed_name in file_bed_dict:
        file_bed_temp = os.path.join(dir_enrich_now, "{}.bed".format(bed_name))
        if not os.path.exists(file_bed_temp):
            exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
                file_bed_dict[bed_name],
                file_bed_temp
            ))
        file_bed_dict[bed_name] = file_bed_temp
    
    bed_dict = list(map(
        lambda d: (d[13:-4], os.path.join(dir_cooc, d)),
        os.listdir(dir_cooc)
    ))
    bed_dict.sort(key=lambda d: d[0], reverse=True)
    bed_dict = dict(bed_dict)
    for bed_name in bed_dict:
        enrich(
            file_region_change=bed_dict[bed_name],
            type_change=".{}".format(bed_name),
            file_bed_dict=file_bed_dict,
            dir_enrich_now=dir_enrich_now,
        )
    data_res = list()
    for file_single in os.listdir(dir_enrich_now):
        if not (file_single.startswith("enrich.") and file_single.endswith(".json")):
            continue
        type_name = file_single[7:-5]
        with open(os.path.join(dir_enrich_now, file_single)) as f:
            data = json.load(f)
            for k in data:
                data_res.append({
                    "type": type_name,
                    "region": k,
                    "EnrichScore": data[k]
                })
    df = pd.DataFrame(data_res)
    df.sort_values("type", inplace=True)
    df["log2(EnrichScore)"] = np.log2(df["EnrichScore"])
    fig = plt.figure(figsize=(len(df["type"].unique()) * len(df["region"].unique()) * 0.1, 3))
    ax = fig.add_subplot()
    sns.barplot(
        df,
        x="type",
        y="log2(EnrichScore)",
        hue="region",
        ax=ax,
        palette=palette
    )
    ax.set_title("N70-Sensitive Enrich")
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
        os.path.join(dir_base, "enrich.{}.pdf".format(dict_type_name)),
        bbox_inches="tight",
        dpi=200
    )
    # plt.show()
    plt.close()
# %%
file_bed_dict = {
    "Promoter": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions/mm10.promoter.ncbiRefSeq.WithUCSC.bed",
    "GeneBody": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions/mm10.genebody.ncbiRefSeq.WithUCSC.bed",
    "EnhancerMEF": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230724.beds/MEF.bed",
    "EnhancerESC": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230724.beds/ESC.bed",
    "Enhancer10k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer10K.bed",
    "Enhancer20k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer20K.bed",
    "Enhancer30k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer30K.bed",
    "Enhancer40k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer40K.bed",
    "Enhancer50k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer50K.bed",
}
# %%
dir_peak = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/peak"
file_bed_peaks = dict(map(
    lambda d: (d, os.path.join(dir_peak, d, "{}.e5_peaks.bed".format(d))),
    os.listdir(dir_peak)
))
# %%
enrich_main(
    file_bed_dict=file_bed_dict,
    dict_type_name="GenomeRegion"
)
# %%
enrich_main(
    file_bed_dict=file_bed_peaks,
    dict_type_name="CUTTAGPeaks"
)
# %%
enrich_main(
    file_bed_dict=file_motif,
    dict_type_name="Motif"
)
# %%
