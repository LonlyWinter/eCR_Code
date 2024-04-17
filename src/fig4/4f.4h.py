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
from scipy import stats
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


def read_signal(file_signal: str):
    with open(file_signal) as f:
        res_data = np.asarray(list(map(
            lambda dd: float(dd.strip()),
            f
        )))
    return res_data


def get_peak_and_bw_path(dir_peak: str, dir_bw: str, sample_single: str, region_type: str, region_cutoff: str):
    sample_name = "{}.{}".format(sample_single, region_type)
    file_single_peak = os.path.join(
        dir_peak,
        sample_name,
        "{}.{}_peaks.bed".format(
            sample_name,
            region_cutoff
        )
    )
    file_single_bw = os.path.join(dir_bw, "{}.bw".format(sample_name))
    assert os.path.exists(file_single_peak), "File not exits: {}, {}, {}".format(
        sample_single,
        region_type,
        region_cutoff
    )
    return sample_name, file_single_peak, file_single_bw


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
        random.randint(10000, 99999)
    )))
    tmp_bed_file = os.path.join(tmp_dir, os.path.basename(bw_file).replace(".bw", ".calc.{}.bed".format(
        random.randint(10000, 99999)
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


def adjust_box_widths(ax, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])


def t_test(d1, d2):
    t = stats.ttest_rel(d1, d2, alternative="less").pvalue
    # if t == 0:
    #     logger.warning("T-test value is 0: {}".format(np.all(d1 == d2)))
    #     return ""
    if t < 0.001:
        return "***"
    if t < 0.01:
        return "**"
    if t < 0.05:
        return "*"
    return "-"


def plot_boxplot(df: pd.DataFrame, title: str, ext: str):

    f = plt.figure(figsize=(len(df["class"].unique())*0.7 + 0.5, 3.5))
    f.subplots_adjust(hspace=0.5)
    ax = f.add_subplot(1,1,1)

    colors = {
        "NanogN70.Oct4": "#C7A609",
        "Nanog.Oct4": "#16557A"
    }
    sns.boxplot(
        df,
        x="class",
        y="signal",
        hue="sample",
        palette=colors,
        showfliers=False,
        zorder=100,
        width=0.6,
        ax=ax
    )
    adjust_box_widths(ax, 0.7)

    # 更改颜色
    colors_now = list()
    for c in ax.get_children():
        if not isinstance(c, PathPatch):
            continue
        col = list(c.get_facecolor())
        colors_now.append(tuple(col))
        c.set_edgecolor(col)
        col[3] = 0.5
        c.set_facecolor(col)
    
    for i, li in enumerate(ax.get_lines()):
        # print(li.get_color())

        if i % 10 >= 5:
            k = 1
        else:
            k = 0
        li.set_color(colors_now[k])
        

    xs = dict(zip(
        df["class"].unique(),
        range(len(df["class"].unique()))
    ))
    # ax.set_ylim(-0.05, 1.25)
    y_lim = ax.get_ylim()
    y_unit = (y_lim[1] - y_lim[0]) * 0.02
    ax.set_ylim(y_lim[0], y_unit * 5 + y_lim[1])


    try:
        ts = dict()
        for class_single, df_temp in df.groupby("class"):
            s = t_test(
                df_temp[df_temp["sample"] == region_samples[0]]["signal"].to_numpy(),
                df_temp[df_temp["sample"] == region_samples[1]]["signal"].to_numpy(),
            )
            ts[class_single] = s
        for class_single, df_temp in df.groupby("class"):
            if ts[class_single] == "":
                continue
            x1, x2 = xs[class_single] - 0.15, xs[class_single] + 0.15
            
            q3 = df_temp[df_temp["sample"] == region_samples[0]]["signal"].quantile(0.75)
            q1 = df_temp[df_temp["sample"] == region_samples[0]]["signal"].quantile(0.25)
            y3 = q3 + 1.5 * (q3 - q1) + y_unit
            q3 = df_temp[df_temp["sample"] == region_samples[1]]["signal"].quantile(0.75)
            q1 = df_temp[df_temp["sample"] == region_samples[1]]["signal"].quantile(0.25)
            y4 = q3 + 1.5 * (q3 - q1) + y_unit
            y2 = max(y3, y4) + y_unit
            # print(class_single, y3, y4, q3, q1)
            ax.text(xs[class_single], y2+y_unit, ts[class_single], ha='center', va='center', color="black", fontsize="large")
            ax.plot(
                [x1, x1, x2, x2],
                [y3, y2, y2, y4],
                lw=0.5,
                color="black"
            )
    except:
        print(traceback.format_exc())


    ax.set_ylabel("Signal")
    ax.set_xlabel("")
    ax.set_title(title)
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False,
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig("{}.{}.pdf".format(title, ext), dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()

# %%
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230830.EachOtherBoxplot"
dir_signal = os.path.join(dir_base, "signal")
os.makedirs(dir_signal, exist_ok=True)
os.chdir(dir_base)
# %%
file_cdesk = "/mnt/liudong/CDesk/CDesk_cli.py"
dir_bw = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/bw"
dir_peak = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/peak"
dir_bed = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230829.EachOther.heatmap2/Brg1New_Flag/bed_types_extend5k"
type_beds = ("1101", "0101", "1100", "0100")
# %%
region_samples = list(set(map(
    lambda d: d.rsplit(".", 2)[0],
    os.listdir(dir_bw)
)))
region_samples.sort()
# %%
region_types = list(set(map(
    lambda d: d.rsplit(".", 1)[1],
    os.listdir(dir_peak)
)) - set(("Flag", "Brg1New", "Nanog", "Brg1")))
region_types.sort(key=lambda d: int(d[5:]) if d.startswith("ATAC") else ord(d[3]))
# %%














# %%
func_list = list()
args_list = list()
for region_type, type_bed in product(region_types, type_beds):
    data_plot = list()
    for region_sample in region_samples:
        for class_i in ("Flag", "Brg1New", region_type):
            sample_name_temp, file_peak_temp, file_bw_temp = get_peak_and_bw_path(
                dir_peak=dir_peak,
                dir_bw=dir_bw,
                sample_single=region_sample,
                region_type=class_i,
                region_cutoff="e5"
            )
            func_list.append(calc_bed_signal)
            args_list.append(dict(
                bed_file=os.path.join(dir_bed, "{}.extend5000bp.bed".format(type_bed)),
                bw_file=file_bw_temp,
                file_res=os.path.join(dir_signal, "{}.{}.{}.txt".format(
                    type_bed,
                    class_i,
                    region_sample,
                )),
                tmp_dir=dir_base
            ))
multi_task(
    func_list=func_list,
    arg_list=args_list,
    task_num=64
)
# %%
for type_bed in type_beds:
    for region_type in region_types:
        if region_type.startswith("ATAC"):
            continue
        data_plot = list()
        for region_sample in region_samples:
            for class_i in ("Flag", "Brg1New", region_type):
                file_signal_temp = os.path.join(dir_signal, "{}.{}.{}.txt".format(
                    type_bed,
                    class_i,
                    region_sample,
                ))
                for signal_temp in read_signal(file_signal_temp):
                    data_plot.append({
                        "sample": region_sample,
                        "class": class_i,
                        "signal": signal_temp
                    })
        df = pd.DataFrame(data_plot)
        plot_boxplot(df, title="{}.{}".format(type_bed, region_type), ext="addFlagAndBrg1")
# %%
for type_bed in type_beds:
    for region_type in region_types:
        if region_type.startswith("ATAC"):
            continue
        data_plot = list()
        for region_sample in region_samples:
            for class_i in (region_type, ):
                file_signal_temp = os.path.join(dir_signal, "{}.{}.{}.txt".format(
                    type_bed,
                    class_i,
                    region_sample,
                ))
                for signal_temp in read_signal(file_signal_temp):
                    data_plot.append({
                        "sample": region_sample,
                        "class": class_i,
                        "signal": signal_temp
                    })
        df = pd.DataFrame(data_plot)
        plot_boxplot(df, title="{}.{}".format(type_bed, region_type), ext="one")
# %%
for type_bed in type_beds:
    data_plot = list()
    for region_type in region_types:
        if not region_type.startswith("ATAC"):
            continue
        for region_sample in region_samples:
            for class_i in (region_type, ):
                file_signal_temp = os.path.join(dir_signal, "{}.{}.{}.txt".format(
                    type_bed,
                    class_i,
                    region_sample,
                ))
                for signal_temp in read_signal(file_signal_temp):
                    data_plot.append({
                        "sample": region_sample,
                        "class": class_i,
                        "signal": signal_temp
                    })
    df = pd.DataFrame(data_plot)
    plot_boxplot(df, title="{}".format(type_bed), ext="ATAC")
# %%
for type_bed in type_beds:
    data_plot = list()
    for region_type in region_types:
        if not region_type.startswith(("H3K27ac", "H3K4me")):
            continue
        for region_sample in region_samples:
            for class_i in (region_type, ):
                file_signal_temp = os.path.join(dir_signal, "{}.{}.{}.txt".format(
                    type_bed,
                    class_i,
                    region_sample,
                ))
                for signal_temp in read_signal(file_signal_temp):
                    data_plot.append({
                        "sample": region_sample,
                        "class": class_i,
                        "signal": signal_temp
                    })
    df = pd.DataFrame(data_plot)
    plot_boxplot(df, title="{}".format(type_bed), ext="k27acAndk4me")

# %%
