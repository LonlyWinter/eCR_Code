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

fontManager.addfont('/mnt/liudong/task/20230731_HT_RNA_all/result/arial.ttf')
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ['Arial']

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


def get_peak_path(dir_peak: str, sample_single: str, region_type: str, region_cutoff: str):
    sample_name = "{}.{}".format(sample_single, region_type)
    file_single = os.path.join(
        dir_peak,
        sample_name,
        "{}.{}_peaks.bed".format(
            sample_name,
            region_cutoff
        )
    )
    assert os.path.exists(file_single), "Peak File not exits: {}, {}, {}".format(
        sample_single,
        region_type,
        region_cutoff
    )
    return file_single


def extend_bed_from_center(file_bed: str, bp: int, dir_result: str = None):
    global logger
    
    assert os.path.exists(file_bed), "file bed not exists: {}".format(file_bed)
    if dir_result is None:
        dir_result = os.path.dirname(file_bed)
    os.makedirs(dir_result, exist_ok=True)
    file_bed_res = os.path.join(
        dir_result,
        os.path.basename(file_bed).replace(".bed", ".extend{}bp.bed".format(bp))
    )
    if os.path.exists(file_bed_res):
        logger.warn("file already extend {} bp: {}".format(bp, file_bed))
    else:
        exec_cmd("cat {0} | cut -f 1-3 | awk '{{a=int(($3-$2)/2)+$2+1;if(a>={1}){{print($1\"\t\"(a-{1})\"\t\"(a+{1}))}}else{{print($1\"\t0\t\"(a+{1}))}}}}' | sort -k1,1 -k2,2n | bedtools merge -i - > {2}".format(
            file_bed,
            bp,
            file_bed_res,
        ))
    return file_bed_res


def split_bed(file_bed: str, count: int, dir_result: str = None):
    assert os.path.exists(file_bed), "file bed not exists: {}".format(file_bed)
    if dir_result is None:
        dir_result = os.path.dirname(file_bed)
    os.makedirs(dir_result, exist_ok=True)
    file_bed_res = os.path.join(
        dir_result,
        os.path.basename(file_bed).replace(".bed", ".split{}.bed".format(count))
    )
    if os.path.exists(file_bed_res):
        logger.warn("file bed already split {}: {}".format(count, file_bed))
    else:
        with open(file_bed_res, "w", encoding="UTF-8") as fw:
            with open(file_bed) as fr:
                for line in fr:
                    chr_temp, start_temp, end_temp = line.split("\t", 2)
                    start_temp = int(start_temp)
                    end_temp = int(end_temp.strip().split("\t", 1)[0])
                    unit_temp = (end_temp - start_temp) // count
                    for i in range(count-1):
                        start_temp_now = i*unit_temp + start_temp
                        fw.write("{}\t{}\t{}\n".format(
                            chr_temp,
                            start_temp_now,
                            start_temp_now+unit_temp
                        ))
                    fw.write("{}\t{}\t{}\n".format(
                        chr_temp,
                        start_temp_now+unit_temp,
                        end_temp
                    ))
    return file_bed_res


def calc_bed_signal_new(bed_file: str, bw_file: str, file_res: str, tmp_dir: str = None, force: bool = False):
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


def calc_bed_signal(bed_file: str, bw_file: str, signal_dir: str, tmp_dir: str = None, force: bool = False):
    """ 根据bed计算信号值
    """

    assert os.path.exists(bed_file), "Bed file not exists: {}".format(bed_file)
    assert os.path.exists(bw_file), "Bw file not exists: {}".format(bw_file)

    if tmp_dir is None:
        tmp_dir = os.path.dirname(signal_dir)

    os.makedirs(signal_dir, exist_ok=True)
    res_signal_file = os.path.join(signal_dir, "{}.{}".format(
        os.path.basename(bed_file)[:-4],
        os.path.basename(bw_file).replace(".bw", ".txt"),
    ))

    # 已运行过就不再运行
    if os.path.exists(res_signal_file) and (not force):
        with open(res_signal_file) as f:
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
    
    exec_cmd(" ".join(["bigWigAverageOverBed", bw_file, tmp_bed_file, tmp_signal_file, "> /dev/null 2>&1"]))
    exec_cmd("""cat """+tmp_signal_file+""" | sort -k1,1n | awk -F "\t" '{print $5}' > """ + res_signal_file)
    
    with open(res_signal_file) as f:
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


def t_test(d1, d2, al):
    t = stats.ttest_rel(d1, d2, alternative=al).pvalue
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


def plot_boxplot(df: pd.DataFrame, title: str, region_samples: list):
    
    f = plt.figure(figsize=(len(df["class"].unique())*0.9, 3.5))
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
    y_lim = ax.get_ylim()
    y_unit = (y_lim[1] - y_lim[0]) * 0.02
    ax.set_ylim(y_lim[0], y_unit * 5 + y_lim[1])


    try:
        for class_single, df_temp in df.groupby("class"):
            single_temp0 = df_temp[df_temp["sample"] == region_samples[0]]["signal"].to_numpy()
            single_temp1 = df_temp[df_temp["sample"] == region_samples[1]]["signal"].to_numpy()
            s = t_test(
                single_temp0,
                single_temp1,
                "two-sided"
            )
            
            x1, x2 = xs[class_single] - 0.15, xs[class_single] + 0.15

            # pd.Series().quantile()

            q1, q3 = np.quantile(single_temp0, [0.25, 0.75])
            y3 = max(single_temp0[single_temp0 < (q3 + 1.5 * (q3 - q1))]) + y_unit

            q1, q3 = np.quantile(single_temp1, [0.25, 0.75])
            y4 = max(single_temp1[single_temp1 < (q3 + 1.5 * (q3 - q1))]) + y_unit
            
            y2 = max(y3, y4) + y_unit
            
            # print(class_single, y2, y3, y4, q3, q1)
            
            ax.text(xs[class_single], y2+y_unit, s, ha='center', va='center', color="black", fontsize="large")
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
    plt.savefig("{}.box.pdf".format(title), dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()


# %%
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230913.k27acAndk4me1"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%
dir_bed = os.path.join(dir_base, "beds")
os.makedirs(dir_bed, exist_ok=True)
# %%
region_cutoff = "e5"
dir_bw = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/bw"
dir_peak = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/peak"
# %%
region_samples = list(set(map(
    lambda d: d.rsplit(".", 2)[0],
    os.listdir(dir_bw)
)))
region_samples.sort()
# %%
region_types = ("H3K27ac", "H3K4me")
# %%
region_type = "Flag"
# %%
file_region_bw = {
    i: os.path.join(dir_bw, "{}.{}.bw".format(i, region_type)) for i in region_samples
}
# %%
file_list = list()
for s in region_samples:
    for t in region_types:
        file_list.append(get_peak_path(
            dir_peak=dir_peak,
            sample_single=s,
            region_type=t,
            region_cutoff=region_cutoff
        ))
# %%
file_ov = os.path.join(dir_base, "{}.ov.{}.bed".format(*region_types))
if not os.path.exists(file_ov):
    file_ov_temp = os.path.join(dir_bed, "{}.ov.{}.temp.bed".format(*region_types))
    exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
        " ".join(file_list),
        file_ov_temp
    ))
    cmd = ""
    for i in file_list:
        cmd = "{}| intersectBed -c -a - -b {}".format(
            cmd,
            i,
        )
    exec_cmd("cat {} {} > {}".format(
        file_ov_temp,
        cmd,
        file_ov
    ))
    exec_cmd("cat {} > {}".format(file_ov, file_ov_temp))
    exec_cmd("echo \"chr\tstart\tend\t{}\" > {}".format("\t".join(map(
        lambda d: os.path.basename(d)[:-13],
        file_list
    )), file_ov))
    exec_cmd("cat {} >> {}".format(file_ov_temp, file_ov))
    exec_cmd("rm -rf {}".format(file_ov_temp))
    
    df = pd.read_table(file_ov)
    
    cols = list(map(lambda d: "{}.{}".format(d[0], d[1]), product(region_samples, region_types)))
    
    for i in cols:
        df[i] = (df[i] > 0).astype(int)
    
    df["type"] = df.apply(lambda d: "".join(map(str, d[cols])), axis=1)
    
    type_need = {
        "1100": "WTSpec",
        "1111": "Common",
        "0011": "N70Spec"
    }
    
    df = df[df["type"].isin(type_need.keys())]
    
    for t, df_temp in df.groupby("type"):
        df_temp[["chr", "start", "end"]].to_csv(
            os.path.join(dir_bed, "{}.bed".format(type_need[t])),
            sep="\t",
            index=False,
            header=False
        )
# %%
func_list = list()
args_list = list()
for file_bed in os.listdir(dir_bed):
    for bw_name in file_region_bw:
        func_list.append(calc_bed_signal)
        args_list.append(dict(
            bed_file=os.path.join(dir_bed, file_bed),
            bw_file=file_region_bw[bw_name],
            signal_dir=os.path.join(dir_base, "Signal.{}".format(region_type)),
            tmp_dir=dir_base
        ))
# %%
multi_task(
    func_list=func_list,
    arg_list=args_list,
)
# %%
data_plot = list()
for file_bed in os.listdir(dir_bed):
    for bw_name in file_region_bw:
        signal_temp = calc_bed_signal(
            bed_file=os.path.join(dir_bed, file_bed),
            bw_file=file_region_bw[bw_name],
            signal_dir=os.path.join(dir_base, "Signal.{}".format(region_type)),
            tmp_dir=dir_base
        )
        for signal_temp_i in signal_temp:
            data_plot.append({
                "class": file_bed[:-4],
                "sample": bw_name,
                "signal": signal_temp_i
            })
# %%
df = pd.DataFrame(data_plot)
df.sort_values(["class", "sample"], inplace=True)
plot_boxplot(
    df=df,
    title="{}".format(region_type),
    region_samples=region_samples
)
# %%
# combine signal
# %%
dir_signal = os.path.join(dir_base, "Signal.{}".format(region_type))
# %%
data = list()
for file_bed in ("WTSpec", "Common", "N70Spec"):
    data_temp = dict()
    for bw_name in file_region_bw:
        file_temp = os.path.join(dir_signal, "{}.{}.{}.txt".format(
            file_bed,
            bw_name,
            region_type
        ))
        with open(file_temp) as f:
            data_temp[bw_name] = np.asarray(list(map(
                lambda dd: float(dd.strip()),
                f
            )))
        
    sample_names = tuple(data_temp.keys())
    for s1, s2 in zip(data_temp[sample_names[0]], data_temp[sample_names[1]]):
        data.append({
            sample_names[0]: s1,
            sample_names[1]: s2,
            "type": file_bed,
        })
df = pd.DataFrame(data)
df.set_index("type", inplace=True)
df.index.name = None
df.to_csv(
    os.path.join(dir_base, "{}.txt".format(region_type)),
    sep="\t"
)
# %%
def plot_heatamp_new(file_data: str):
    df = pd.read_table(file_data, index_col=0)

    fig = plt.figure(figsize=(1, 6))
    grid = plt.GridSpec(
        30,
        7,
        hspace=0.2,
        wspace=0.2,
        figure=fig,
    )
    grid_temp = grid[:-5, 1:]
    ax = fig.add_subplot(
        grid_temp,
        xticklabels=[],
        yticklabels=[],
    )
    grid_temp = grid[-1, 1:]
    ax_color = fig.add_subplot(
        grid_temp,
        xticklabels=[],
        yticklabels=[],
    )
    grid_temp = grid[:-5, 0]
    ax_label = fig.add_subplot(
        grid_temp,
        xticklabels=[],
        yticklabels=[],
        sharey=ax
    )
    y_min = df.max(axis=1).quantile(0.0)
    y_max = df.max(axis=1).quantile(0.9)
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("#030104", "#d57237", "#ecec9e", "#ffffcc"))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("#000000","#49473C","#fcf3cf","red","darkred"))
    norm = mpl.colors.Normalize(
        vmin=y_min,
        vmax=y_max
    )
    plt.colorbar(
        mpm.ScalarMappable(
            norm=norm,
            cmap=cmap
        ),
        cax=ax_color,
        orientation='horizontal',
        label="Signal",
    )
    ax_color.set_xticks(
        [y_min, y_max],
        ["{:.1f}".format(y_min), "{:.1f}".format(y_max)]
    )
    sns.heatmap(
        df,
        ax=ax,
        norm=norm,
        cmap=cmap,
        cbar=False,
        rasterized=True,
    )
    ax.axvline(
        x=1,
        color="black",
        linewidth=1,
        zorder=10,
    )
    n = 0
    df.index.name = "type"
    df.reset_index(inplace=True)
    df_grouped = df.groupby("type")
    for t in df["type"].unique():
        n += len(df_grouped.get_group(t))
        ax.axhline(
            y=n,
            color="black",
            linewidth=1,
            zorder=10,
        )
    
    ax.set_xticklabels(
        [i.get_text().replace(".Oct4", "") for i in ax.get_xticklabels()],
        rotation=90,
        # va="top",
        # ha="right"
    )
    # ax.set_yticks([])
    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(1)
    ax.set_title(os.path.basename(file_data)[:-4])


    # label
    labels = dict(enumerate(df["type"].to_list()))
    label_dict = dict()
    label_max = 0
    for lp in labels:
        lt = labels[lp]
        if lt not in label_dict:
            label_dict[lt] = list()
        label_dict[lt].append(lp)
        label_max = max(label_max, lp)
    label_dict_labels = dict(map(
        lambda d: ((max(label_dict[d]) - min(label_dict[d]))/2 + min(label_dict[d]), d),
        label_dict
    ))
    g = label_max * 0.003
    # g = min(map(lambda d: len(d), label_dict.values())) / 4
    l = g
    label_dict_labels_posi = list()
    label_dict_labels_lab = list()
    y_ax_x_posi = 0.5
    ax_label.set_xlim(-y_ax_x_posi*2, y_ax_x_posi)
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
    ax_label.yaxis.set_ticks([])
    for p, l in zip(label_dict_labels_posi, label_dict_labels_lab):
        ax_label.text(0, p, l, va="center", ha="right",)
    _ = ax_label.yaxis.set_tick_params(left=False)
    ax_label.axis("off")
    ax_label.invert_yaxis()
    
    # fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    plt.savefig(file_data.replace(".txt", ".heatmap.pdf"), dpi=5000, bbox_inches='tight')
    # plt.show()
    plt.close()

plot_heatamp_new("{}.txt".format(region_type))
# %%


















# %%
extend_bp = 5
split_count = 200
# %%
dir_bed_extend = os.path.join(dir_base, "beds_extend{}k".format(extend_bp))
dir_bed_split = os.path.join(dir_base, "beds_extend{}k_split{}".format(extend_bp, split_count))
dir_bed_signal = os.path.join(dir_base, "beds_signal_extend{}k_split{}".format(extend_bp, split_count))
os.makedirs(dir_bed_extend, exist_ok=True)
os.makedirs(dir_bed_split, exist_ok=True)
os.makedirs(dir_bed_signal, exist_ok=True)
# %%
for file_single in os.listdir(dir_bed):
    file_temp = extend_bed_from_center(
        file_bed=os.path.join(dir_bed, file_single),
        bp=extend_bp*1000,
        dir_result=dir_bed_extend
    )
    split_bed(
        file_bed=file_temp,
        count=split_count,
        dir_result=dir_bed_split
    )
# %%
# 计算信号值
# %%
file_bw_dict = {
    ".".join(i): os.path.join(dir_bw, "{1}.{0}.bw".format(*i)) for i in product(region_types, region_samples)
}
# %%
def task_single(dir_type_signal: str, file_bed_single: str, name_bw_single: str, file_bw_dict: dict, dir_bed_split: str):
    file_signal_temp = os.path.join(
        dir_type_signal,
        "{}.{}.txt".format(
            file_bed_single.split(".", 1)[0],
            name_bw_single
        )
    )
    if os.path.exists(file_signal_temp):
        return
    signal_temp = calc_bed_signal_new(
        bed_file=os.path.join(dir_bed_split, file_bed_single),
        bw_file=file_bw_dict[name_bw_single],
        file_res=file_signal_temp
    )
    signal_temp = signal_temp.reshape((
        len(signal_temp) // split_count,
        split_count
    ))
    df = pd.DataFrame(signal_temp)
    df.to_csv(
        file_signal_temp,
        sep="\t",
        index=False,
        header=False
    )

# %%
func_list = list()
arg_list = list()
for file_bed_single, name_bw_single in product(
    os.listdir(dir_bed_split),
    file_bw_dict.keys()
):
    func_list.append(task_single)
    arg_list.append(dict(
        dir_type_signal=dir_bed_signal,
        file_bed_single=file_bed_single,
        name_bw_single=name_bw_single,
        file_bw_dict=file_bw_dict,
        dir_bed_split=dir_bed_split
    ))
multi_task(func_list, arg_list)
# %%
















# %%
num_dict = dict()
for file_single in os.listdir(dir_bed):
    num_dict[file_single[:-4]] = region_num(os.path.join(dir_bed, file_single))
# %%
file_heatmap = os.path.join(dir_base, "Compare.heatmap.{}.{}k.split{}.txt".format(
    "_".join(region_types),
    extend_bp,
    split_count
))
# %%
if not os.path.exists(file_heatmap):
    data = list()
    header = list()
    header_temp = set()
    for t in region_types:
        for s in region_samples:
            data_temp = list()
            for ts in os.listdir(dir_bed):
                ts = ts[:-4]
                file_temp = os.path.join(dir_bed_signal, "{}.{}.{}.txt".format(
                    ts, t, s
                ))
                signal_temp = np.loadtxt(file_temp, dtype=np.float16, ndmin=2)
                data_temp.append(signal_temp)
                if ts in header_temp:
                    continue
                header.extend([ts for _ in range(signal_temp.shape[0])])
                header_temp.add(ts)
            data.append(np.vstack(data_temp))
    df = pd.DataFrame(np.hstack(data))
    df.index = header
    df.index.name = "type"
    df.reset_index(inplace=True)
    df.to_csv(
        file_heatmap,
        sep="\t",
        index=False
    )

df = pd.read_table(
    file_heatmap,
    dtype={"type": str}
)
# %%
df["order"] = df[list(map(str, range(split_count, split_count*2)))].mean(axis=1)
df["order2"] = df["type"].map(lambda d: ("WTSpec", "Common", "N70Spec").index(d))
df.sort_values(["order2", "order"], ascending=False, inplace=True)
df.drop(["order2", "order"], axis=1, inplace=True)
# %%
palette="#2570a1\n#de773f\n#525f42\n#2585a6\n#2f271d\n#c99979\n#5f5d46\n#1b315e\n#1d1626\n#16557A\n#C7A609\n#87C232\n#008792\n#A14C94\n#15A08C\n#8B7E75".split("\n")

linewidth = 0.7
y_min = 0
y_max = 1.5

cmaps = tuple(map(
    lambda d: mpl.colors.LinearSegmentedColormap.from_list("www", ("white", *d)),
    zip(
        ("#ebf2f7", "#fef8f5"),
        ("#96bad2", "#df8a5c"),
        palette[:len(region_types)],
    )
))
norms_y = [y_max, y_max]
norms = tuple(map(
    lambda d: mpl.colors.Normalize(vmin=y_min, vmax=d),
    norms_y
))
# %%
fig = plt.figure(figsize=(4, 7))
grid = plt.GridSpec(
    36,
    len(region_types)*4,
    hspace=0,
    wspace=0.3,
    figure=fig,
)

def plot_single(ax: plt.Axes, df_temp: pd.DataFrame, color: str, norm, cmap, plot_text: bool, num_dict: dict):
    global linewidth, split_count, extend_bp
    type_col = tuple(filter(
        lambda d: d.startswith("type"),
        df_temp.columns
    ))[0]
    df_temp.set_index(type_col, inplace=True)
    sns.heatmap(
        df_temp,
        ax=ax,
        norm=norm,
        cmap=cmap,
        cbar=False,
        rasterized=True
    )
    ax.axvline(
        x=1,
        color=color,
        linewidth=linewidth,
        zorder=10,
    )
    n = 0
    df_temp_groupby = df_temp.reset_index().groupby(type_col)
    ls_all = tuple(df_temp.index.unique())
    ls_max = len(ls_all) - 1
    ytext = len(df_temp.columns) * 1.1
    for li, ls in enumerate(ls_all):
        l = len(df_temp_groupby.get_group(ls))
        n += l
        if plot_text:
            ax.text(
                ytext,
                n - (l / 2),
                "{} (n={})".format(ls, num_dict[ls]),
                va="center",
                ha="left"

            )
        if li == ls_max:
            continue
        ax.axhline(
            y=n,
            color=color,
            # linewidth=linewidth,
            linestyle="--" if ls[:2] == ls_all[li+1][:2] else "-",
            linewidth=0.3 if ls[:2] == ls_all[li+1][:2] else linewidth,
            zorder=10,
        )
    ax.set_xticks(
        [0, split_count/2, split_count],
        [
            "   -{}".format(extend_bp),
            "",
            "{}   ".format(extend_bp),
        ],
        rotation=0,
        color=color
    )
    ax.set_yticks([])
    ax.set_title("")
    ax.set_ylabel("")
    ax.xaxis.set_tick_params(color=color)
    ax.spines[:].set_visible(True)
    ax.spines[:].set_linewidth(linewidth)
    ax.spines[:].set_color(color)

x_max = len(region_types)*2 - 1
for i in range(len(region_types)*2):
    grid_temp = grid[:-7, i*2:i*2+2]
    ax = fig.add_subplot(grid_temp)
    # type_temp = "type" if i > 1 else "type.{}".format(region_type_base)
    type_temp = "type"
    # if i % 2 == 0:
    #     title = region_types[i // 2]
    # else:
    if i == 2:
        i = 1
    elif i == 1:
        i = 2
    else:
        pass
    plot_single(
        ax=ax,
        df_temp=df[[type_temp] + list(map(str, range(split_count*i, split_count*(i+1))))],
        color="black",
        norm=norms[i//2],
        cmap=cmaps[i//2],
        plot_text=i == x_max,
        num_dict=num_dict
    )
    # ax.set_xlabel(region_samples[i%2].replace(".Oct4", ""), rotation=45)

    # 删除heatmap元素
    # if mask:
    #     ax.get_children()[0].remove()
    

for cmap, norm, start_temp, name_temp, y_max_temp in zip(
    cmaps,
    norms,
    range(0, len(region_types)*4, 4),
    region_types,
    norms_y
):
    grid_temp = grid[-1:, start_temp:start_temp+4]
    ax_color = fig.add_subplot(grid_temp)
    plt.colorbar(
        mpm.ScalarMappable(
            norm=norm,
            cmap=cmap
        ),
        cax=ax_color,
        orientation='horizontal',
        label=name_temp,
    )
    ax_color.set_xticks(
        [y_min, y_max_temp],
        ["     {:.1f}".format(y_min), "{:.1f}     ".format(y_max_temp)]
    )

plt.savefig(
    os.path.join(dir_base, "{}.{}k.split{}.heatmap.20230913.pdf".format(
        "_".join(region_types),
        extend_bp,
        split_count
    )),
    dpi=5000,
    bbox_inches="tight"
)
plt.close()

# %%


















# %%
# 对到基因上面
# %%
dir_gene = os.path.join(dir_base, "beds_gene")
os.makedirs(dir_gene, exist_ok=True)
# %%
# 对到promoter上面
file_promoter = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions.20230901/mm10.promoter.ncbiRefSeq.WithUCSC.bed"
file_tf = "/mnt/liudong/data/Genome/mm10/mm10.tf.txt"
# %%
for file_single in os.listdir(dir_bed):
    if file_single == "Common.bed":
        continue
    file_gene_promoter_temp = os.path.join(dir_gene, file_single.replace(".bed", ".promoter.txt"))
    exec_cmd("intersectBed -wa -a {} -b {} | cut -f 5 | sort | uniq > {}".format(
        file_promoter,
        os.path.join(dir_bed, file_single),
        file_gene_promoter_temp
    ))
    gene_great_temp = read_genes(file_gene_promoter_temp.replace("promoter", "great"))
    gene_promoter_temp = read_genes(file_gene_promoter_temp)
    write_gene(
        file_path=file_gene_promoter_temp.replace("promoter", "all"),
        genes=gene_great_temp.union(gene_promoter_temp)
    )
# %%
gene_tfs = read_genes(file_tf)
for file_single in os.listdir(dir_gene):
    write_gene(
        file_path=os.path.join(dir_gene, file_single.replace(".txt", ".tf.txt")),
        genes=read_genes(os.path.join(dir_gene, file_single)) & gene_tfs
    )
# %%
def enrich_single(file_gene: str):
    dir_result = os.path.dirname(file_gene)
    os.makedirs(dir_result, exist_ok=True)
    file_log = file_gene.replace(".txt", ".log")
    exec_cmd(" ".join([
        "CDesk_cli",
        "FunctionalEnrichment",
        dir_result,
        "species=mouse",
        "GeneSymbols={}".format(file_gene),
        "> {} 2>&1".format(file_log)
    ]))
# %%
gene_tfs = read_genes(file_tf)
file_gene=os.path.join(dir_gene, "N70Spec.great.20k.txt")
# write_gene(
#     file_path=file_gene.replace(".txt", ".tf.txt"),
#     genes=read_genes(file_gene) & gene_tfs
# )
# enrich_single(
#     file_gene=file_gene
# )
# %%
# ov cluster1
file_gene_c1 = "/mnt/liudong/task/20230731_HT_RNA_all/result/20230803.cluster/12.2023_08_03_14_18_48.select.copy/fpkm_genes_final/Cluster1.txt"
# %%
write_gene(
    file_path=file_gene.replace(".txt", ".ov.cluster1.txt"),
    genes=read_genes(file_gene_c1) & read_genes(file_gene)
)
# %%
func_list = list()
arg_list = list()
for file_single in os.listdir(dir_gene):
    if not file_single.endswith(".all.txt"):
        continue
    func_list.append(enrich_single)
    arg_list.append(dict(
        file_gene=os.path.join(dir_gene, file_single)
    ))
# multi_task(func_list, arg_list, 2)
# %%

































# %%
# 找Motif
# %%
dir_motif = os.path.join(dir_base, "beds_motif")
os.makedirs(dir_motif, exist_ok=True)
# %%
def motif_finding(file_bed: str, dir_result: str, species: str = "mm10"):
    result_dir_now = os.path.join(dir_result, os.path.basename(file_bed)[:-4])
    if not os.path.exists(result_dir_now):
        os.makedirs(result_dir_now)

    kr = os.path.join(result_dir_now, "knownResults.txt")
    if os.path.exists(kr):
        return
    exec_cmd("findMotifsGenome.pl {} {} {} -size given -p 64 -mask -preparsedDir ~/homer_data > {} 2>&1".format(
        file_bed,
        species,
        result_dir_now,
        os.path.join(result_dir_now, "run.log"),
    ))
# %%
# func_list = list()
# arg_list = list()
for file_single in os.listdir(dir_bed):
    if file_single == "Common.bed":
        continue
    func_list.append(motif_finding)
    arg_list.append(dict(
        file_bed=os.path.join(dir_bed, file_single),
        dir_result=dir_motif
    ))
multi_task(func_list, arg_list, 4)
# %%
