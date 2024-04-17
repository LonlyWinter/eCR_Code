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
import random
import logging
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


def calc_bed_signal(bed_file: str, bw_file: str, signal_dir: str, tmp_dir: str, force: bool = False):
    """ 根据bed计算信号值
    """

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


def read_signal(file_signal: str):
    with open(file_signal) as f:
        res_data = np.asarray(list(map(
            lambda dd: float(dd.strip()),
            f
        )))
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


# %%
# extend_bp = int(sys.argv[1])
extend_bp = 1000
name_extend = "base" if extend_bp == 0 else "{}bp".format(extend_bp)
dir_bw = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.pipline/bw"
dir_base = "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230810.Nanog.Binding.{}".format(name_extend)
dir_signal = os.path.join(dir_base, "Signal")
dir_classes = os.path.join(dir_base, "types")
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%
















# %%
# 找出Nanog的Motif Region
# %%
file_homer_motif = "/mnt/liudong/task/20221103_huangtao_rowdata/result/homer.KnownMotifs.mm10.191105.bed"
file_nanog_motif = os.path.join(dir_base, "Nanog.bed")
# %%
if extend_bp == 0:
    exec_cmd("grep \"Nanog(\" {} | cut -f 1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(file_homer_motif, file_nanog_motif))
else:
    exec_cmd("grep \"Nanog(\" {0} | cut -f 1-3 | awk '{{a=int(($3-$2)/2)+$2+1;if(a>={2}){{print($1\"\t\"(a-{2})\"\t\"(a+{2}))}}else{{print($1\"\t0\t\"(a+{2}))}}}}' | sort -k1,1 -k2,2n | bedtools merge -i - > {1}".format(
        file_homer_motif,
        file_nanog_motif,
        extend_bp//2
    ))
# %%














# %%
# 计算MEF & MES的开闭矩阵
file_mef_bed = "/mnt/liudong/task/20221103_huangtao_rowdata/result/0.mapping/peak/MEF.e5_peaks.bed"
file_esc_bed = "/mnt/liudong/task/20221103_huangtao_rowdata/result/0.mapping/peak/mES.e5_peaks.bed"
file_mef_esc_matrix = os.path.join(dir_base, "MEF_ESC.bed")
# %%
file_mef_esc_temp = os.path.join(dir_base, "MEF_ESC.tmp.bed")
exec_cmd("cat {} {} | cut -f 1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
    file_mef_bed,
    file_esc_bed,
    file_mef_esc_temp
))
cmd = ""
for i in (file_mef_bed, file_esc_bed):
    cmd = "{}| intersectBed -c -a - -b {}".format(
        cmd,
        i,
    )
exec_cmd("cat {} {} > {}".format(
    file_mef_esc_temp,
    cmd,
    file_mef_esc_matrix
))
exec_cmd("rm -rf {}".format(file_mef_esc_temp))
# %%













# %%
# 计算Nanog Region的类别
file_nanog_motif_type = os.path.join(dir_base, "Nanog.type.bed")
# %%
exec_cmd("intersectBed -loj -a {} -b {} | cut -f 1-3,7-8 | sort -k1,1 -k2,2n | uniq > {}".format(
    file_nanog_motif,
    file_mef_esc_matrix,
    file_nanog_motif_type
))
# %%
class_list = (
    ("BothNo", "==\".\"", "==\"-1\""),
    ("MEFSpec", ">=1", "==0"),
    ("ESCSpec", "==0", ">=1"),
    ("BothYes", ">=1", ">=1"),
)
# %%
os.makedirs(dir_classes, exist_ok=True)
file_types = dict(map(lambda d: (
    d[0],
    os.path.join(dir_base, "types", "Nanog.type.{}.bed".format(d[0]))
), class_list))
# %%
for class_i in class_list:
    cmd = "cat {} | ".format(
        file_nanog_motif_type
    ) + """awk '{if(($4""" + class_i[1] + """)&&($5""" + class_i[2] + """))print $1"\t"$2"\t"$3}'""" + " > {}".format(
        file_types[class_i[0]]
    )
    exec_cmd(cmd)
# %%













# %%
# 计算数量
# %%
data_num = dict()
for i in file_types:
    data_num[i] = region_num(file_types[i])
# %%
fig = plt.figure(figsize=(3, 3))
ax = plt.subplot()
s = sum(data_num.values())
ax.pie(
    x=list(data_num.values()),
    labels=list(data_num.keys()),
    startangle=60,
)
legend_t = ax.legend()
legend_t_handlers = legend_t.legendHandles
legend_t_texts = [d.get_text() for d in legend_t.texts]
legend_t_texts = list(map(
    lambda d: "{}, {:.1f}%, n={}".format(d, data_num[d] / s * 100, data_num[d]),
    legend_t_texts
))
ax.legend(
    legend_t_handlers,
    legend_t_texts,
    bbox_to_anchor=(0.5, 0),
    loc="upper center",
    frameon=False
)
plt.savefig(
    os.path.join(dir_base, "RegionNum.pie.pdf"),
    bbox_inches='tight',
    dpi=200
)
# plt.show()
plt.close()
# %%
data_num["All_motif"] = region_num(file_nanog_motif)
data_num["All_motif_type"] = region_num(file_nanog_motif_type)
# %%
data_num["type_sum"] = sum(data_num.values())
# %%
with open("RegionNum.json", "w", encoding="UTF-8") as f:
    json.dump(data_num, f, indent=4)
# %%














# %%
# 计算信号值
# %%
region_samples = list(set(map(
    lambda d: d.rsplit(".", 2)[0],
    os.listdir(dir_bw)
)))
region_samples.sort()
# %%
region_types = tuple(set(map(
    lambda d: d.rsplit(".", 2)[1],
    os.listdir(dir_bw)
)) - set(("H3K27ac", "H3K4me")))
region_types = tuple(filter(
    lambda d: not d.startswith("ATAC"),
    region_types
))
# %%
func_list = list()
args_list = list()
for region_type_single in region_types:
    data_plot = list()
    for region_sample_single in region_samples:
        file_bw = os.path.join(dir_bw, "{}.{}.bw".format(region_sample_single, region_type_single))
        assert os.path.exists(file_bw), "BigWig file not exists: {}".format(region_sample_single, region_type_single)
        for file_type_single in file_types:
            func_list.append(calc_bed_signal)
            args_list.append(dict(
                bed_file=file_types[file_type_single],
                bw_file=file_bw,
                signal_dir=dir_signal,
                tmp_dir=dir_base
            ))
multi_task(
    func_list=func_list,
    arg_list=args_list,
    task_num=64
)
# %%
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
# %%
for region_type_single in region_types:
    if region_type_single != "Flag":
        continue
    data_plot = list()
    for region_sample_single in region_samples:
        file_bw = os.path.join(dir_bw, "{}.{}.bw".format(region_sample_single, region_type_single))
        assert os.path.exists(file_bw), "BigWig file not exists: {}".format(region_sample_single, region_type_single)
        for file_type_single in file_types:
            if file_type_single not in ("MEFSpec", "ESCSpec"):
                continue
            for signal_temp in calc_bed_signal(
                bed_file=file_types[file_type_single],
                bw_file=file_bw,
                signal_dir=dir_signal,
                tmp_dir=dir_base
            ):
                data_plot.append({
                    "sample": region_sample_single,
                    "class": file_type_single,
                    "signal": signal_temp
                })
    
    f = plt.figure(figsize=(2, 3.5))
    f.subplots_adjust(hspace=0.5)
    ax = f.add_subplot(1,1,1)

    colors = {
        "NanogN70.Oct4": "#C7A609",
        "Nanog.Oct4": "#16557A"
    }
    df = pd.DataFrame(data_plot)
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
            y1, y2 = y_lim[1] - y_unit, y_lim[1]
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
    ax.set_title(region_type_single)
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False,
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    os.makedirs(os.path.join(dir_base, "bind"), exist_ok=True)
    plt.savefig(os.path.join(dir_base, "bind", "{}.two.pdf".format(region_type_single)), dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()
    # break
pdf_merge(
    pdf_dir=os.path.join(dir_base, "bind"),
    pdf_end=".two.pdf",
    pdf_result=os.path.join(dir_base, "bind.two.pdf")
)
# %%

















# %%
# 看差异Region上面的Brg1/SS18是否存在差异
# %%
dir_open = os.path.join(dir_base, "open")
os.makedirs(dir_open, exist_ok=True)
# %%
classes_need = tuple(map(lambda d: d[0], class_list))
file_bed_open = dict()
for region_base_i in ("Flag", "Nanog"):
    for class_i in classes_need:
        data_signal_list = list()
        for i in ("Nanog", "NanogN70"):
            file_temp = os.path.join(dir_signal, "Nanog.type.{}.{}.Oct4.{}.txt".format(class_i, i, region_base_i))
            signal_temp = read_signal(file_temp)
            data_signal_list.append(signal_temp)
        bool_temp = (data_signal_list[1] - data_signal_list[0]) > 0.05
        with open(os.path.join(dir_open, "{}.{}.bool.txt".format(class_i, region_base_i)), "w", encoding="UTF-8") as f:
            f.write("\n".join(map(
                str,
                bool_temp
            )))
        file_bed_temp = os.path.join(dir_classes, "Nanog.type.{}.bed".format(class_i))
        df = pd.read_table(file_bed_temp, header=None)
        df = df[bool_temp]
        file_bed_open[class_i] = os.path.join(dir_open, "{}.{}.open.bed".format(class_i, region_base_i))
        df.to_csv(
            file_bed_open[class_i],
            header=False,
            index=False,
            sep="\t",
        )
        for region_sample in region_samples:
            for region_type in region_types:
                file_name_temp = "Nanog.type.{}.{}.{}.txt".format(class_i, region_sample, region_type)
                signal_temp = read_signal(
                    os.path.join(dir_signal, file_name_temp)
                )[bool_temp]
                with open(os.path.join(dir_open, file_name_temp), "w", encoding="UTF-8") as f:
                    f.write("\n".join(map(
                        str,
                        signal_temp
                    )))
# %%
file_open_all = os.path.join(dir_open, "Nanog.type.All.bed")
exec_cmd(" ".join([
    "cat",
    " ".join(file_bed_open.values()),
    "| sort -k1,1 -k2,2n | bedtools merge -i - >",
    file_open_all
]))
# %%
bafs = tuple(set(region_types) - set(("Nanog", "Flag")))
# %%
for i in bafs:
    for region_sample in region_samples:
        calc_bed_signal(
            bed_file=file_open_all,
            bw_file=os.path.join(dir_bw, "{}.{}.bw".format(region_sample, i)),
            signal_dir=dir_open,
            tmp_dir=dir_base
        )
# %%
classes_now = list(classes_need)
classes_now.remove("BothYes")
classes_now.remove("MEFSpec")
for region_type in bafs:
    data_plot = list()
    for region_sample in region_samples:
        for class_i in classes_now:
            file_signal_temp = os.path.join(
                dir_open,
                "Nanog.type.{}.{}.{}.txt".format(class_i, region_sample, region_type)
            )
            for signal_temp in read_signal(file_signal_temp):
                data_plot.append({
                    "sample": region_sample,
                    "class": class_i,
                    "signal": signal_temp
                })

    f = plt.figure(figsize=(2, 3.5))
    f.subplots_adjust(hspace=0.5)
    ax = f.add_subplot(1,1,1)

    colors = {
        "NanogN70.Oct4": "#C7A609",
        "Nanog.Oct4": "#16557A"
    }
    df = pd.DataFrame(data_plot)
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
            y1, y2 = y_lim[1] - y_unit, y_lim[1]
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
    ax.set_title(region_type)
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False,
    )
    # fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
    plt.savefig(os.path.join(dir_open, "{}.pdf".format(region_type)), dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()

pdf_merge(
    pdf_dir=dir_open,
    pdf_end=".pdf",
    pdf_result=os.path.join(dir_base, "open.pdf")
)
# %%






















# %%
# ESCSpec比较符合趋势，看这些region是否会倾向于Promoter/Enhancer/GeneBody
# %%
file_bed_dict = {
    "Promoter": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions/mm10.promoter.ncbiRefSeq.WithUCSC.bed",
    "GeneBody": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230807.GeneRegions/mm10.genebody.ncbiRefSeq.WithUCSC.bed",
    "EnhancerMEF": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230724.beds/MEF.bed",
    "EnhancerESC": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230724.beds/ESC.bed"
}
# %%
dir_enrich = os.path.join(dir_base, "enrich")
os.makedirs(dir_enrich, exist_ok=True)
# %%
for bed_name in file_bed_dict:
    file_bed_temp = os.path.join(dir_enrich, "{}.bed".format(bed_name))
    exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | uniq | bedtools merge -i - > {}".format(
        file_bed_dict[bed_name],
        file_bed_temp
    ))
    file_bed_dict[bed_name] = file_bed_temp
# %%
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
def enrich(file_region_change: str, type_change: str):
    num_bed_dict = dict()
    for bed_name in file_bed_dict:
        file_bed_temp = os.path.join(dir_enrich, "{}.ov{}.bed".format(bed_name, type_change))
        exec_cmd("intersectBed -wa -a {} -b {} | sort -k1,1 -k2,2n | uniq > {}".format(
            file_region_change,
            file_bed_dict[bed_name],
            file_bed_temp
        ))
        num_bed_dict[bed_name] = region_num(file_bed_temp)
    
    num_bed_dict["Change_all"] = region_num(file_region_change)
    
    for bed_name in file_bed_dict:
        num_bed_dict["{}_All".format(bed_name)] = region_num(file_bed_dict[bed_name])
    
    with open(os.path.join(dir_enrich, "num{}.json".format(type_change)), "w", encoding="UTF-8") as f:
        json.dump(num_bed_dict, f, indent=4)
    
    score_bed_dict = dict()
    for bed_name in file_bed_dict:
        score_bed_dict[bed_name] = enrich_score(
            file_bed_now=file_region_change,
            file_bed_background=file_bed_dict[bed_name],
            file_bed_ov=file_bed_dict[bed_name].replace(".bed", ".ov{}.bed".format(type_change))
        )
    
    with open(os.path.join(dir_enrich, "enrich{}.json".format(type_change)), "w", encoding="UTF-8") as f:
        json.dump(score_bed_dict, f, indent=4)
# %%
for type_change in (".BothNo", ".BothYes", ".MEFSpec", ".ESCSpec"):
    enrich(
        file_region_change=os.path.join(dir_classes, "Nanog.type{}.bed".format(type_change)),
        type_change=type_change
    )
for i in ("Flag", "Nanog"):
    for t in classes_need:
        enrich(
            file_region_change=os.path.join(dir_open, "{}.{}.open.bed".format(t, i)),
            type_change=".{}Selected.{}".format(t, i)
        )
# %%
data_res = list()
for file_single in os.listdir(dir_enrich):
    if not (file_single.startswith("enrich.") and file_single.endswith(".json")):
        continue
    type_name = file_single[7:-5]
    with open(os.path.join(dir_enrich, file_single)) as f:
        data = json.load(f)
        for k in data:
            data_res.append({
                "type": type_name,
                "region": k,
                "EnrichScore": data[k]
            })
df = pd.DataFrame(data_res)
# %%
df["log2(EnrichScore)"] = np.log2(df["EnrichScore"])
# %%
fig = plt.figure(figsize=(len(df["type"].unique()) * 0.6, 3))
ax = fig.add_subplot()
sns.barplot(
    df,
    x="type",
    y="log2(EnrichScore)",
    hue="region",
    ax=ax
)
ax.set_title("Nanog Motif Enrich")
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
ax.set_xticklabels(
    [i.get_text() for i in ax.get_xticklabels()],
    rotation=30,
    va="top",
    ha="right"
)
plt.savefig(
    os.path.join(dir_base, "enrich.pdf"),
    bbox_inches="tight",
    dpi=200
)
# plt.show()
plt.close()
# %%

















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
    
    bed_dict = list(map(
        lambda d: (d[13:-4], os.path.join(dir_cooc, d)),
        os.listdir(dir_cooc)
    ))
    bed_dict.sort(key=lambda d: d[0], reverse=True)
    bed_dict = dict(bed_dict)
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
    multi_task(func_list, args_list, 12)
    
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
    df["zscore"] = df["zscore"].astype(float)
    df.to_csv(
        os.path.join(dir_base, "enrich.{}.txt".format(dict_type_name)),
        sep="\t",
        index=False
    )
    df["b_index"] = df["b"].map(lambda d: tuple(file_bed_dict.keys()).index(d))
    df.sort_values(["a", "b_index"], inplace=True)
    df = df[~df["a"].isin(("CO9", "OC1"))]
    df_all = df.copy()
    b_all = tuple(df_all["b"].unique())
    nnnn = 14
    for i in range(len(b_all) // nnnn + 1):
        b_temp = b_all[i*nnnn:i*nnnn+nnnn]
        df = df_all[df_all["b"].isin(set(b_temp))]
        fig = plt.figure(figsize=(len(df["a"].unique()) * len(df["b"].unique()) * 0.15, 3))
        ax = fig.add_subplot()
        sns.barplot(
            df,
            x="a",
            y="zscore",
            hue="b",
            ax=ax,
            palette=palette
        )
        # xmin, xmax = ax.get_xlim()
        # ax.axhline(
        #     -np.log10(0.01),
        #     xmin=xmin,
        #     xmax=xmax,
        #     linewidth=1,
        #     color="black",
        #     alpha=0.1
        # )
        ax.set_title("N70-Sensitive Enrich")
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
            os.path.join(dir_base, "enrich.zscore.{}.{}.new.pdf".format(dict_type_name, i)),
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
    # "Enhancer10k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer10K.bed",
    # "Enhancer20k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer20K.bed",
    # "Enhancer30k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer30K.bed",
    # "Enhancer40k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer40K.bed",
    # "Enhancer50k": "/mnt/liudong/task/20230703_HT_Oct4NanogN70_CUTTAG/result/20230818.Region/beds/Enhancer50K.bed",
}
# %%
enrich_main(
    file_bed_dict=file_bed_dict,
    dict_type_name="GenomeRegion"
)
