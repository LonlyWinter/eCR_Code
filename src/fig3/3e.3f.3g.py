#!/usr/bin/env python
# cython: language_level=3
# -*- coding: UTF-8 -*-
# %%
import warnings
warnings.filterwarnings("ignore")
# %%
from sklearn import preprocessing
import concurrent.futures as fu
from pyg2plot import Plot, JS
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mpt
import matplotlib.cm as mpm
from itertools import product
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import traceback
import logging
import sys
import os

logging.basicConfig(level='INFO', format='%(asctime)s - %(levelname)s - %(message)s')

mpl.use("Agg")
fontManager.addfont("/mnt/runtask/CDesk/src/Tasks/arial.ttf")
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]

# %%
def get_samples(file_meta: str, important_index: int):
    """ 获取样本的信息
    """
    head_tail_names = set(("head", "tail"))

    assert os.path.exists(file_meta), "meta not exists: {}".format(file_meta)
    df = pd.read_excel(file_meta, sheet_name="meta")
    df.fillna("", inplace=True)
    assert tuple(df.columns) == ("sample", "tag_time", "tag_group"), "meta file columns error!"
    assert len(df) == len(df["sample"].unique()), "sample dumplates !"
    assert len(set(df["tag_group"].unique()) - head_tail_names) == 2, "tag group need 2 !"

    # tag_time, tag_group
    samples_group = dict()
    samples_head_tail = dict()
    samples_time = list(filter(
        lambda d: d.strip() != "",
        df[~df["tag_group"].isin(head_tail_names)]["tag_time"].unique()
    ))
    for group_name, df_temp in df.groupby("tag_group"):
        if group_name not in head_tail_names:
            assert tuple(df_temp["tag_time"]) == tuple(samples_time), "tag time order not same !"
            samples_group[group_name] = df_temp["sample"].to_list()
            continue
        samples_head_tail[group_name] = df_temp["sample"].to_list()
    
    def padding_false(add_head: bool, len_samples: int):
        list_now = [[ 1 for _ in range(i)] for i in range(1, len_samples+1)]
        len_final = max(map(
            lambda d: len(d),
            list_now
        ))
        for index in range(len(list_now)):
            gap = [0 for _ in range(len_final - len(list_now[index]))]
            if add_head:
                list_now[index] = tuple(gap + list_now[index])
            else:
                list_now[index] = tuple(list_now[index] + gap)
            list_now[index] = "".join(map(str, list_now[index]))
        list_now = list_now[:-1]
        if add_head:
            list_now.reverse()
        return list_now
    
    # need
    samples_need = df["sample"].to_list()
    
    # pat_samples
    samples_pat_samples = {
        k: samples_head_tail.get("head", list()) + samples_group[k] + samples_head_tail.get("tail", list()) for k in samples_group
    }
    
    # pat_last
    samples_pat_last = {
        k: samples_group[k][-1] for k in samples_group
    }

    # pat
    ks = list(samples_group.keys())
    assert len(samples_pat_samples[ks[0]]) == len(samples_pat_samples[ks[1]]), "sample has not same length: {}, {}".format(*samples_pat_samples)
    samples_len = len(samples_pat_samples[ks[0]])
    samples_pat = [
        padding_false(True, samples_len),
        padding_false(False, samples_len)
    ]
    samples_pat_type = ["CO", "OC"]
    
    return samples_pat_type, samples_pat, samples_pat_samples, samples_need, samples_pat_last, samples_time, samples_head_tail, samples_group, list(df["tag_group"].unique())[important_index]


def exec_cmd(cmd: str, log_file: str = None, log_file_mode: str = None):
    pipe = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.STDOUT ,shell=True,universal_newlines=True)
    if log_file is None:
        while True:
            buff = pipe.stdout.readline()
            if (buff == "") and (pipe.poll() != None):
                break
        return
    with open(log_file, log_file_mode, encoding="UTF-8") as f:
        while True:
            buff = pipe.stdout.readline()
            if (buff == "") and (pipe.poll() != None):
                break
            buff = buff.strip()
            if buff == "":
                continue
            f.write("{}\n".format(buff))


def calc_bed_signal(bed_file: str, bw_file: str, signal_dir: str, tmp_dir: str, force: bool = False):
    """ 根据bed计算信号值
    """

    res_signal_file = os.path.join(signal_dir, os.path.basename(bw_file).replace(".bw", ".txt"))
    tmp_signal_file = os.path.join(tmp_dir, os.path.basename(bw_file).replace(".bw", ".signal.txt"))
    tmp_bed_file = os.path.join(tmp_dir, os.path.basename(bw_file).replace(".bw", ".calc.bed"))

    # 已运行过就不再运行
    if os.path.exists(res_signal_file) and (not force):
        return res_signal_file
    
    os.system("cat -n " + bed_file + """ | awk -F "\t" '{if(NR>1){print $2"\t"$3"\t"$4"\t"$1}}' > """ + tmp_bed_file)
    
    os.system(" ".join(["bigWigAverageOverBed", bw_file, tmp_bed_file, tmp_signal_file, "> /dev/null 2>&1"]))
    os.system("""cat """+tmp_signal_file+""" | sort -k1,1n | awk -F "\t" '{print $5}' > """ + res_signal_file)
    
    with open(res_signal_file) as f:
        res_data = np.asarray(list(map(
            lambda dd: float(dd.strip()),
            f
        )))

    os.system(" ".join(["rm", "-rf", tmp_signal_file]))
    os.system(" ".join(["rm", "-rf", tmp_bed_file]))
    return res_data


def extend_bed(bed_file: str, bed_extend: int, result_dir: str, tmp_dir: str, force: bool = False):
    """ 将bed扩大
    """

    tmp_bed_file = os.path.join(tmp_dir, os.path.basename(bed_file))
    res_bed_file = os.path.join(result_dir, "{}bp.{}".format(bed_extend, os.path.basename(bed_file)))

    
    # 已运行过就不再运行
    if os.path.exists(res_bed_file) and (not force):
        return res_bed_file

    tmp_data = list()
    with open(bed_file) as fr:
        for line in fr:
            line_data = line.strip().split("\t")
            chr_temp, start_temp, end_temp = line_data[0], line_data[1], line_data[2]
            start_temp = int(start_temp)
            end_temp = int(end_temp)
            tmp_data.append("{}\t{}\t{}".format(
                chr_temp,
                0 if start_temp < bed_extend else start_temp - bed_extend,
                end_temp + bed_extend
            ))
    with open(tmp_bed_file, "w", encoding="UTF-8") as fw:
        fw.write("\n".join(tmp_data))
    # os.system(" ".join([
    #     "cat",
    #     bed_file,
    #     "| awk -F \"\\t\"",
    #     """\'{print $1"\t"($2-"""+str(bed_extend)+""")"\t"($3+"""+str(bed_extend)+""")}\'""",
    #     ">",
    #     tmp_bed_file
    # ]))
    os.system(" ".join([
        "cat",
        tmp_bed_file,
        "| sort -k1,1 -k2,2n | bedtools merge -i - >",
        res_bed_file
    ]))
    os.system(" ".join(["rm", "-rf", tmp_bed_file]))
    return res_bed_file


def bed_combine(bed_file: list[str], ext_len: int, result_dir: str, tmp_dir: str, need_head: bool = True, file_name_tag: str = "", force: bool = False):
    """ 将bed合并
    """
    
    res_txt_file = os.path.join(result_dir, "{}merge.ov.bed".format(file_name_tag))
    tmp_cat_file = os.path.join(tmp_dir, "cat.bed")
    tmp_merge_file = os.path.join(tmp_dir, "merge.bed")

    # 已运行过就不再运行
    if os.path.exists(res_txt_file) and (not force):
        return res_txt_file
    
    # merge
    os.system("cat {} > {}".format(
        " ".join(bed_file),
        tmp_cat_file
    ))
    os.system("cut -f 1-3 {} | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
        tmp_cat_file,
        tmp_merge_file
    ))

    cmd = ""
    header = list()
    for bed_file_temp in bed_file:
        cmd = "{}| intersectBed -c -a - -b {}".format(
            cmd,
            bed_file_temp
        )
        if need_head:
            header.append(os.path.basename(bed_file_temp)[:-ext_len].split(".",1)[-1])
    cmd = "cat {} {} > {}".format(
        tmp_merge_file,
        cmd,
        res_txt_file
    )
    os.system(cmd)

    with open(tmp_cat_file, "w") as f:
        if need_head:
            f.write("\t".join(["chr", "start", "end"] + header))
        f.write("\n")
    
    os.system("cat {} {} > {}".format(
        tmp_cat_file,
        res_txt_file,
        tmp_merge_file
    ))
    os.system("cat {} > {}".format(
        tmp_merge_file,
        res_txt_file
    ))

    # for bed_file_temp in bed_file:
    #     os.system(" ".join(["rm", "-rf", bed_file_temp]))
    os.system(" ".join(["rm", "-rf", tmp_cat_file]))
    os.system(" ".join(["rm", "-rf", tmp_merge_file]))

    return res_txt_file


def cluster_samples(signal_file: str, samples_pat_type: list, samples_pat: list, samples_pat_samples: dict, samples_need: list, samples_pat_last: list, bw_dir: list[str], result_dir: str, signal_dir: str, tmp_dir: str, force: bool = False):
    """ 将样本分类
    """

    res_num_file = os.path.join(result_dir, "cluster.num.{}".format(os.path.basename(signal_file)))
    res_signal_file = os.path.join(result_dir, "cluster.signal.{}".format(os.path.basename(signal_file)))
    res_signal_file_ori = os.path.join(result_dir, "cluster.signal_ori.{}".format(os.path.basename(signal_file)))

    # 已运行过就不再运行
    if os.path.exists(res_signal_file) and (not force):
        return res_signal_file, res_signal_file_ori
    
    # 读取信息
    def get_cluster(se: pd.Series):
        nonlocal samples_pat_type, samples_pat, samples_pat_samples, samples_pat_last
        for ii in samples_pat_samples:
            pat_now = "".join(map(
                lambda d: "0" if d == 0 else "1",
                se[samples_pat_samples[ii]]
            ))
            se['last_{}'.format(ii)] = "0" if se[samples_pat_last[ii]] == 0 else "1"
            se['cluster_{}'.format(ii)] = "Other"
            for i in range(len(samples_pat)):
                for index, pat_temp in enumerate(samples_pat[i], start=1):
                    if pat_temp != pat_now:
                        continue
                    se['cluster_{}'.format(ii)] = "{}{}".format(samples_pat_type[i], index)
                    break
                if se['cluster_{}'.format(ii)] != "Other":
                    break
        return se
    
    if not os.path.exists(res_num_file):
        df_peak_signal = pd.read_table(signal_file)
        df_peak_signal = df_peak_signal.apply(get_cluster, axis=1)
        df_peak_signal.to_csv(
            res_num_file,
            index=False,
            sep="\t"
        )
    else:
        df_peak_signal = pd.read_table(res_num_file)
    
    # bw file
    files_bw_dict = dict()
    for i in samples_need:
        for d in bw_dir:
            file_temp = os.path.join(d, "{}.bw".format(i))
            if not os.path.exists(file_temp):
                continue
            files_bw_dict[i] = file_temp
            break
        assert i in files_bw_dict, "No bw file: {}, {}".format(i, bw_dir)

    # 计算每一列的signal
    # original
    for i in samples_need:
        df_peak_signal[i] = calc_bed_signal(
            bed_file=res_num_file,
            bw_file=files_bw_dict[i],
            signal_dir=signal_dir,
            tmp_dir=tmp_dir,
            force=force
        ).tolist()
    df_peak_signal.to_csv(
        res_signal_file_ori,
        index=False,
        sep="\t"
    )
    # normalized
    for i in samples_need:
        df_peak_signal[i] = preprocessing.scale(df_peak_signal[i]).tolist()
        df_peak_signal[i] = df_peak_signal[i] - df_peak_signal[i].min()
    df_peak_signal.to_csv(
        res_signal_file,
        index=False,
        sep="\t"
    )

    return res_signal_file, res_signal_file_ori


def draw_image(signal_file: str, sample_group_now: str, min_percent: float, max_percent: float, samples_pat_samples: dict, result_dir: str, remove_cluster: list, sample_time: list, force: bool = False):
    """ 画一张图
    """
    title_name = sample_group_now
    res_file = os.path.join(result_dir, "{}{}.pdf".format(title_name, ".remove_cluster" if len(remove_cluster) > 0 else ""))

    
    # 已运行过就不再运行
    if os.path.exists(res_file) and (not force):
        return res_file

    df = pd.read_table(signal_file)
    cluster_name = "cluster_{}".format(sample_group_now)
    df = df[df[cluster_name] != "Other"]


    # 去除不要的类
    if len(remove_cluster) > 0:
        remove_cluster = set(remove_cluster)
        df = df[~df[cluster_name].isin(remove_cluster)]

    df.sort_values([cluster_name], inplace=True)
    df.set_index(cluster_name, inplace=True)
    df_temp = df[samples_pat_samples[sample_group_now]]
    # def get_score(se: pd.Series):
    #     y = np.array(se.to_list())
    #     x = np.array(range(1, y.shape[0]+1))
    #     theta, _, _, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)
    #     se['score'] = -round(theta.flatten()[0], 1)
    #     return se
    
    # df_temp = df_temp.apply(get_score, axis=1)
    df_temp.reset_index(inplace=True)
    df_temp_co = df_temp[df_temp[cluster_name].str.startswith("CO")]
    df_temp_oc = df_temp[df_temp[cluster_name].str.startswith("OC")]
    df_temp = pd.concat((
        df_temp_co.sort_values(cluster_name),
        df_temp_oc.sort_values(cluster_name),
    ))
    # df_temp.drop("score", inplace=True, axis=1)
    df_temp.set_index(cluster_name, inplace=True)
    cutoff_max = -100
    cutoff_min = 100
    for i in samples_pat_samples[sample_group_now]:
        cutoff_max = max(cutoff_max, df_temp[i].quantile(max_percent))
        cutoff_min = min(cutoff_max, df_temp[i].quantile(min_percent))
    

    x_n = len(samples_pat_samples[sample_group_now])
    fig = plt.figure(figsize=(x_n*0.5, 14))
    grid = plt.GridSpec(16, 24, hspace=3, wspace=0) 
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_ax = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_ax = fig.add_subplot(grid[-1, 5:19], yticklabels=[], sharex=y_ax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("#030104", "#d57237", "#ecec9e", "#ffffcc"))
    sns.heatmap(
        df_temp,
        cmap=cmap,
        vmax=cutoff_max,
        vmin=cutoff_min,
        ax=main_ax,
        # cbar=None,
        cbar_ax=x_ax,
        cbar_kws={
            "orientation": "horizontal",
        },
        rasterized=True
    )
    x_ax.spines[:].set_visible(True)
    x_ax.set_xticks(
        [cutoff_min, cutoff_max],
        ["{:.1f}".format(cutoff_min), "{:.1f}".format(cutoff_max)]
    )
    x_ax.set_xlabel("Normalized Signal")
    labels = dict(enumerate(df_temp.index))
    main_ax.set_title(title_name)
    main_ax.xaxis.set_ticklabels(ticklabels=sample_time, rotation=0)
    _ = main_ax.set_ylabel("")
    _ = main_ax.yaxis.set_tick_params(left=False)
    
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
    # g = label_max * 0.005
    g = min(map(lambda d: len(d), label_dict.values())) / 3
    l = g
    label_dict_labels_posi = list()
    label_dict_labels_lab = list()
    y_ax_x_posi = sum(y_ax.get_xlim())/2
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
        y_ax.plot(
            [y_ax_x_posi, y_ax_x_posi],
            d_t,
            c='black'
        )
    # y_ax.yaxis.set_ticks(
    #     label_dict_labels_posi,
    #     label_dict_labels_lab
    # )
    y_ax.yaxis.set_ticks([])
    for p, l in zip(label_dict_labels_posi, label_dict_labels_lab):
        y_ax.text(-0.5, p, l, ha="right", va="center")
    _ = y_ax.yaxis.set_tick_params(left=False)
    y_ax.axis("off")
    y_ax.invert_yaxis()
    plt.savefig(res_file, dpi=500, bbox_inches='tight')
    # plt.savefig(res_file.replace(".pdf", ".png"), dpi=5000, bbox_inches='tight')
    plt.close()


def draw_image_by_one(signal_file: str, important_group: str, min_percent: float, max_percent: float, samples_pat_samples: dict, samples_group: dict, result_dir: str, remove_cluster: list, remove_same_cluster: bool, sample_time: list, samples_head_tail: dict, force: bool = False):
    """ 画一张图
    """
    title_name = "Align_by_{}".format(important_group)
    res_file = os.path.join(result_dir, "{}{}.pdf".format(title_name, ".remove_cluster" if len(remove_cluster) > 0 else ""))

    
    # 已运行过就不再运行
    if os.path.exists(res_file) and (not force):
        return res_file

    df = pd.read_table(signal_file)
    cluster_name = "cluster_{}".format(important_group)
    df = df[df[cluster_name] != "Other"]

    # 去除不要的类
    if len(remove_cluster) > 0:
        remove_cluster = set(remove_cluster)
        df = df[~df[cluster_name].isin(remove_cluster)]

    df.sort_values([cluster_name], inplace=True)
    # df.set_index(cluster_name, inplace=True)

    important_less_group = list(samples_pat_samples.keys())
    important_less_group.remove(important_group)
    important_less_group = important_less_group[0]
    samples_now_need = set(samples_pat_samples[important_group]).union(set(samples_pat_samples[important_less_group]))
    
    df_temp = df
    def get_score1(se: pd.Series):
        y = np.array(se.to_list())
        x = np.array(range(1, y.shape[0]+1))
        theta, _, _, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)
        return -round(theta.flatten()[0]/100, 1)
    df_temp['score1'] = df_temp[samples_pat_samples[important_group]].apply(get_score1, axis=1)
    # def get_score2(se: pd.Series):
    #     y = np.array(se.to_list())
    #     x = np.array(range(1, y.shape[0]+1))
    #     theta, _, _, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)
    #     return -round(theta.flatten()[0], 6)
    # important_index_less = list(range(len(samples_pat_samples)))
    # important_index_less.remove(important_index)
    # important_index_less = important_index_less[0]
    # df_temp['score2'] = df_temp[samples_pat_samples[important_index_less]].apply(get_score2, axis=1)
    last_key = "last_{}".format(important_less_group)
    df_temp[last_key] = df_temp[last_key].astype("str")
    df_temp['score2'] = (df_temp[cluster_name].map(lambda d: "0" if d.startswith("OC") else "1") != df_temp[last_key]).astype("int")
    
    # df_temp.reset_index(inplace=True)
    df_temp_co = df_temp[df_temp[cluster_name].str.startswith("CO")]
    df_temp_oc = df_temp[df_temp[cluster_name].str.startswith("OC")]
    # df_temp_oc['score'] = -df_temp_oc['score']
    df_temp = pd.concat((
        df_temp_co.sort_values([cluster_name, "score1", "score2"]),
        df_temp_oc.sort_values([cluster_name, "score1", "score2"]),
    ))
    df_temp.set_index(cluster_name, inplace=True)
    df_temp = df_temp[list(samples_now_need)]
    # df_temp.drop(["score1", "score2"], inplace=True, axis=1)
    cutoff_max = -10000
    cutoff_min = 10000
    for i in samples_now_need:
        cutoff_max = max(cutoff_max, df_temp[i].quantile(max_percent))
        cutoff_min = min(cutoff_max, df_temp[i].quantile(min_percent))
    
    cluster_list = [
        samples_head_tail.get("head", list()),
        samples_group[important_group],
        samples_group[important_less_group],
        samples_head_tail.get("tail", list())
    ]
    cluster_list_name = [
        "head",
        important_group,
        important_less_group,
        "tail"
    ]
    cluster_list_len_sum = sum(map(
        lambda d: len(d),
        cluster_list
    ))
    x_n = cluster_list_len_sum + len(cluster_list)
    grid_min = 2
    fig = plt.figure(figsize=(x_n*0.5, 12))
    grid = plt.GridSpec(16, cluster_list_len_sum*2+grid_min, hspace=1, wspace=0.8)
    main_ax_list = list()
    is_first = True
    for i in range(len(cluster_list)):
        if len(cluster_list[i]) == 0:
            continue
        grid_min_temp = grid_min + len(cluster_list[i]) * 2
        if is_first:
            ax_t = fig.add_subplot(grid[:-2, grid_min:grid_min_temp])
            is_first = False
        else:
            ax_t = fig.add_subplot(grid[:-2, grid_min:grid_min_temp], xticklabels=[], sharey=main_ax_list[0])
        main_ax_list.append(ax_t)
        grid_min = grid_min_temp
    y_ax = fig.add_subplot(grid[:-2, 0:2], xticklabels=[], sharey=main_ax_list[0])
    x_ax = fig.add_subplot(grid[-1, 11:27], yticklabels=[], sharex=y_ax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("#030104", "#d57237", "#ecec9e", "#ffffcc"))

    def main_ax_single(i: int, name: str, sample_list_now: list):
        nonlocal df_temp, cmap, cutoff_min, cutoff_max, main_ax_list, x_ax, sample_time, samples_pat_samples

        main_ax = main_ax_list[i]
        df_temp_now = df_temp[sample_list_now]

        if i == 0:
            sns.heatmap(
                df_temp_now,
                cmap=cmap,
                vmax=cutoff_max,
                vmin=cutoff_min,
                ax=main_ax,
                cbar_ax=x_ax,
                cbar_kws={
                    "orientation": "horizontal",
                },
                rasterized=True
            )
        else:
            sns.heatmap(
                df_temp_now,
                cmap=cmap,
                vmax=cutoff_max,
                vmin=cutoff_min,
                ax=main_ax,
                cbar=None,
                rasterized=True
            )
        
        if name not in ("head", "tail"):
            main_ax.xaxis.set_ticklabels(ticklabels=sample_time, rotation=0)
            _ = main_ax.set_xlabel(name)
        else:
            _ = main_ax.set_xlabel("")
            # _ = main_ax.set_xlabel(name)
        _ = main_ax.xaxis.set_tick_params(left=False)
        _ = main_ax.set_ylabel("")
        _ = main_ax.yaxis.set_tick_params(left=False)
    ii = 0
    for i in range(len(cluster_list)):
        if len(cluster_list[i]) == 0:
            continue
        main_ax_single(ii, cluster_list_name[i], cluster_list[i])
        ii += 1

    x_ax.spines[:].set_visible(True)
    x_ax.set_xticks(
        [cutoff_min, cutoff_max],
        ["{:.1f}".format(cutoff_min), "{:.1f}".format(cutoff_max)]
    )
    x_ax.set_xlabel("Normalized Signal")
    labels = dict(enumerate(df_temp.index))
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
    # g = label_max * 0.005
    g = min(map(lambda d: len(d), label_dict.values())) / 4
    l = g
    label_dict_labels_posi = list()
    label_dict_labels_lab = list()
    y_ax_x_posi = max(y_ax.get_xlim())
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
        y_ax.plot(
            [y_ax_x_posi, y_ax_x_posi],
            d_t,
            c='black'
        )
    y_ax.yaxis.set_ticks([])
    for p, l in zip(label_dict_labels_posi, label_dict_labels_lab):
        y_ax.text(0, p, l, va="center", ha="right",
                #   fontsize=4
        )
    _ = y_ax.yaxis.set_tick_params(left=False)
    y_ax.axis("off")
    y_ax.invert_yaxis()
    plt.savefig(res_file, dpi=3000, bbox_inches='tight')
    # plt.savefig(res_file.replace(".pdf", ".png"), dpi=500, bbox_inches='tight')
    plt.close()


def draw_image_by_one_sankey(signal_file: str, important_group: str, samples_pat_samples: dict, result_dir: str, remove_cluster: list, sample_time, samples_head_tail, force: bool = False):
    """ 画一张图
    """
    title_name = "Align_by_{}".format(important_group)
    res_file = os.path.join(result_dir, "{}.sankey{}.html".format(title_name, ".remove_cluster" if len(remove_cluster) > 0 else ""))
    res_file2 = os.path.join(result_dir, "{}.bar{}.pdf".format(title_name, ".remove_cluster" if len(remove_cluster) > 0 else ""))

    
    # 已运行过就不再运行
    if os.path.exists(res_file2) and (not force):
        return res_file2

    
    df = pd.read_table(signal_file)
    
    cols_need = list(filter(
        lambda d: d.startswith("cluster_"),
        df.columns
    ))
    col_important = "cluster_{}".format(important_group)
    cols_need.remove(col_important)
    col_important_less = cols_need[0]
    cols_need.append(col_important)

    # df = df[cols_need]
    se_bool = df[cols_need[0]] != "Other"
    for i in range(1, len(cols_need)):
        se_bool = se_bool | (df[cols_need[i]] != "Other")
    df = df[se_bool]
    
    # 去除不要的类
    if len(remove_cluster) > 0:
        remove_cluster = set(remove_cluster)
        for i in cols_need:
            df = df[~df[col_important].isin(remove_cluster)]

    data = list()
    for i in samples_pat_samples:
        df["cluster_{}".format(i)] = df["cluster_{}".format(i)].map(lambda d: "{}_{}".format(i, d))
        

    for (col_important_name, col_important_less_name), df_temp in df.groupby([col_important, col_important_less]):
        data.append({
            "source": col_important_name,
            "target": col_important_less_name,
            "value": len(df_temp)
        })
    data.sort(key=lambda d: d['source'][12:])

    p = Plot("Sankey")
    p.set_options({
        "appendPadding": 50,
        "data": data,
        "sourceField": 'source',
        "targetField": 'target',
        "weightField": 'value',
        "nodeWidthRatio": 0.008,
        "nodePaddingRatio": 0.03,
        "nodeDraggable": True,
        "nodeSort": JS("""(a, b) => {
    try{
        const tg_a = a.name.split('_')[1].startsWith('OC')*20-10;
        const tg_b = b.name.split('_')[1].startsWith('OC')*20-10;
        const t_a = parseInt(a.name.split('_')[1].substr(2,1));
        const t_b = parseInt(b.name.split('_')[1].substr(2,1));
        return (t_b + tg_b) - (t_a + tg_a);
    }catch(e){
        return -1;
    }
}"""),
        "linkSort": JS("""(a, b) => b.value - a.value""")
    })
    p.render(res_file)
    with open(res_file, "a") as f:
        f.write("""
<style>
h3{
text-align: center;
position: absolute;
top: 10px;
right: 50px;
}
</style>
<script>
document.title = \"""" + title_name + """\";
document.body.insertAdjacentHTML("beforeEnd", "<h3>""" + title_name + """</h3>");
</script>""")


    data = list()
    last_key = col_important_less.replace("cluster", "last")
    df[last_key] = df[last_key].astype("str")
    for i, df_temp in df.groupby(col_important):
        c_tmp = i.rsplit("_",1)[1]
        if c_tmp == "Other":
            continue
        c_tmp_tmp = "0" if c_tmp.startswith("OC") else "1"
        s_t = len(df_temp[df_temp[last_key] == c_tmp_tmp])
        f_t = len(df_temp) - s_t
        data.append({
            "cluster": c_tmp,
            "success": s_t,
            "fail": f_t
        })
    df = pd.DataFrame(data)
    df['sum'] = df["fail"] + df["success"]
    df['fail_scale'] = df['fail'] / df['sum'] * 100
    df['success_scale'] = df["success"] / df['sum'] * 100
    df.sort_values("cluster", inplace=True)

    fig = plt.figure(figsize=(len(df['cluster'].unique())*0.8+2, 6))
    ax = plt.subplot()
    width = 0.7
    ax.bar(
        np.arange(len(df)),
        (df['success_scale'] + df['fail_scale']).to_numpy(),
        width=width,
        tick_label=df['cluster'],
        label="fail",
        color="#bb0000"
    )
    ax.bar(
        np.arange(len(df)),
        df['success_scale'].to_numpy(),
        width=width,
        tick_label=df['cluster'],
        label="success",
        color="#4d87de"
    )
    for x, (_, y) in zip(np.arange(len(df)), df.iterrows()):
        plt.text(x, 104, "{:.1f}%".format(y["fail"] / y["sum"] * 100), va="center", ha="center")
        plt.text(x, y['fail_scale'] / 2 + y["success_scale"], "{}".format(y["fail"]), va="top", ha="center", color="white")
        plt.text(x, y['success_scale'] / 2, "{}".format(y["success"]), va="bottom", ha="center", color="white")
    ax.set_ylim(0, 110)
    ax.set_yticks([])
    ax.set_title(title_name)
    ax.legend()
    plt.legend(bbox_to_anchor=(1, 0.5), loc=2)
    fig.savefig(
        res_file2,
        dpi=200
    )
    plt.close()


def get_genes_from_bed(bed_all_file: str, samples_pat_samples: dict, promoter_file: str, tf_file: str, important_group: str, result_dir: str, logger: logging.Logger):

    result_dir_bed = os.path.join(result_dir, "cluster")
    result_dir_bed_fail = "{}_fail".format(result_dir_bed)

    if not os.path.exists(result_dir_bed):
        os.makedirs(result_dir_bed)
    if not os.path.exists(result_dir_bed_fail):
        os.makedirs(result_dir_bed_fail)

    logger.info("get bed from bed ...")

    df = pd.read_table(bed_all_file)
    for i in samples_pat_samples:
        n = i
        i = "cluster_{}".format(i)
        for t, df_temp in df.groupby(i):
            if t == "Other":
                continue
            # if t in remove_cluster:
            #     continue
            file_name_temp = os.path.join(result_dir_bed, "{}_{}.bed".format(n, t))
            df_temp[["chr", "start", "end"]].to_csv(
                file_name_temp,
                header=False,
                index=False,
                sep="\t"
            )

    result_dir_gene = bed_to_gene(
        bed_dir=result_dir_bed,
        promoter_file=promoter_file,
        tf_file=tf_file,
        logger=logger
    )
    
    cols_need = list(filter(
        lambda d: d.startswith("cluster_"),
        df.columns
    ))
    col_important = "cluster_{}".format(important_group)
    cols_need.remove(col_important)
    col_important_less = cols_need[0]
    cols_need.append(col_important)
    
    last_key = col_important_less.replace("cluster", "last")
    df[last_key] = df[last_key].astype("str")
    for (col_important_name, last_less), df_temp in df.groupby([col_important, last_key]):
        if col_important_name == "Other":
            continue
        # if col_important_name in remove_cluster:
        #     continue
        last_aim = "0" if col_important_name.startswith("OC") else "1"
        if str(last_less) == last_aim:
            continue
        file_name_temp = os.path.join(result_dir_bed_fail, "{}_{}.bed".format(important_group, col_important_name))
        df_temp.to_csv(
            file_name_temp,
            header=False,
            index=False,
            sep="\t"
        )
    del df

    result_dir_gene_fail = bed_to_gene(
        bed_dir=result_dir_bed_fail,
        promoter_file=promoter_file,
        tf_file=tf_file,
        logger=logger
    )

    return result_dir_bed, result_dir_gene, result_dir_bed_fail, result_dir_gene_fail


def bed_to_gene(bed_dir: str, promoter_file: str, tf_file: str, logger: logging.Logger):
    result_dir_gene = "{}_genes".format(bed_dir)
    gene_summary_file = "{}_gene_summary.txt".format(bed_dir)
    if not os.path.exists(result_dir_gene):
        os.makedirs(result_dir_gene)

    # 打开各个bed
    logger.info("get gene ...")
    data_summary = list()
    df_tfs = pd.read_table(tf_file)
    tfs = set(df_tfs['Symbol'].to_list())
    del df_tfs
    get_sample_gene = lambda f: set(filter(
        lambda dd: dd != "",
        map(
            lambda d: d.strip(),
            f
        )
    ))

    for file_single in os.listdir(bed_dir):
        file_single = os.path.join(bed_dir, file_single)
        region_num = 0
        with open(file_single) as f:
            for line in f:
                if line.strip() == "":
                    break
                region_num += 1
        tmp = os.path.join(result_dir_gene, "temp.bed")
        type_temp = os.path.basename(file_single)[:-4]
        gene_file_temp = os.path.join(result_dir_gene, "{}.txt".format(type_temp))
        os.system("intersectBed -wa -a {} -b {} | sort | uniq > {}".format(
            promoter_file,
            file_single,
            tmp
        ))
        os.system("""cat """ + tmp + """ | awk -F '\t' '{print $5}' | sort | uniq > """ + gene_file_temp)
        os.system("rm -rf {}".format(tmp))
        # tf
        with open(gene_file_temp) as f:
            gene_temp = get_sample_gene(f)
        tf_temp = tfs & gene_temp
        with open(gene_file_temp.replace(".txt", ".tf.txt"), "w") as f:
            f.write("\n".join(tf_temp))
        data_summary.append({
            "type": type_temp,
            "region": region_num,
            "gene": len(gene_temp),
            "tf": len(tf_temp)
        })
    df = pd.DataFrame(data_summary)
    df.sort_values("type", inplace=True)
    df.to_csv(gene_summary_file, index=False, sep="\t")
    
    return result_dir_gene


def plot_nums_of_bed(cluster_bed_dir: str, result_dir: str, remove_cluster: list, force: bool = False):

    res_file = os.path.join(result_dir, "number_of_peaks{}.pdf".format(
        ".remove_cluster" if len(remove_cluster) > 0 else ""
    ))
    # 已运行过就不再运行
    if os.path.exists(res_file) and (not force):
        return

    df = pd.DataFrame(list(map(
        lambda d: (*d[:-4].rsplit("_",1), d),
        os.listdir(cluster_bed_dir)
    )), columns=["sample", "type", "file"])

    df = df[df['type'].str.len() > 2]

    # 去除不要的类
    if len(remove_cluster) > 0:
        remove_cluster = set(remove_cluster)
        df = df[~df['type'].isin(remove_cluster)]


    df.sort_values(["sample", "type"], inplace=True)

    def get_bed_num(file_name: str):
        nonlocal cluster_bed_dir
        line_num = 0
        with open(os.path.join(cluster_bed_dir, file_name)) as f:
            for line in f:
                if line.strip() == "":
                    continue
                line_num += 1
        return line_num
    
    df['num'] = df['file'].map(get_bed_num)
    
    fig = plt.figure(figsize=(len(df['type'].unique())*0.5+1, 6))
    colors = ["#16557A","#C7A609"]
    ax = sns.barplot(
        df,
        x="type",
        y="num",
        hue="sample",
        palette=colors
    )
    # y_dict = {i.get_position()[1]: int(np.power(10, float(i.get_text()))) for i in ax.get_yticklabels()}
    # ax.set_yticks(
    #     list(y_dict.keys()),
    #     list(y_dict.values())
    # )

    sample_loc = {i.get_text():i.get_position()[0] for i in ax.get_xticklabels()}
    g = max(map(
        lambda d: d.get_position()[1],
        ax.get_yticklabels()
    ))*0.02
    for s, df_t in df.groupby("type"):
        for i in range(2):
            plt.text(
                x=sample_loc[s]+(i*0.45-0.2),
                y=df_t.iloc[i]['num']+g,
                s=str(df_t.iloc[i]['num']),
                c=colors[i],
                rotation=90,
                ha="center"
            )
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Number of peaks")
    plt.legend(title=None)
    plt.savefig(res_file, dpi=500, bbox_inches='tight')
    plt.close()


def signal_single_boxplot(df: pd.DataFrame, file_name: str, keys_compare: list, logger: logging.Logger, ylabel: str = "Signal Z-Score"):
    df_now = df.stack().reset_index()
    df_now.columns = ["peak", "sample", ylabel]
    df_now['day'] = df_now['sample'].map(
        lambda d: d.split("_", 1)[1]
    )
    df_now['type'] = df_now['sample'].map(
        lambda d: d.split("_", 1)[0]
    )
    df_now['order'] = df_now['day'].map(lambda d: int(d[1:]))
    df_now.sort_values(["order", 'type'], inplace=True)
    keys_compare = list(keys_compare)
    keys_compare.sort()
    
    f = plt.figure(figsize=(18, 4))
    col = "k"
    f.add_subplot(1,2,1)
    if len(keys_compare) == 3:
        colors = ["#CE0013","#16557A","#C7A609"]
    else:
        colors = ["#16557A","#C7A609"]
    fig = sns.boxplot(x="day", y=ylabel, hue="type", data=df_now, palette=colors, showfliers=False)
    fig.set_ylim(df_now[ylabel].min() - 0.5, df_now[ylabel].max() + len(keys_compare)*3 - 4)
    fig.set_title(os.path.basename(file_name).rsplit(".",1)[0])
    fig.set_xlabel("")

    def t_test(df_now_day, type_list, day):
        nonlocal ylabel
        d1 = df_now_day[(df_now_day['type'] == type_list[0]) & (df_now_day['day'] == day)][ylabel].to_numpy()
        d2 = df_now_day[(df_now_day['type'] == type_list[1]) & (df_now_day['day'] == day)][ylabel].to_numpy()
        # logger.info(type_list, day, d1.shape, d2.shape)
        t = stats.ttest_rel(d1, d2).pvalue
        # assert t > 0, "{} {} {}".format(type_list, day, t)
        if t == 0:
            logger.warning("T-test value is 0: {}, {}".format(day, np.all(d1 == d2)))
            return ""
        if t < 0.001:
            return "***"
        if t < 0.01:
            return "**"
        if t < 0.05:
            return "*"
        return "-"

    # statistical annotation
    y_max = -1000
    y_min = 1000
    for day, df_now_day in df_now.groupby("day"):
        x = list(map(
            lambda d: d.get_position()[0],
            filter(
                lambda d: d.get_text() == day,
                plt.xticks()[1]
            )
        ))[0]
        if len(keys_compare) == 3:
            # DR与Oct4Nanog
            t = t_test(df_now_day, (keys_compare[0], keys_compare[1]), day)
            x1, x2 = x - 0.27, x   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = df_now_day[ylabel].quantile(0.95) + 1, 0.1
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            # Oct4Nanog与Oct4NanogN70
            t = t_test(df_now_day, (keys_compare[1], keys_compare[2]), day)
            x1, x2 = x, x + 0.27   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = df_now_day[ylabel].quantile(0.95) + 1.5, 0.1
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            # Oct4Nanog与Oct4NanogN70
            t = t_test(df_now_day, (keys_compare[1], keys_compare[2]), day)
            x1, x2 = x - 0.27, x + 0.27   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = df_now_day[ylabel].quantile(0.95) + 2.5, 0.1
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            y_max = max(y_max, df_now_day[ylabel].quantile(0.95) + 2.5)
            y_min = min(y_min, df_now_day[ylabel].quantile(0))
        if len(keys_compare) == 2:
            t = t_test(df_now_day, keys_compare, day)
            x1, x2 = x - 0.2, x + 0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = df_now_day[ylabel].quantile(0.95) + 1, 0.1
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            y_max = max(y_max, df_now_day[ylabel].quantile(0.95) + 1)
            y_min = min(y_min, df_now_day[ylabel].quantile(0))
    g = (y_max - y_min) / 20
    fig.set_ylim(y_min-g, y_max+g*2)
    axis = f.add_subplot(1,1,1)
    tl1 = plt.scatter([0], [0], marker="$***$", color=col)
    tl2 = plt.scatter([0], [0], marker="$**$", color=col)
    tl3 = plt.scatter([0], [0], marker="$*$", color=col)
    tl4 = plt.scatter([0], [0], marker="$-$", color=col)
    plt.delaxes(axis)
    leg_t = plt.legend(
        handles=[tl1, tl2, tl3, tl4],
        labels=["p < 0.001", "p < 0.01", "p < 0.05", "p >= 0.05"],
        bbox_to_anchor=(1, 0.5), loc=2
    )
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.gca().add_artist(leg_t)
    plt.savefig(file_name, bbox_inches='tight', dpi=1000)
    # plt.show()
    plt.close()


def plot_enrich(enrich_dir: str, enrich_type: str, remove_cluster: list, logger: logging.Logger, padj: float = 0.01, gene_min: int = 2):
    
    end_str = ".{}Terms.txt".format(enrich_type)
    lim = len(end_str)
    go_file_all = list(map(
        lambda d: d[:-lim],
        filter(
            lambda d: d.endswith(end_str) and (len(d.rsplit("_",1)[1].split(".",1)[0]) > 2),
            os.listdir(enrich_dir)
        )
    ))
    if len(remove_cluster) > 0:
        go_file_all = list(filter(
            lambda d: not d.endswith(tuple(remove_cluster)),
            go_file_all
        ))
    go_file_all.sort()
    x_labels = list(set(map(
        lambda d: d.rsplit("_",1)[0],
        go_file_all
    )))
    x_labels.sort()
    
    # label_index = 0
    data = list()
    for i in go_file_all:
        file_name = os.path.join(enrich_dir, "{}.{}Terms.txt".format(
            i,
            enrich_type
        ))
        if not os.path.exists(file_name):
            continue
        # if i.rsplit("_",1)[0] != x_labels[label_index]:
        #     continue
        
        df_temp = pd.read_table(file_name)
        df_temp = df_temp[df_temp['p.adjust'] < padj]
        for term_temp, se in df_temp.iterrows():
            if len(se["genes"].split(",")) < gene_min:
                continue
            y_t = term_temp[1]
            if enrich_type == "KEGG":
                y_t = y_t.rsplit(" - ", 1)[0]
            data.append({
                "x": i,#.rsplit("_",1)[1],
                "y": y_t,
                "y_term": term_temp[0],
                "p.adjust": se["p.adjust"],
                "GeneRatio": eval(se["GeneRatio"]),
                "GeneNum": len(se["genes"].split(","))
            })
        del df_temp
        
    df_data = pd.DataFrame(data)

    if len(df_data) == 0:
        logger.warning("{} no data".format(enrich_type))
        return
    
    if len(df_data) == 0:
        logger.warning("{} no data by y filter".format(enrich_type))
        return
    df_data.to_csv(
        "{}.{}.txt".format(enrich_dir, enrich_type),
        sep="\t",
        index=False
    )
    
    df_data.sort_values(["x", "p.adjust"], inplace=True)
    ys = set(df_data[(df_data["p.adjust"] < padj) & (df_data["GeneNum"] >= gene_min)]["y"].unique())
    df_data = df_data[df_data["y"].isin(ys)]
    df_data["-log10(p.adjust)"] = -np.log10(df_data["p.adjust"])
    df_data = df_data[df_data["-log10(p.adjust)"] >= int(-np.log10(padj) - 1)]
    
    fig = plt.figure(figsize=(len(df_data['x'].unique())*0.5+1, len(df_data['y'].unique())*0.4))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("white", "red"))
    ax = sns.scatterplot(
        df_data,
        x="x",
        y="y",
        hue="-log10(p.adjust)",
        size="GeneRatio",
        sizes=(1, 200),
        palette=cmap,
    )
    _ = ax.xaxis.set_ticklabels(
        [i.get_text() for i in ax.xaxis.get_ticklabels()],
        rotation=90
    )
    # y轴
    y_ori_label = ax.yaxis.get_ticklabels()
    _ = ax.set_ylim(
        y_ori_label[0].get_position()[1] - 0.5,
        y_ori_label[-1].get_position()[1] + 0.5
    )
    # x轴
    # _ = ax.set_xlabel(x_labels[label_index])
    _ = ax.set_xlabel("")
    _ = ax.set_ylabel("")
    _ = ax.set_title(enrich_type)
    xlim_ori = ax.get_xlim()
    _ = ax.set_xlim(xlim_ori[0]-0.5, xlim_ori[1]+0.5)

    legend_t = ax.legend()
    legend_t_handlers = legend_t.legendHandles
    legend_t_texts = [i.get_text() for i in legend_t.texts]
    legend_t_texts[0] = "p.adjust"
    for i in range(1, len(legend_t_texts)):
        try:
            t =  -float(legend_t_texts[i])
            legend_t_texts[i] = "1e-{}".format(legend_t_texts[i])
        except:
            break

    plt.legend(
        legend_t_handlers,
        legend_t_texts,
        bbox_to_anchor=(1, 1),
        loc="upper left"
    )

    plt.savefig(os.path.join(
        enrich_dir,
        "..",
        "img",
        "{}_{}{}.pdf".format(
            os.path.basename(enrich_dir),
            enrich_type,
            ".remove_cluster" if len(remove_cluster) > 0 else ""
        )
    ), dpi=500, bbox_inches='tight')
    plt.close()


def enrich_genes(gene_dir: str, remove_cluster: list, species: str, force: bool, logger: logging.Logger):
    result_dir = "{}_enrich".format(gene_dir)
    gene_files = list(map(
        lambda d: os.path.join(gene_dir, d),
        filter(
            lambda d: not d.endswith(".tf.txt"),
            os.listdir(gene_dir)
        )
    ))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def run_enrich(file_single, result_dir, species, force):
        enrich_result_file = os.path.join(result_dir, os.path.basename(file_single)[:-4])
        
        if os.path.exists("{}.GOTerms.txt".format(enrich_result_file)) and (not force):
            return
        enrich_log_file = "{}.log".format(enrich_result_file)
        
        cmd = [
            os.path.join(os.path.dirname(__file__), "..", "..", "CDesk_cli"),
            "FunctionalEnrichment",
            result_dir,
            "species={}".format(species),
            "GeneSymbols={}".format(file_single),
            "> {} 2>&1".format(enrich_log_file)
        ]
        os.system(cmd)
        # logger.info(os.path.basename(file_single)[:-4])

    
    # 计算信号值
    error_file = os.path.join(result_dir, "error.log")
    def task_done(f: fu.Future):
        nonlocal error_file
        # 线程异常处理
        e = f.exception()
        if e:
            with open(error_file, "a") as f:
                f.write(traceback.format_exc())
                f.write("\n")
    task_all = list()
    with fu.ThreadPoolExecutor(max_workers=16) as executor:
        for file_single in gene_files:
            task_temp = executor.submit(
                run_enrich,
                file_single=file_single,
                result_dir=result_dir,
                species=species,
                force=force
            )
            # 任务回调
            task_temp.add_done_callback(task_done)
            task_all.append(task_temp)
        fu.wait(task_all)
    
    # plot
    remove_cluster_null = list()
    plot_enrich(result_dir, "GO", remove_cluster_null, logger, padj=0.01, gene_min=2)
    plot_enrich(result_dir, "KEGG", remove_cluster_null, logger, padj=0.01, gene_min=2)
    if len(remove_cluster) > 0:
        plot_enrich(result_dir, "GO", remove_cluster, logger, padj=0.01, gene_min=2)
        plot_enrich(result_dir, "KEGG", remove_cluster, logger, padj=0.01, gene_min=2)


def plot_motif(motif_dir: str, logger: logging.Logger, remove_cluster: list = None, remove_motifs: set = None, log_p_value: float = -2, log_p_value_point: float = -5, log_p_value_min: float = -50, ratio_min: float = 3):
    logger.info("Plot motif: {}".format(motif_dir))

    data = list()
    save_dir = os.path.join(motif_dir, "..", "img")

    if remove_motifs is None:
        remove_motifs = set()
    if remove_cluster is None:
        remove_cluster = list()

    name_handle = lambda d: d.replace(")", "").replace("?", "").replace(",,", ",").strip(",").strip()

    for type_single in os.listdir(motif_dir):
        kr = os.path.join(motif_dir, type_single, "knownResults.txt")
        if not os.path.exists(kr):
            continue
        df_temp = pd.read_table(kr)
        df_temp['Ratio'] = df_temp["% of Target Sequences with Motif"].map(
            lambda d: float(d[:-1])
        ) / df_temp["% of Background Sequences with Motif"].map(
            lambda d: float(d[:-1])
        )
        df_temp = df_temp[["Motif Name", "P-value", "Ratio"]]
        df_temp["Log P-value"] = df_temp["P-value"].map(lambda d: np.log10(d))
        for _, se in df_temp.iterrows():
            log_p_value_now = float(se["Log P-value"])
            r = float(se["Ratio"])
            m_n = se["Motif Name"].split("/", 1)[0]
            n_t = m_n.split("(", 1)
            t_t = name_handle(n_t[1]) if len(n_t) == 2 else ""
            s_t = type_single.rsplit("_", 1)
            data.append({
                "sample": s_t[0],
                "cluster": s_t[1] if len(s_t) == 2 else "Other",
                "sample_cluster": type_single,
                "motif": n_t[0],
                "type": t_t if len(t_t) > 0 else "Unknown",
                "Log P-value": log_p_value_now,
                "motif_name": m_n,
                "-log10(p-value)": -log_p_value_min if log_p_value_now < log_p_value_min else -log_p_value_now,
                "Ratio": r
            })
        del df_temp

    if len(data) == 0:
        logger.info("motif no data, by read ...")
        return
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df_data = pd.DataFrame(data)
    df_data.sort_values(["sample_cluster", "type", "motif"], inplace=True)
    df_data.to_csv("{}.txt".format(motif_dir), sep="\t", index=False)

    df_data_pivot = df_data.pivot_table(index="motif_name", columns="sample_cluster", values="-log10(p-value)")
    df_data_pivot.fillna(0, inplace=True)

    y_need = set(df_data[df_data["Log P-value"] < log_p_value]["motif_name"].unique()) - remove_motifs
    df_data = df_data[df_data["motif_name"].isin(y_need)]
    df_data_pivot = df_data_pivot[df_data_pivot.index.isin(y_need)]
    # df_data = df_data[df_data["Ratio"] != np.inf]

    if len(remove_cluster) > 0:
        df_data = df_data[~df_data["cluster"].isin(remove_cluster)]

        x_need = set(df_data["sample_cluster"].unique())
        df_data_pivot = df_data_pivot[list(filter(
            lambda d: d in x_need,
            df_data_pivot.columns
        ))]


    if len(df_data) == 0:
        logger.info("motif no data, by filter")
        return
    
    df_mask = df_data_pivot.copy()
    for i in df_mask.columns:
        df_mask[i] = df_mask[i].map(lambda d: d == 0)

    # group
    group_y = dict()
    group_x = dict()
    for x, df_temp in df_data.groupby("sample"):
        group_x[x] = list(df_temp["sample_cluster"].unique())
        group_x[x].sort()
        del df_temp
    for y, df_temp in df_data.groupby("type"):
        group_y[y] = list(df_temp["motif_name"].unique())
        group_y[y].sort()
        del df_temp
    group_x_keys = list(group_x.keys())
    group_x_keys.sort(reverse=True)
    group_x_keys = tuple(group_x_keys)
    group_y_keys = list(group_y.keys())
    group_y_keys.sort()
    group_y_keys = tuple(group_y_keys)

    fig_size = (
        len(df_data['sample_cluster'].unique())*0.5+1,
        len(df_data['motif_name'].unique())*0.5+1
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("white","#F2FAB0","#CE0013","#5E191A"))
    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=abs(log_p_value_min)
    )

    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(
        len(df_data['motif_name'].unique())+3,
        len(df_data['sample_cluster'].unique()),
        hspace=0.2,
        wspace=0.2,
        figure=fig,
    )
    
    ax_list = list()
    grid_min_y = 0
    for y in group_y_keys:
        grid_min_x = 0
        grid_min_y_temp = grid_min_y + len(group_y[y])
        ax_list_temp = list()
        for i, x in enumerate(group_x_keys):
            grid_min_x_temp = grid_min_x + len(group_x[x])
            grid_temp = grid[grid_min_y:grid_min_y_temp, grid_min_x:grid_min_x_temp]
            if i == 0:
                if len(ax_list) == 0:
                    ax_t = fig.add_subplot(
                        grid_temp,
                        xticklabels=[],
                        yticklabels=[],
                    )
                else:
                    ax_t = fig.add_subplot(
                        grid_temp,
                        xticklabels=[],
                        yticklabels=[],
                    )
            else:
                if len(ax_list) == 0:
                    ax_t = fig.add_subplot(
                        grid_temp,
                        # sharex=ax_list_temp[0],
                        xticklabels=[],
                        yticklabels=[],
                    )
                else:
                    ax_t = fig.add_subplot(
                        grid_temp,
                        # sharex=ax_list_temp[0],
                        xticklabels=[],
                        yticklabels=[],
                    )
            ax_list_temp.append(ax_t)
            grid_min_x = grid_min_x_temp
        grid_min_y = grid_min_y_temp
        ax_list.append(ax_list_temp)
    
    g = max(int(grid_min_x * 0.2), 1)
    color_ax = fig.add_subplot(grid[grid_min_y+2:, g:-g], xticklabels=[], yticklabels=[])

        
    mask_point = [[],[]]
    log_p_value_point_temp = -log_p_value_point
    for _, se in df_data.iterrows():
        # print(se["Log P-value"], se["Ratio"])
        if se["-log10(p-value)"] < log_p_value_point_temp:
            continue
        if se["Ratio"] < ratio_min:
            continue
        mask_point[0].append(se["sample_cluster"])
        mask_point[1].append(se["motif_name"])

    def get_lable(ss, s, tick_index, label_index, s_n = 1):
        nonlocal name_handle

        tick_labels = list()
        label_name = set()
        for i in ss:
            if s in i:
                t = i.rsplit(s,s_n)
                tick_labels.append(t[tick_index])
                label_name.add(name_handle(t[label_index]))
            else:
                tick_labels.append(i)
        return tick_labels, label_name.pop() if len(label_name) == 1 else ""

    
    def ax_single(y, x, y_max, x_max):
        nonlocal df_data_pivot, df_mask, cmap, norm, ax_list, group_x, group_y, group_x_keys, group_y_keys, mask_point, name_handle
        main_ax = ax_list[y][x]
        x_l = group_x_keys[x]
        y_l = group_y_keys[y]

        df_data_pivot_now = df_data_pivot.loc[group_y[y_l], group_x[x_l]]
        sns.heatmap(
            df_data_pivot_now,
            mask=df_mask.loc[list(df_data_pivot_now.index), list(df_data_pivot_now.columns)].to_numpy(),
            norm=norm,
            cmap=cmap,
            ax=main_ax,
            cbar=None,
            zorder=100,
            linecolor='#F0F0F0',
            linewidth=0.01,
            linestyle="--"
        )
        if (y_max - y) == 1:
            xtick_labels, xlabel_name = get_lable(df_data_pivot_now.columns, "_", 1, 0)
            main_ax.set_xticklabels(xtick_labels, rotation=0)
        else:
            xlabel_name = ""
            main_ax.xaxis.set_ticks([])
            main_ax.xaxis.set_major_locator(mpt.NullLocator())
        
        if x == 0:
            ytick_labels, ylabel_name = get_lable(df_data_pivot_now.index, "(", 0, 1)
            if x_max == 1:
                ylabel_name = name_handle(ylabel_name).replace(",", "\n")
                main_ax.yaxis.set_label_position("right")
                main_ax.set_ylabel(ylabel_name, rotation=0, va="center", ha="left")
            else:
                ylabel_name = ""
                main_ax.set_ylabel(ylabel_name, rotation=0)
            main_ax.set_yticklabels(ytick_labels, rotation=0)
        elif (x_max - x) == 1:
            ytick_labels, ylabel_name = get_lable(df_data_pivot_now.index, "(", 0, 1)
            ylabel_name = name_handle(ylabel_name).replace(",", "\n")
            main_ax.yaxis.set_ticks([])
            main_ax.yaxis.set_label_position("right")
            main_ax.set_ylabel(ylabel_name, rotation=0, va="center", ha="left")
        else:
            main_ax.yaxis.set_ticks([])
            ylabel_name = ""
            main_ax.set_ylabel(ylabel_name, rotation=0)

        # main_ax.grid(True, ls="--", lw=0.2, color='#F0F0F0', zorder=0)
        # main_ax.grid(True, ls="--", lw=0.5, color='black', zorder=0)

        x_half = 0
        x_s_start = df_data_pivot_now.columns[0].rsplit("_",1)[1][:2]
        for i in range(1, len(df_data_pivot_now.columns)):
            if df_data_pivot_now.columns[i].rsplit("_",1)[1][:2] == x_s_start:
                continue
            x_half = i
            break
        if (x_half > 0) and (x_half < (len(df_data_pivot_now.columns) - 1)):
            main_ax.plot(
                [x_half, x_half],
                [0, len(df_data_pivot_now.index)],
                ls="--",
                lw=0.8,
                c="black",
                zorder=110
            )
        
        y_dict = dict(map(lambda d: (d[1], d[0]), enumerate(df_data_pivot_now.index)))
        x_dict = dict(map(lambda d: (d[1], d[0]), enumerate(df_data_pivot_now.columns)))

        xxx = list()
        yyy = list()
        for i in range(len(mask_point[0])):
            if (mask_point[0][i] not in x_dict) or (mask_point[1][i] not in y_dict):
                continue
            xxx.append(x_dict[mask_point[0][i]]+0.5)
            yyy.append(y_dict[mask_point[1][i]]+0.5)
        main_ax.scatter(
            xxx,
            yyy,
            marker="*",
            c="black",
            zorder=110
        )

        main_ax.set_xlabel(xlabel_name)
        main_ax.yaxis.set_tick_params(left=False)
        main_ax.xaxis.set_tick_params(bottom=False)

        main_ax.spines[:].set_visible(True)
        main_ax.spines[:].set_zorder(110)

    for y, x in product(range(len(group_y_keys)), range(len(group_x_keys))):
        ax_single(y, x, len(group_y_keys), len(group_x_keys))
    
    def ax_single_corlor():
        nonlocal cmap, norm, log_p_value_point, log_p_value_min, color_ax

        plt.colorbar(
            mpm.ScalarMappable(
                norm=norm,
                cmap=cmap
            ),
            cax=color_ax,
            orientation='horizontal',
            label="-log10(p-value)",
        )
        color_ax.set_xticks(
            [0, -log_p_value_min],
            ["0", str(-log_p_value_min)]
        )
    ax_single_corlor()

    # plt.show()
    save_dir = os.path.join(motif_dir, "..", "img")
    plt.savefig(os.path.join(
        save_dir,
        "{}{}.pdf".format(
            os.path.basename(motif_dir),
            ".remove_cluster" if len(remove_cluster) > 0 else ""
        )
    ), dpi=500, bbox_inches='tight')
    plt.close()

    # return

    save_dir = os.path.join(save_dir, "motif")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for y_i, y in enumerate(group_y_keys):
        ax_list = [[] for _ in range(y_i+1)]
        fig_size = (
            len(df_data['sample_cluster'].unique())*0.5+1,
            len(group_y[y])*0.5+1
        )
        fig = plt.figure(figsize=fig_size)
        grid = plt.GridSpec(
            len(group_y[y])+4,
            len(df_data['sample_cluster'].unique()),
            hspace=0.2,
            wspace=0.2,
            figure=fig,
        )
        grid_min_x = 0
        for i, x in enumerate(group_x_keys):
            grid_min_x_temp = grid_min_x + len(group_x[x])
            grid_temp = grid[:-4, grid_min_x:grid_min_x_temp]
            if i == 0:
                ax_t = fig.add_subplot(
                    grid_temp,
                    xticklabels=[],
                    yticklabels=[],
                )
            else:
                ax_t = fig.add_subplot(
                    grid_temp,
                    sharex=ax_list_temp[0],
                    yticklabels=[],
                )
            grid_min_x = grid_min_x_temp
            ax_list[-1].append(ax_t)
        color_ax = fig.add_subplot(grid[-2:, g:-g], xticklabels=[], yticklabels=[])
        for x in range(len(group_x_keys)):
            ax_single(y_i, x, y_i+1, len(group_x_keys))
        ax_single_corlor()
        # plt.show()
        plt.savefig(os.path.join(
            save_dir,
            "{}_{}{}.pdf".format(
                os.path.basename(motif_dir),
                y,
                ".remove_cluster" if len(remove_cluster) > 0 else ""
            )
        ), dpi=500, bbox_inches='tight')
        plt.close()


def find_motif(bed_dir: str, summit_file: str, tmp_dir: str, species: str, remove_cluster: list, remove_motifs: set, sample_time: list, samples_pat_samples: dict, logger: logging.Logger, extend_num: int = 25):
    result_dir = os.path.join(
        os.path.dirname(bed_dir),
        "{}_motif".format(os.path.basename(bed_dir))
    )

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    def run_motif(bed_file, summit_file, species, result_dir, extend_num, tmp_dir):
        bed_file_tmp = os.path.join(tmp_dir, os.path.basename(bed_file))
        result_dir_now = os.path.join(result_dir, os.path.basename(bed_file)[:-4])
        if not os.path.exists(result_dir_now):
            os.makedirs(result_dir_now)

        kr = os.path.join(result_dir_now, "knownResults.txt")
        if os.path.exists(kr):
            return

        bed_file_summit = os.path.join(result_dir_now, os.path.basename(bed_file))
        os.system("intersectBed -wa -a {} -b {} | sort | uniq > {}".format(
            summit_file,
            bed_file,
            bed_file_tmp
        ))
        bed_file_summit = extend_bed(
            bed_file=bed_file_tmp,
            bed_extend=extend_num,
            result_dir=result_dir_now,
            tmp_dir=tmp_dir
        )
        os.system("cat -n "+ bed_file_summit +"""| awk -F '\t' '{print $1"\t"$2"\t"$3"\t"$4"\t+"}' > """ + bed_file_tmp)
        os.system("findMotifsGenome.pl {} {} {} -size given -p 8 -mask -preparsedDir ~/homer_data > {} 2>&1".format(
            bed_file_tmp,
            species,
            result_dir_now,
            os.path.join(result_dir_now, "run.log"),
        ))
        os.system("rm -rf {}".format(bed_file_tmp))
        # logger.info(os.path.basename(file_single)[:-4])

    
    # 计算信号值
    error_file = os.path.join(result_dir, "error.log")
    def task_done(f: fu.Future):
        nonlocal error_file
        # 线程异常处理
        e = f.exception()
        if e:
            with open(error_file, "a") as f:
                f.write(traceback.format_exc())
                f.write("\n")
    
    task_all = list()
    with fu.ThreadPoolExecutor(max_workers=4) as executor:
        for bed_file in os.listdir(bed_dir):
            bed_file = os.path.join(bed_dir, bed_file)
            task_temp = executor.submit(
                run_motif,
                bed_file=bed_file,
                summit_file=summit_file,
                species=species,
                result_dir=result_dir,
                extend_num=extend_num,
                tmp_dir=tmp_dir
            )
            # 任务回调
            task_temp.add_done_callback(task_done)
            task_all.append(task_temp)
        fu.wait(task_all)

    try:
        plot_motif(
            motif_dir=result_dir,
            remove_motifs=remove_motifs,
            logger=logger
        )
        if len(remove_cluster) > 0:
            plot_motif(
                motif_dir=result_dir,
                remove_cluster=remove_cluster,
                remove_motifs=remove_motifs,
                logger=logger
            )
    except:
        logger.error("Plot motif error: {}".format(traceback.format_exc()))
        

def main(file_meta: str, bed_file_dir: list[str], bed_file_ext: str, summit_file_ext: str, promoter_file: str, tf_file: str, species: str, bw_dir: list[str], remove_cluster: list, remove_motifs: set, result_dir: str, logger: logging.Logger, important_index: int = 0, min_percent: float = 0.1, max_percent: float = 0.7, remove_same_cluster: bool = True, force: bool = False):
    remove_cluster_null = list()

    tmp_dir = os.path.join(result_dir, "tmp")
    dir_all = [result_dir, tmp_dir]
    for i in dir_all:
        if os.path.exists(i):
            continue
        os.makedirs(i)
    
    # 获取样本信息
    samples_pat_type, samples_pat, samples_pat_samples, samples_need, samples_pat_last, sample_time, samples_head_tail, samples_group, important_group = get_samples(file_meta, important_index)
    sample_time_all = samples_head_tail.get("head", list()) + sample_time + samples_head_tail.get("tail", list())

    # 文件夹
    signal_dir = os.path.join(result_dir, "signal")
    if not os.path.exists(signal_dir):
        os.makedirs(signal_dir)
    bed_dir_merge = os.path.join(result_dir, "beds/merge")
    if not os.path.exists(bed_dir_merge):
        os.makedirs(bed_dir_merge)
    bed_dir_sample = os.path.join(result_dir, "beds/sample")
    if not os.path.exists(bed_dir_sample):
        os.makedirs(bed_dir_sample)

    def get_ext_files(dir_path: list[str], ext: str):
        nonlocal samples_need
        res = dict()
        end_str = "{}.bed".format(ext)
        for i in dir_path:
            for root, _, files in os.walk(i):
                for file_single in files:
                    if not file_single.endswith(end_str):
                        continue
                    file_name = file_single[:-len(end_str)]
                    if file_name not in samples_need:
                        continue
                    assert file_name not in res, "{} {} duplicate: {}, {}".format(file_name, end_str, res[file_name], os.path.join(i, root, file_single))
                    res[file_name] = os.path.join(i, root, file_single)
        
        sample_gap = set(samples_need) - set(res.keys())
        assert len(sample_gap) == 0, "sample missing: {}".format(sample_gap)
        return list(res.values())

    logger.info("bed dir: {}", bed_file_dir)
    bed_file_ori = get_ext_files(
        dir_path=bed_file_dir,
        ext=bed_file_ext
    )
    logger.info("bed file ori: {}".format(bed_file_ori))

    # 将bed扩大
    logger.info("extend bed ...")
    assert len(bed_file_ori) > 0, "No bed files !"
    bed_file = list()
    for bed_file_ori_temp in bed_file_ori:
        bed_file.append(extend_bed(
            bed_file=bed_file_ori_temp,
            bed_extend=0,
            result_dir=bed_dir_sample,
            tmp_dir=tmp_dir,
            force=force
        ))
    
    # 将bed合并
    logger.info("combine bed ...")
    signal_num_file_ori = bed_combine(
        bed_file=bed_file,
        ext_len=len(bed_file_ext) + 4,
        result_dir=bed_dir_merge,
        tmp_dir=tmp_dir,
        force=force
    )

    
    summit_file_ori = get_ext_files(
        dir_path=bed_file_dir,
        ext=summit_file_ext
    )

    # 将bed合并
    logger.info("combine summit bed ...")
    # print(summit_file_ori)
    assert len(summit_file_ori) > 0, "No summit files !"
    summit_file = bed_combine(
        bed_file=summit_file_ori,
        ext_len=len(summit_file_ext) + 4,
        result_dir=bed_dir_merge,
        tmp_dir=tmp_dir,
        file_name_tag="summit.",
        need_head=False,
        force=force
    )

    
    # 将数据分类
    logger.info("cluster ...")
    signal_file, signal_file_ori = cluster_samples(
        signal_file=signal_num_file_ori,
        samples_pat_type=samples_pat_type,
        samples_pat=samples_pat,
        samples_pat_samples=samples_pat_samples,
        samples_need=samples_need,
        samples_pat_last=samples_pat_last,
        bw_dir=bw_dir,
        result_dir=bed_dir_merge,
        signal_dir=signal_dir,
        tmp_dir=tmp_dir,
        force=force
    )

    # 画图
    logger.info("ploting ...")
    image_dir = os.path.join(result_dir, "img")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for sample_group_now in samples_pat_samples:
        draw_image(
            signal_file=signal_file,
            sample_group_now=sample_group_now,
            min_percent=min_percent,
            max_percent=max_percent,
            samples_pat_samples=samples_pat_samples,
            result_dir=image_dir,
            remove_cluster=remove_cluster_null,
            sample_time=sample_time_all,
            force=force
        )
        if len(remove_cluster) > 0:
            draw_image(
                signal_file=signal_file,
                sample_group_now=sample_group_now,
                min_percent=min_percent,
                max_percent=max_percent,
                samples_pat_samples=samples_pat_samples,
                result_dir=image_dir,
                remove_cluster=remove_cluster,
                sample_time=sample_time_all,
                force=force
            )
    
    # 取每一类bed出来
    logger.info("get genes ...")
    dir_bed, dir_gene, dir_bed_fail, dir_gene_fail = get_genes_from_bed(
        bed_all_file=signal_file,
        samples_pat_samples=samples_pat_samples,
        promoter_file=promoter_file,
        tf_file=tf_file,
        important_group=important_group,
        result_dir=result_dir,
        logger=logger
    )

    # cluster peak num
    logger.info("peak num ...")
    plot_nums_of_bed(
        cluster_bed_dir=dir_bed,
        result_dir=image_dir,
        remove_cluster=remove_cluster_null,
        force=force
    )
    if len(remove_cluster) > 0:
        logger.info("peak num remove ...")
        plot_nums_of_bed(
            cluster_bed_dir=dir_bed,
            result_dir=image_dir,
            remove_cluster=remove_cluster,
            force=force
        )

    
    # align one
    if len(samples_pat_samples) > 1:
        logger.info("align heatmap ...")
        draw_image_by_one(
            signal_file=signal_file,
            important_group=important_group,
            min_percent=min_percent,
            max_percent=max_percent,
            samples_pat_samples=samples_pat_samples,
            samples_group=samples_group,
            result_dir=image_dir,
            remove_cluster=remove_cluster_null,
            remove_same_cluster=remove_same_cluster,
            sample_time=sample_time,
            samples_head_tail=samples_head_tail,
            force=force
        )
    
        logger.info("align sankey ...")
        draw_image_by_one_sankey(
            signal_file=signal_file,
            important_group=important_group,
            samples_pat_samples=samples_pat_samples,
            result_dir=image_dir,
            remove_cluster=remove_cluster_null,
            sample_time=sample_time,
            samples_head_tail=samples_head_tail,
            force=force
        )
        if len(remove_cluster) > 0:
            logger.info("align heatmap remove ...")
            draw_image_by_one(
                signal_file=signal_file,
                important_group=important_group,
                min_percent=min_percent,
                max_percent=max_percent,
                samples_pat_samples=samples_pat_samples,
                samples_group=samples_group,
                result_dir=image_dir,
                remove_cluster=remove_cluster,
                remove_same_cluster=remove_same_cluster,
                sample_time=sample_time,
                samples_head_tail=samples_head_tail,
                force=force
            )
        
            logger.info("align sankey remove ...")
            draw_image_by_one_sankey(
                signal_file=signal_file,
                important_group=important_group,
                samples_pat_samples=samples_pat_samples,
                result_dir=image_dir,
                remove_cluster=remove_cluster,
                sample_time=sample_time,
                samples_head_tail=samples_head_tail,
                force=force
            )
    
    # 富集基因
    logger.info("enrich ...")
    enrich_genes(
        gene_dir=dir_gene,
        remove_cluster=remove_cluster,
        species=species,
        force=force,
        logger=logger,
    )
    
    if len(samples_pat_samples) > 1:
        logger.info("enrich fail ...")
        enrich_genes(
            gene_dir=dir_gene_fail,
            remove_cluster=remove_cluster,
            species=species,
            force=force
        )


    # motif
    logger.info("motif ...")
    find_motif(
        bed_dir=dir_bed,
        summit_file=summit_file,
        tmp_dir=tmp_dir,
        species=species,
        remove_cluster=remove_cluster,
        remove_motifs=remove_motifs,
        sample_time=sample_time,
        samples_pat_samples=samples_pat_samples,
        logger=logger
    )
    
    if len(samples_pat_samples) > 1:
        logger.info("motif fail ...")
        find_motif(
            bed_dir=dir_bed_fail,
            summit_file=summit_file,
            tmp_dir=tmp_dir,
            species=species,
            remove_cluster=remove_cluster,
            remove_motifs=remove_motifs,
            sample_time=sample_time,
            samples_pat_samples=samples_pat_samples,
            logger=logger
        )

    os.system(" ".join(["rm", "-rf", tmp_dir]))
    logger.info("done")

if __name__ == "__main__":
    main(
        file_meta="/mnt/liudong/task/20230203_HT_NanogOct4_RNAseq/src/Nanog.heatmap.meta.20230625.xlsx",
        bed_file_dir=["/mnt/liudong/task/20221103_huangtao_rowdata/result/0.mapping/peak"],
        bed_file_ext=".e5_peaks",
        summit_file_ext=".e5_summits",
        promoter_file="/mnt/liudong/data/Genome/mm10/mm10.promoter.ncbiRefSeq.WithUCSC.bed",
        tf_file="/mnt/zhaochengchen/Data/mm10/Mus_musculus_TF.txt",
        species="mm10",
        bw_dir=["/mnt/liudong/task/20230203_HT_NanogOct4_RNAseq/result/bw"],
        remove_cluster=["CO9", "OC1"],
        remove_motifs=set(),
        result_dir="/mnt/liudong/task/20240116_HT/result.sort/fig3/3e.3f.3g",
        logger=logging.getLogger(),
        important_index=1,
    )