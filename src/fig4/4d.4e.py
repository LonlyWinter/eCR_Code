# -*- coding: UTF-8 -*-
# cython: language_level=3

'''
@File    :   setup.py
@Modif   :   2024/01/25 17:01:21
@Author  :   winter
@Version :   0.1
@Contact :   winter_lonely@126.com
@License :   (C)Copyright winter
@Desc    :   
'''

from matplotlib.font_manager import fontManager
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats
from itertools import combinations
import matplotlib.ticker as mpt
import matplotlib.pyplot as plt
import logging
import time
import json
import re
import os

mpl.use("Agg")
fontManager.addfont(os.path.join(os.path.dirname(__file__), "..", "arial.ttf"))
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]

class TaskSrc:
    def init(self) -> None:
        self.label = "03.Analysis"

    
    def get_meta_data(self, file_meta: str) -> tuple[dict, tuple, tuple]:
        df = pd.read_excel(file_meta)
        cols_need = ["tag", "sample_ctrl", "sample_treat"]
        cols_gap = set(cols_need) - set(df.columns)
        assert len(cols_gap) == 0, "file meta columns missing: {}".format(cols_need)
        df = df[cols_need]
        df.dropna(inplace=True)
        
        assert len(df["tag"].unique()) == len(df), "tag duplicate!"
        
        sample_all = tuple(set(df["sample_treat"].to_list() + df["sample_ctrl"].to_list()))
        tag_all = tuple(df["tag"].to_list())

        sample_dict = dict()
        for _, se_temp in df.iterrows():
            sample_dict[se_temp["tag"]] = (se_temp["sample_treat"], se_temp["sample_ctrl"])

        return (sample_dict, sample_all, tag_all)


    def get_ext_files(self, dir_path: list[str], ext: str, suffix: str, samples_need: tuple, err_msg: str) -> dict:
        res = dict()
        end_str = "{}{}".format(ext, suffix)
        for i in dir_path:
            for root, _, files in os.walk(i):
                for file_single in files:
                    if not file_single.endswith(end_str):
                        continue
                    file_name = file_single[:-len(end_str)]
                    if file_name not in samples_need:
                        continue
                    assert file_name not in res, "{} {} duplicate".format(file_name, end_str)
                    res[file_name] = os.path.join(i, root, file_single)
        
        sample_gap = set(samples_need) - set(res.keys())
        assert len(sample_gap) == 0, "{}: {}".format(err_msg, sample_gap)
        return res
    

    def plot_single(self, tag_name: str, tag_need: tuple, meta_data: dict, file_signal: str, dir_result: str, dir_gene: str, logger: logging.Logger):
        region_name = os.path.basename(file_signal)[:-4]
        file_plot = os.path.join(dir_result, "{}_{}.pdf".format(tag_name, region_name))

        if os.path.exists(file_plot):
            logger.warn("skip {} {}".format(tag_name, region_name))

        df = pd.read_table(file_signal)
        df.columns = list(map(
            lambda d: d.strip("#").strip("'"),
            df.columns
        ))
        
        
        names = list()

        for i in tag_need:
            i0, i1 = tuple(map(
                lambda d: "{}.bw".format(d),
                meta_data[i]
            ))
            ix = "{0} Change({1} - {2})".format(
                i,
                *meta_data[i]
            )
            df[ix] = df[i0] - df[i1]
        #    df[ix] = np.log2(df[i0] / df[i0])
            names.append(ix)
        xx, yy = names
        df_now = df[["chr", "start", "end"] + names]#.iloc[:10000]
        
        assert len(df_now) > 0, "No data"
        
        xy_lim = max(int(max(df_now[xx].quantile(0.999), df_now[yy].quantile(0.999)) * 10) // 2 / 5, 0.1)
    
        xy_num = len(df_now)
        
        logger.info("{}, {}, xy_lim: {}, xy_num: {}".format(tag_name, region_name, xy_lim, xy_num))

        xy_cutoff = 0
        xy_cutoff_reverse = -xy_cutoff

        def calc_grid(se: pd.Series):
            nonlocal xy_cutoff, xy_cutoff_reverse, xx, yy
            xi, yi = se[xx], se[yy]
            if (xi > xy_cutoff) and (yi > xy_cutoff):
                return "right_upper"
            elif (xi < xy_cutoff_reverse) and (yi > xy_cutoff):
                return "left_upper"
            elif (xi < xy_cutoff_reverse) and (yi < xy_cutoff_reverse):
                return "left_lower"
            elif (xi > xy_cutoff) and (yi < xy_cutoff_reverse):
                return "right_lower"
            else:
                return "xy_axis"

        df_now["grid"] = df_now.apply(calc_grid, axis=1)
        df_now_grouped = df_now.groupby("grid")
        xy_percent = list(map(
            lambda d: len(df_now_grouped.get_group(d)) if d in set(df_now["grid"].unique()) else 0,
            ("right_upper", "left_upper", "left_lower", "right_lower", "xy_axis")
        ))

        for g, df_temp in df_now_grouped:
            file_bed_grid_temp = os.path.join(dir_gene, "{}_{}.{}.bed".format(tag_name, region_name, g))
            df_temp[["chr", "start", "end", xx, yy]].to_csv(file_bed_grid_temp, sep="\t", index=False, header=False)

        xy_percent = tuple(map(lambda d: d / xy_num * 100, xy_percent))

        df_now = df_now[(df_now[xx].abs() < xy_lim*1.1) & (df_now[yy].abs() < xy_lim*1.1)]
        x = df_now[xx].values
        y = df_now[yy].values
        cor, pval = stats.spearmanr(x, y)
        xy = np.vstack([x, y])  #  将两个维度的数据叠加
        z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_subplot()
        ax.scatter(
            x,
            y,
            c=z,
            s=10,
            cmap='Blues',
            rasterized=True
        )
        ax.axvline(
            x=0,
            c="black",
            linewidth=1,
            linestyle="--",
            dashes=(5, 5)
        )
        ax.axhline(
            y=0,
            c="black",
            linewidth=1,
            linestyle="--",
            dashes=(5, 5)
        )
        xy_unit = xy_lim / 10
        ax.text(
            -xy_lim + xy_unit,
            xy_lim - xy_unit*2,
            "Cor={:.2f}".format(
                cor,
            ),
            va="bottom",
            ha="left",
        )
        for i, (xi, yi) in zip(
            range(4),
            (
                (
                    xy_lim - xy_unit*3,
                    xy_unit
                ),
                (
                    -xy_lim + xy_unit*3,
                    xy_unit
                ),
                (
                    -xy_lim + xy_unit*3,
                    -xy_unit
                ),
                (
                    xy_lim - xy_unit*3,
                    -xy_unit
                ),
            ),
        ):
            ax.text(
                xi,
                yi,
                "{:.1f}%".format(xy_percent[i]),
                va="center",
                ha="center",
            )
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)
        xy_locator = int(xy_lim * 10) // 2 / 10 + 0.1
        ax.xaxis.set_major_locator(mpt.MultipleLocator(xy_locator))
        ax.yaxis.set_major_locator(mpt.MultipleLocator(xy_locator))
        ax.set_title("{} {}".format(tag_name, region_name))
        ax.set_xlabel(xx)
        ax.set_ylabel(yy)
        plt.savefig(
            file_plot,
            dpi=5000,
            bbox_inches="tight"
        )
        plt.close()



    def calc_change(self, tag_name: str, dir_result: str, tag_need: tuple, meta_data: dict, file_bws: dict, file_beds: dict, dir_common_signal: str, logger: logging.Logger):

        dir_gene = os.path.join(dir_result, "Gene")
        dir_signal = os.path.join(dir_result, "Signals")
        dir_plot = os.path.join(dir_result, "Plot")
        os.makedirs(dir_gene, exist_ok=True)
        os.makedirs(dir_signal, exist_ok=True)
        os.makedirs(dir_plot, exist_ok=True)

        file_beds_need = tuple([*tuple(map(
            lambda d: file_beds[d],
            meta_data[tag_need[0]]
        )), *tuple(map(
            lambda d: file_beds[d],
            meta_data[tag_need[1]]
        ))])
        file_bws_need = tuple([*tuple(map(
            lambda d: file_bws[d],
            meta_data[tag_need[0]]
        )), *tuple(map(
            lambda d: file_bws[d],
            meta_data[tag_need[1]]
        ))])
        
        # peak
        file_peak_merge = os.path.join(dir_result, "Peak.bed")
        if not os.path.exists(file_peak_merge):
            self.exec_cmd("cat {} | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
                " ".join(file_beds_need),
                file_peak_merge
            ), logger)
        file_signal = os.path.join(dir_signal, "Peak.txt")
        if not os.path.exists(file_signal):
            cmd = "multiBigwigSummary BED-file -b {} --outRawCounts {} -o {} -p 8 --BED {}".format(
                " ".join(file_bws_need),
                file_signal,
                file_signal.replace(".txt", ".npz"),
                file_peak_merge
            )
            self.exec_cmd(cmd, logger)
        for file_single in os.listdir(dir_common_signal):
            if os.path.exists(os.path.join(dir_signal, file_single)):
                continue
            self.exec_cmd("ln -s {} {}/".format(os.path.join(dir_common_signal, file_single), dir_signal), logger)
        

        file_signal_all = list(filter(
            lambda d: d.endswith(".txt"),
            os.listdir(dir_signal)
        ))
        file_signal_all.sort(key=lambda d: os.path.getsize(os.path.join(dir_signal, d)))
        for file_single in file_signal_all:
            self.plot_single(
                tag_name=tag_name,
                tag_need=tag_need,
                meta_data=meta_data,
                file_signal=os.path.join(dir_signal, file_single),
                dir_result=dir_plot,
                dir_gene=dir_gene,
                logger=logger
            )
        return dir_gene

    def calc_bed_bin(self, bin_size: int, chr_len: str, file_res: str):
        if os.path.exists(file_res):
            return
        data = list()
        with open(chr_len) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t",1)
                if len(line[0]) > 5:
                    continue
                chr_temp = line[0]
                chr_len_max = int(line[1])
                chr_posi_now = 1
                while chr_posi_now < chr_len_max:
                    chr_posi_next = min(chr_posi_now + bin_size, chr_len_max)
                    data.append("{}\t{}\t{}".format(
                        chr_temp,
                        chr_posi_now,
                        chr_posi_next
                    ))
                    chr_posi_now = chr_posi_next
        with open(file_res, "w") as f:
            f.write("\n".join(data))


    def main(
            self,
            species: str,
            bin_background: str,
            peak_file_ext: str,
            **kwargs
        ):
        logger: logging.Logger = kwargs['logger']

        dir_base = os.getcwd()
        data_species = get_pipline_data(species)

        meta_data, sample_all, tag_all = self.get_meta_data(kwargs['return'][0])
        file_bws = self.get_ext_files(
            dir_path=kwargs['return'][1].strip(",").split(","),
            ext="",
            suffix=".bw",
            samples_need=sample_all,
            err_msg="Bw file missing"
        )
        file_beds = self.get_ext_files(
            dir_path=kwargs['return'][2].strip(",").split(","),
            ext=peak_file_ext,
            suffix=".bed",
            samples_need=sample_all,
            err_msg="Bed file missing"
        )
        file_regions = dict(map(
            lambda dd: ("GivenRegin{}.{}".format(dd[0], os.path.basename(dd[1])[:-4]), dd[1]),
            enumerate(tuple(filter(
                lambda d: os.path.exists(d),
                kwargs['return'][3].strip(",").split(",")
            )), start=1)
        ))
        bin_background_list = tuple(map(
            lambda d: int(d.strip()),
            bin_background.strip(",").split(",")
        ))
        dir_bed_bins = os.path.join(dir_base, "Bed_bins")
        os.makedirs(dir_bed_bins, exist_ok=True)
        for bin_size in bin_background_list:
            file_bin_temp = os.path.join(dir_bed_bins, "Bin{}k.bed".format(bin_size))
            file_regions["Bin{}k".format(bin_size)] = file_bin_temp
            self.calc_bed_bin(
                bin_size=bin_size * 1000,
                chr_len=data_species["chromInfo"],
                file_res=file_bin_temp
            )



        dir_common_signal = os.path.join(dir_base, "Common_Signals")
        os.makedirs(dir_common_signal, exist_ok=True)
        
        # 计算signal
        func_list = list()
        arg_list = list()
        for k in file_regions.keys():
            file_signal = os.path.join(dir_common_signal, "{}.txt".format(k))
            if os.path.exists(file_signal):
                continue
            cmd = "multiBigwigSummary BED-file -b {} --outRawCounts {} -o {} -p 8 --BED {}".format(
                " ".join(file_bws.values()),
                file_signal,
                file_signal.replace(".txt", ".npz"),
                file_regions[k]
            )
            func_list.append(self.exec_cmd)
            arg_list.append(dict(cmd=cmd, logger=logger))
        cdesk_task_num = max(int(os.environ.get("cdesk_task_num", 8)) // 8, 2)
        self.multi_task(func_list=func_list, arg_list=arg_list, logger=logger, task_num=cdesk_task_num)

        
        for x, y in combinations(tag_all, 2):
            k = "{}_{}".format(x, y)
            dir_gene_temp = self.calc_change(
                tag_name=k,
                dir_result=os.path.join(dir_base, k),
                tag_need=(x, y),
                meta_data=meta_data,
                file_beds=file_beds,
                file_bws=file_bws,
                dir_common_signal=dir_common_signal,
                logger=logger
            )
            # 对到region上面
            for file_single in os.listdir(dir_gene_temp):
                if not file_single.endswith(".bed"):
                    continue
                file_single_temp = os.path.join(dir_gene_temp, file_single)
                self.exec_cmd(" ".join([
                    os.path.join(os.path.dirname(__file__), "..", "..", "CDesk_cli"),
                    "GetRegionGenes",
                    dir_gene_temp,
                    "species={}".format(species),
                    "file_beds={}".format(file_single_temp),
                    "> {} 2>&1".format(file_single_temp.replace(".bed", ".log"))
                ]), logger)


TaskSrc.__doc__ = """ ChangeInRegions

计算不同类别之间变化的相关性：散点的密度图

### 一、输入参数说明

### 二、输出文件说明

### 三、任务作者

Dong LIU
"""