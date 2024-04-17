#!/usr/bin/env python
# cython: language_level=3
# -*- coding: UTF-8 -*-
'''
@File    :   SpecificRegion.py
@Modif   :   2023/07/17 13:47:57
@Author  :   winter
@Version :   0.1
@Contact :   winter_lonely@126.com
@License :   (C)Copyright winter
@Desc    :   
'''

import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import product
from pyg2plot import Plot, JS
import matplotlib.ticker as mpt
import matplotlib.cm as mpm
from matplotlib.patches import PathPatch
from matplotlib.font_manager import fontManager
import matplotlib.patches as patches
import concurrent.futures as fu
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
import logging
import random
import re
import os

mpl.use("Agg")
fontManager.addfont(os.path.join(os.path.dirname(__file__), "..", "arial.ttf"))
mpl.rc('pdf', fonttype=42)
mpl.rcParams['font.sans-serif'] = ["Arial"]

class TaskSrc:
    def init(self) -> None:
        self.label = "03.Analysis"


    def handle_name(self, names: tuple[str], name_align: str, logger: logging.Logger):
        pat = re.compile("\.|-|_")
        names_now = tuple(map(
            lambda d: pat.split(d),
            names
        ))
        name_align_now = pat.split(name_align)
        num_name = len(names)
        # 从头开始找
        index_head = 0
        while True:
            is_same = num_name > 1
            for i in range(1, num_name):
                if names_now[i][index_head] == names_now[i-1][index_head]:
                    continue
                is_same = False
                if not is_same:
                    break
            if not is_same:
                break
            index_head += 1
        logging.info("names now: {}, index head: {}".format(names_now, index_head))
        # 从尾巴开始找
        index_tail = -1
        while True:
            is_same = num_name > 1
            for i in range(1, num_name):
                if names_now[i][index_tail] == names_now[i-1][index_tail]:
                    continue
                is_same = False
                if not is_same:
                    break
            if not is_same:
                break
            index_tail -= 1
        logging.info("names now: {}, index head: {}, index tail: {}".format(names_now, index_head, index_tail))
        # 结果
        head_common = ".".join(names_now[0][:index_head])
        tail_common = ".".join(names_now[0][index_tail+1:]) if index_tail < -1 else ""
        logging.info("names now: {}, index head: {}, head common: {}, index tail: {}, tail common: {}".format(
            names_now,
            index_head,
            head_common,
            index_tail,
            tail_common
        ))
        names_now = tuple(map(
            lambda d: ".".join(d[index_head:index_tail+1]) if index_tail < -1 else ".".join(d[index_head:]),
            names_now
        ))
        try:
            name_align = ".".join(name_align_now[index_head:index_tail+1]) if index_tail < -1 else ".".join(name_align_now[index_head:]),
            if name_align not in names_now:
                name_align = "no"
        except:
            name_align = "no"
        return ".".join((
            head_common,
            tail_common
        )).strip("."), names_now, name_align


    def region_num(self, file_path: str):
        num = 0
        with open(file_path) as f:
            for line in f:
                if line.strip() == '':
                    continue
                num += 1
        return num


    def venn2(self, area1_ori: int, area2_ori: int, area_cross_ori: int, labels: tuple, title: str, margin: float = 0.5):

        w_ori = 3
        h_ori = 2
        colors = ["#16557A", "#C7A609"]
        alpha = [0.5, 0.5]

        assert len(labels) == 2, "labels need 2 length !"
        label_colors = ["black", "black"]
        label_sizes = [11, 11]

        nums = [area1_ori, area_cross_ori, area2_ori]
        num_colors = ["black", "black", "black"]
        num_sizes = [11, 11, 11]


        # 去除margin的区域画图
        w = w_ori - (margin * 2)
        h = h_ori - (margin * 2)

        # area1 = np.log10(area1 + 1)
        # area2 = np.log10(area2 + 1)
        # area_cross = np.log10(area_cross + 1) ** 8
        area_cross = float(area_cross_ori) * 1.8
        area1 = float(area1_ori)
        area2 = float(area2_ori)
        # 直径
        w1 = (area1 + area_cross) / (area1 + area2 + area_cross) / 2 * w
        w2 = (area2 + area_cross) / (area1 + area2 + area_cross) / 2 * w
        wc = area_cross / (area1 + area2 + area_cross) * w

        w1, w2, wc = round(w1, 2), round(w2, 2), round(wc, 2)

        fig = plt.figure(figsize=(w_ori, h_ori))
        ax = fig.add_axes([0, 0, 1, 1])

        ax.xaxis.set_major_formatter(mpt.NullFormatter())
        ax.yaxis.set_major_formatter(mpt.NullFormatter())
        ax.yaxis.set_tick_params(left=False)
        ax.xaxis.set_tick_params(bottom=False)
        [ax.spines[i].set_visible(False) for i in ("top", "bottom", "left", "right")]

        ax.set_xlim((0, w_ori))
        ax.set_ylim((0, h_ori))
        
        h_center = h / 2 + margin

        # 画圆
        for x, r, i in zip(
            [
                w1 + margin,
                w1 * 2 - wc + w2 + margin
            ],
            [
                w1,
                w2
            ],
            range(2)
        ):
            c = patches.Circle(
                (x, h_center),
                r,
                facecolor=colors[i],
                alpha=alpha[i],
            )
            ax.add_patch(c)

        # 画圆内的数字
        for x, i in zip(
            [
                (w1 * 2 - wc) / 2 + margin,
                w1 * 2 + margin - (wc / 2),
                (w2 * 2 - wc) / 2 + (w1 * 2) + margin
            ],
            range(3)
        ):
            ax.text(
                x=x,
                y=h_center,
                s=str(nums[i]),
                fontdict={
                    "size": num_sizes[i],
                    "color": num_colors[i],
                },
                ha="center",
                va="center"
            )
        
        # 画圆外的标签
        for x, y, i in zip(
            [
                w1 / 5 + margin,
                w1 * 2 + w2 * 2 - wc - w2 / 5 + margin
            ],
            [
                w1 * 2 + margin,
                w1 * 2 + margin,
            ],
            range(2)
        ):
            ax.text(
                x=x,
                y=y,
                s=str(labels[i]),
                fontdict={
                    "size": label_sizes[i],
                    "color": label_colors[i],
                },
                ha="right" if i == 0 else "left",
                va="top"
            )
        jac = nums[1] / sum(nums)
        ax.set_title("{}\nJaccard={:.2f}".format(
            title,
            jac
        ))

        # plt.show()
        plt.savefig(
            "Venn.{}.pdf".format(title),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()


    def bed_change(self, file_beds: tuple[str], dir_result: str, logger: logging.Logger, palette: tuple, name_beds: tuple, head_tail_common: str, name_align: str = "no", title: str = ""):
        os.makedirs(dir_result, exist_ok=True)

        num_beds = len(name_beds)

        title = "{}\n{}".format(
            ".".join((
                title,
                "_".join(name_beds),
            )).strip("."),
            head_tail_common
        ).strip(".").replace("..", ".")

        file_bed_all = os.path.join(dir_result, "{}.{}.All.tmp.bed".format(
            "_".join(name_beds),
            head_tail_common
        ).strip(".").replace("..", "."))
        file_bed_all_num = os.path.join(dir_result, "{}.{}.All.txt".format(
            "_".join(name_beds),
            head_tail_common
        ).strip(".").replace("..", "."))
        file_bed_all_num_temp = os.path.join(dir_result, "{}.{}.All.num.tmp.bed".format(
            "_".join(name_beds),
            head_tail_common
        ).strip(".").replace("..", "."))
        self.exec_cmd("cat {} | cut -f 1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(
            " ".join(file_beds),
            file_bed_all
        ), logger)
        cmd = ""
        for i in range(num_beds):
            cmd = "{}| intersectBed -c -a - -b {}".format(
                cmd,
                file_beds[i],
            )
        cmd = "cat {} {} > {}".format(
            file_bed_all,
            cmd,
            file_bed_all_num_temp
        )
        self.exec_cmd(cmd, logger)
        with open(file_bed_all_num, "w", encoding="UTF-8") as f:
            f.write("\t".join(["chr", "start", "end"] + list(name_beds)))
            f.write("\n")
        self.exec_cmd("cat {} >> {}".format(file_bed_all_num_temp, file_bed_all_num), logger)
        self.exec_cmd("rm -rf {}".format(file_bed_all_num_temp), logger)
        self.exec_cmd("rm -rf {}".format(file_bed_all), logger)
        
        df = pd.read_table(file_bed_all_num)
        df[["chr", "start", "end"]].to_csv(
            file_bed_all_num.replace(".txt", ".bed"),
            index=False,
            header=False,
            sep="\t"
        )

        width = 0.6

        fig = plt.figure(figsize=(
            len(name_beds) * 1.3,
            9 if name_align in name_beds else 5
        ))
        ax = plt.subplot()
        if name_align in name_beds:
            data_dict = dict()
            data_dict[name_align] = dict()
            data_dict[name_align]["Gain"] = 0
            data_dict[name_align]["Loss"] = 0
            data_dict[name_align]["Reserve"] = len(df[df[name_align] > 0])
            for s in name_beds:
                if s == name_align:
                    continue
                df_temp = df[(df[s] > 0) & (df[name_align] == 0)]
                df_temp[["chr", "start", "end"]].to_csv(
                    os.path.join(dir_result, "{}.{}.Gain.bed".format(
                        s,
                        head_tail_common
                    ).strip(".").replace("..", ".")),
                    index=False,
                    header=False,
                    sep="\t"
                )
                df_temp_loss = df[(df[s] == 0) & (df[name_align] > 0)]
                df_temp_loss[["chr", "start", "end"]].to_csv(
                    os.path.join(dir_result, "{}.{}.Loss.bed".format(
                        s,
                        head_tail_common
                    ).strip(".").replace("..", ".")),
                    index=False,
                    header=False,
                    sep="\t"
                )
                df_temp_reserve = df[(df[s] > 0) & (df[name_align] > 0)]
                df_temp_reserve[["chr", "start", "end"]].to_csv(
                    os.path.join(dir_result, "{}.{}.Reserve.bed".format(
                        s,
                        head_tail_common
                    ).strip(".").replace("..", ".")),
                    index=False,
                    header=False,
                    sep="\t"
                )
                data_dict[s] = dict()
                data_dict[s]["Gain"] = len(df_temp)
                data_dict[s]["Loss"] = len(df_temp_loss)
                data_dict[s]["Reserve"] = len(df_temp_reserve)
            data = list()
            for i in data_dict:
                for k in data_dict[i]:
                    data.append({
                        "sample": i,
                        "type": k,
                        "num": data_dict[i][k]
                    })
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(dir_result, "gain_loss.txt"), index=False, sep="\t")
            for k in data_dict:
                data_dict[k]["Gain"] = data_dict[k]["Loss"] + data_dict[k]["Reserve"] + data_dict[k]["Gain"]
                data_dict[k]["Reserve"] = data_dict[k]["Loss"] + data_dict[k]["Reserve"]
            data = list()
            for i in data_dict:
                for k in data_dict[i]:
                    data.append({
                        "sample": i,
                        "type": k,
                        "num": data_dict[i][k]
                    })
            df = pd.DataFrame(data)
            palette = list(palette[:num_beds])
            # 画三层
            for t, c in zip(
                ["Gain", "Reserve", "Loss"],
                [palette, "#8B7E75", palette],
            ):
                df_temp = df[df["type"] == t]
                ax.bar(
                    np.arange(len(df_temp)),
                    np.asarray(df_temp['num'].to_list()),
                    width=width,
                    tick_label=df_temp["sample"],
                    color=c if t != "Loss" else "white",
                    label=t,
                    hatch="//" if t == "Loss" else None,
                    edgecolor=c,
                    linewidth=0.1
                )
            # 画两个横线
            n = df[(df["type"] == "Reserve") & (df["sample"] == name_align)]["num"].to_list()[0]
            xs = [-(width / 2 - 0.01), num_beds + (width / 2) - 1.01]
            ax.plot(
                xs,
                [n, n],
                color="black"
            )
            ax.plot(
                xs,
                [0, 0],
                color="black"
            )
            # 画align_one的数量
            plt.text(
                0,
                n // 2,
                s=n,
                va="center",
                ha="center"
            )
            # 画其它类别的数量
            y_max = 0
            name_beds_other = tuple(filter(
                lambda d: d != name_align,
                name_beds
            ))
            num_now = dict()
            for x, s in zip(np.arange(1, num_beds), name_beds_other):
                df_temp_groupby = df.set_index("sample").groupby("type")
                gain_temp = df_temp_groupby.get_group("Gain").loc[s]["num"] - df_temp_groupby.get_group("Reserve").loc[s]["num"]
                reserve_temp = df_temp_groupby.get_group("Reserve").loc[s]["num"] - df_temp_groupby.get_group("Loss").loc[s]["num"]
                loss_temp = df_temp_groupby.get_group("Loss").loc[s]["num"]
                y_max = max(
                    gain_temp + reserve_temp + loss_temp,
                    y_max
                )
                num_now[x] = (
                    (gain_temp // 2 + reserve_temp + loss_temp, gain_temp),
                    (reserve_temp // 2 + loss_temp, reserve_temp),
                    (loss_temp // 2, loss_temp),
                )
            for x in num_now:
                for i in range(3):
                    plt.text(
                        x,
                        num_now[x][i][0],
                        num_now[x][i][1],
                        va="center",
                        ha="center"
                    )
                plt.text(
                    x,
                    - y_max * 0.04,
                    num_now[x][0][1] + num_now[x][2][1],
                    va="center",
                    ha="center"
                )
            # 设置xy轴
            ax.set_ylim(- y_max * 0.08, y_max * 1.05)
            ax.set_yticks([])
            legend_t = ax.legend()
            legend_t_labels = list(map(
                lambda d: d.get_text(),
                legend_t.texts
            ))
            legend_t_handlers = legend_t.legendHandles
            legend_t_handlers[0].set_color(palette[1])
            legend_t_handlers[2].set_edgecolor(palette[1])
            ax.legend(
                legend_t_handlers,
                legend_t_labels,
                frameon=False,
                bbox_to_anchor=(0, 1),
                loc="upper left"
            )
        elif num_beds == 2:
            df["id"] = df.apply(lambda d: "{}:{}_{}".format(d["chr"], d["start"], d["end"]), axis=1)
            df_temp = df.copy()
            for s in name_beds:
                df_temp = df_temp[df_temp[s] > 0]
            df_temp[["chr", "start", "end"]].to_csv(
                os.path.join(dir_result, "{}.{}.Common.bed".format(
                    "_".join(name_beds),
                    head_tail_common
                ).strip(".").replace("..", ".")),
                index=False,
                header=False,
                sep="\t"
            )
            ids_common = set(df_temp["id"].unique())
            num_common = len(ids_common)

            num_list = list()
            for s in name_beds:
                df_temp = df[(df[s] > 0) & (~df["id"].isin(ids_common))]
                df_temp[["chr", "start", "end"]].to_csv(
                    os.path.join(dir_result, "{}.{}.Spec.bed".format(
                        s,
                        head_tail_common
                    ).strip(".").replace("..", ".")),
                    index=False,
                    header=False,
                    sep="\t"
                )
                num_list.append(len(df_temp["id"].unique()))
            
            bottom_num = num_list[0] + num_common
            top_num = bottom_num + num_list[1]
            palette = palette[1:3]
            
            # 画三层柱状图
            ax.bar(
                name_beds,
                [num_list[0], 0],
                bottom=[0, 0],
                width=width,
                color=palette,
                linewidth=0.1
            )
            ax.bar(
                name_beds,
                [num_common, num_common],
                bottom=[num_list[0], num_list[0]],
                width=width,
                color=["#8B7E75", "#8B7E75"],
                linewidth=0.1
            )
            ax.bar(
                name_beds,
                [0, num_list[1]],
                bottom=[bottom_num, bottom_num],
                width=width,
                color=palette,
                linewidth=0.1
            )
            # 画数量
            num_data = list()
            # 0总数
            num_data.append((
                0,
                num_list[0] + num_common + top_num * 0.05,
                num_list[0] + num_common,
            ))
            # 0自己
            num_data.append((
                0,
                num_list[0] / 2,
                num_list[0],
            ))
            # 1总数
            num_data.append((
                1,
                num_list[0] - top_num * 0.05,
                num_list[1] + num_common,
            ))
            # 1自己
            num_data.append((
                1,
                num_list[0] + num_common + num_list[1] / 2,
                num_list[1],
            ))
            for i in num_data:
                plt.text(
                    *i,
                    va="center",
                    ha="center"
                )
            # 公共
            plt.text(
                0.5,
                num_list[0] + num_common / 2,
                num_common,
                # rotation=90,
                va="center",
                ha="center"
            )
            # 画两根线
            n = (
                num_list[0],
                num_list[0] + num_common,
            )
            xs = [-(width / 2 - 0.01), num_beds + (width / 2) - 1.01]
            for i in n:
                ax.plot(
                    xs,
                    [i, i],
                    color="black"
                )

            # 设置xy轴
            ax.set_ylim(- top_num * 0.05, top_num * 1.05)
            ax.set_yticks([])
        else:
            logger.error("No fit plot pat: {}, {}".format(name_align, name_beds))
        ax.set_ylabel("Region Num Change")
        ax.set_title(title)
        ax.set_xticklabels(
            [i.get_text() for i in ax.get_xticklabels()],
            # rotation=30,
            # va="top",
            # ha="right"
        )
        # plt.show()
        plt.savefig(
            os.path.join(dir_result, "..", "Bar.{}.pdf".format(title.replace(" ", "_").replace("\n", ".").strip("."))),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()

        
    def calc_bed_signal(self, bed_file: str, bw_file: str, file_res: str, logger: logging.Logger, tmp_dir: str = None, force: bool = False):
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

        self.exec_cmd("cat -n " + bed_file + """ | awk -F "\t" '{print $2"\t"$3"\t"$4"\t"$1}' > """ + tmp_bed_file, logger)
        
        self.exec_cmd(" ".join(["bigWigAverageOverBed", bw_file, tmp_bed_file, tmp_signal_file]), logger)
        self.exec_cmd("""cat """+tmp_signal_file+""" | sort -k1,1n | awk -F "\t" '{print $5}' > """ + file_res, logger)
        
        with open(file_res) as f:
            res_data = np.asarray(list(map(
                lambda dd: float(dd.strip()),
                f
            )))

        self.exec_cmd(" ".join(["rm", "-rf", tmp_signal_file]), logger)
        self.exec_cmd(" ".join(["rm", "-rf", tmp_bed_file]), logger)
        return res_data



    def calc_signal(self, dir_result: str, file_beds: list, file_bws: list, logger: logging.Logger):
        os.makedirs(dir_result, exist_ok=True)
        
        func_list = list()
        arg_list = list()
        for i in range(len(file_beds)):
            func_list.append(self.calc_bed_signal)
            arg_list.append(dict(
                bed_file=file_beds[i],
                bw_file=file_bws[i],
                file_res=os.path.join(dir_result, "{}.{}.signal.txt".format(
                    os.path.basename(file_beds[i])[:-4],
                    os.path.basename(file_bws[i])[:-3],
                )),
                logger=logger
            ))
        self.multi_task(func_list, arg_list, logger, 8)

        data_signal = list()
        for i in range(len(file_beds)):
            # 提交任务
            data_temp = self.calc_bed_signal(
                bed_file=file_beds[i],
                bw_file=file_bws[i],
                file_res=os.path.join(dir_result, "{}.{}.signal.txt".format(
                    os.path.basename(file_beds[i])[:-4],
                    os.path.basename(file_bws[i])[:-3],
                )),
                logger=logger
            )
            data_signal.append(data_temp)
        return data_signal

    
    def extend_bed_from_center(self, file_bed: str, bp: int, logger: logging.Logger, dir_result: str = None):
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
            self.exec_cmd("cat {0} | cut -f 1-3 | awk '{{a=int(($3-$2)/2)+$2+1;if(a>={1}){{print($1\"\t\"(a-{1})\"\t\"(a+{1}))}}else{{print($1\"\t0\t\"(a+{1}))}}}}' | sort -k1,1 -k2,2n | bedtools merge -i - > {2}".format(
                file_bed,
                bp,
                file_bed_res,
            ), logger)
        return file_bed_res


    def split_bed(self, file_bed: str, count: int, logger: logging.Logger, dir_result: str = None):
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


    def plot_heatmap(self, file_bed_dict: dict, file_dict_num: dict, file_bws: dict, file_summits: dict, dir_result: str, logger: logging.Logger, extend_bp: int, split_count: int, palette: tuple):

        dir_heatmap = os.path.join(dir_result, "Heatmap")
        file_heatmap = os.path.join(dir_heatmap, "Compare.heatmap.{}k.split{}.txt".format(
            extend_bp,
            split_count
        ))
        dir_heatmap_extend = os.path.join(dir_heatmap, "Beds.Extend{}k".format(extend_bp))
        dir_heatmap_split = os.path.join(dir_heatmap, "Beds.Split")
        dir_signal_heatmap_split = os.path.join(dir_heatmap, "Signal.Split")

        file_bw_keys = list(file_bws.keys())
        names_bed = [file_bw_keys[0], "common", file_bw_keys[1]]
        if not os.path.exists(file_heatmap):
            if len(file_summits) != 2:
                # 不用summit对齐
                for k, file_single in tuple(file_bed_dict.items()):
                    file_temp = self.extend_bed_from_center(
                        file_bed=file_single,
                        bp=extend_bp*1000,
                        logger=logger,
                        dir_result=dir_heatmap_extend
                    )
                    file_bed_dict[k] = self.split_bed(
                        file_bed=file_temp,
                        count=split_count,
                        logger=logger,
                        dir_result=dir_heatmap_split
                    )
                file_beds_temp = [file_bed_dict[i] for i in names_bed]
            else:
                dir_heatmap_tosummit = os.path.join(dir_heatmap, "Beds.ToSummit")
                os.makedirs(dir_heatmap_tosummit, exist_ok=True)
                # summit对齐
                file_beds_temp = ["" for _ in range(3)]
                for i, k, b in zip(
                    range(3),
                    names_bed,
                    (names_bed[0], names_bed[2], names_bed[2])
                ):
                    file_temp = os.path.join(dir_heatmap_tosummit, "{}.{}.toSummits.bed".format(b, k))
                    self.exec_cmd("intersectBed -wa -a {} -b {} | cut -f 1-3 | sort | uniq > {}".format(
                        file_summits[b],
                        file_bed_dict[k],
                        file_temp
                    ), logger)
                    file_temp = self.extend_bed_from_center(
                        file_bed=file_temp,
                        bp=extend_bp*1000,
                        logger=logger,
                        dir_result=dir_heatmap_extend
                    )
                    file_beds_temp[i] = self.split_bed(
                        file_bed=file_temp,
                        count=split_count,
                        logger=logger,
                        dir_result=dir_heatmap_split
                    )

            signal_data = dict(zip(
                map(
                    lambda d: "{}@{}".format(d[0], d[1]),
                    product(file_bw_keys, names_bed)
                ),
                self.calc_signal(
                    dir_result=dir_signal_heatmap_split,
                    file_beds=file_beds_temp * len(file_bws),
                    file_bws=[file_bws[file_bw_keys[i // len(file_beds_temp)]] for i in range(len(file_beds_temp) * len(file_bw_keys))],
                    logger=logger
                )
            ))

            data = list()
            header = list()
            header_temp = set()
            for s in file_bws:
                data_temp = list()
                for ts in names_bed:
                    signal_temp = signal_data["{}@{}".format(s, ts)]
                    signal_temp = signal_temp.reshape((
                        len(signal_temp) // split_count,
                        split_count
                    ))
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
        # 排序
        df["order"] = df[list(map(str, range(split_count*2)))].mean(axis=1)
        df["order2"] = df["type"].map(lambda d: tuple(names_bed).index(d))
        df.sort_values(["order2", "order"], ascending=False, inplace=True)
        df.drop(["order2", "order"], axis=1, inplace=True)
        
        linewidth = 0.7

        y_min = [
            df[list(map(str, range(split_count*i, split_count*(i+1))))].max(axis=1).quantile(0.05) for i in range(len(file_bw_keys))
        ]
        y_max = [
            df[list(map(str, range(split_count*i, split_count*(i+1))))].max(axis=1).quantile(0.6) for i in range(len(file_bw_keys))
        ]
        norms = tuple(map(
            lambda d: mpl.colors.Normalize(vmin=0, vmax=d),
            y_max
        ))
        
        def get_map(d):
            color, y_min, y_max = d
            map_temp = mpl.colors.LinearSegmentedColormap.from_list("www", ("white", color), N=int(y_max * 10)+1)
            color_mid = mpl.colors.to_hex(map_temp(int(y_min*10)))

            return mpl.colors.LinearSegmentedColormap.from_list("www", ("white", color_mid, color))

        cmaps = tuple(map(get_map, zip(palette[1:], y_min, y_max)))
        
        fig = plt.figure(figsize=(len(file_bw_keys), 7))
        grid = plt.GridSpec(
            40,
            len(file_bw_keys) * 4,
            hspace=0,
            wspace=0.3,
            figure=fig,
        )

        def plot_single(ax: plt.Axes, df_temp: pd.DataFrame, color: str, norm, cmap, plot_text: bool):
            nonlocal linewidth, split_count, extend_bp, file_dict_num
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
                        "{}{} (n={})".format(
                            ls,
                            "" if ls == "common" else " Spec",
                            file_dict_num[ls]
                        ),
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

        x_max = len(file_bw_keys) - 1
        for i in range(len(file_bw_keys)):
            grid_temp = grid[:-4, i*4:i*4+4]
            ax = fig.add_subplot(grid_temp)
            type_temp = "type"
            plot_single(
                ax=ax,
                df_temp=df[[type_temp] + list(map(str, range(split_count*i, split_count*(i+1))))],
                color="black",
                norm=norms[i],
                cmap=cmaps[i],
                plot_text=i == x_max
            )

        for cmap, norm, start_temp, name_temp, y_max_temp in zip(
            cmaps,
            norms,
            range(0, len(file_bw_keys) * 4, 4),
            file_bw_keys,
            y_max
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
                [0, y_max_temp],
                ["     0.0", "{:.1f}     ".format(y_max_temp)]
            )

        plt.savefig(
            os.path.join(dir_result, "Heatmap.{}.{}k.split{}.pdf".format(
                "_".join(names_bed),
                extend_bp,
                split_count
            )),
            dpi=5000,
            bbox_inches="tight"
        )
        plt.close()


    def plot_box(self, file_bed: str, file_bws: tuple[str], dir_result: str, logger: logging.Logger, palette: tuple, name_beds: tuple, head_tail_common: str, title: str = ""):

        fig = plt.figure(figsize=(
            len(name_beds) * 0.4,
            3
        ))
        ax = plt.subplot()
        data_signal = self.calc_signal(
            os.path.join(dir_result, "Box.signal"),
            [file_bed for _ in range(len(file_bws))],
            list(file_bws),
            logger
        )
        
        title = "{}\n{}".format(
            ".".join((
                title,
                "_".join(name_beds)
            )).strip("."),
            head_tail_common
        ).strip(".").replace("..", ".")

        # 画box
        data = list()
        for n, s in zip(name_beds, data_signal):
            for i in s:
                data.append({
                    "signal": i,
                    "bed": n
                })
        sns.boxplot(
            pd.DataFrame(data),
            x="bed",
            y="signal",
            palette=palette[1:],
            showfliers=False,
            ax=ax,
            width=0.5,
        )
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
            k = i // 5
            li.set_color(colors_now[k])

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticklabels(
            [i.get_text() for i in ax.get_xticklabels()],
            # rotation=30,
            # va="top",
            # ha="right"
        )
        # plt.show()
        plt.savefig(
            os.path.join(dir_result, "Box.{}.pdf".format(title.replace(" ", "_").replace("\n", "."))),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close()


    def cor_heatmap(self, file_bed_region: str, name_bed_region: str, file_bw: tuple, dir_result: str, logger: logging.Logger):
        os.makedirs(dir_result, exist_ok=True)

        self.exec_cmd("multiBigwigSummary BED-file -b {} -o {} -p 8 --BED {}".format(
            " ".join(file_bw),
            os.path.join(dir_result, "Cor.{}.npz".format(name_bed_region)),
            file_bed_region
        ), logger)
        
        self.exec_cmd("plotCorrelation -in {} --corMethod spearman --skipZeros --plotTitle \"{}\" --whatToPlot scatterplot --plotFileFormat pdf -o {} --outFileCorMatrix {}".format(
            os.path.join(dir_result, "Cor.{}.npz".format(name_bed_region)),
            name_bed_region,
            os.path.join(dir_result, "..", "Cor.{}.pdf".format(name_bed_region)),
            os.path.join(dir_result, "Cor.{}.txt".format(name_bed_region))
        ), logger)    


    def main(
            self,
            region_names: str,
            name_align: str,
            title: str,
            palette_ori: str,
            calc_cor: str,
            **kwargs
        ):
        logger: logging.Logger = kwargs['logger']
        
        palette = tuple(["white"] + palette_ori.split("\n"))
        file_beds = tuple(kwargs['return'][0].split(","))
        name_beds = tuple(filter(
            lambda d: d != "",
            map(
                lambda dd: dd.strip(),
                region_names.split("\n")
            )
        ))
        
        file_bws_ori = tuple(kwargs['return'][2].split(","))
        file_bws = tuple(filter(
            lambda d: os.path.exists(d),
            file_bws_ori
        ))

        
        file_summits_ori = tuple(kwargs['return'][1].split(","))
        file_summits = tuple(filter(
            lambda d: os.path.exists(d),
            file_summits_ori
        ))
        
        file_beds_ori = file_beds
        file_beds = tuple(filter(
            lambda d: os.path.exists(d),
            file_beds_ori
        ))
        num_beds = len(file_beds)
        assert num_beds > 0, "file_beds not exists: {}".format(file_beds_ori)

        
        if (region_names == "auto") or (len(name_beds) != len(file_beds)):
            name_beds = tuple(map(
                lambda d: os.path.basename(d)[:-4],
                file_beds
            ))
        logger.info("name_beds: {}, name_align: {}".format(
            name_beds,
            name_align
        ))
        head_tail_common, name_beds, name_align = self.handle_name(name_beds, name_align, logger)
        logger.info("common: {}, name_beds: {}, name_align: {}".format(
            head_tail_common,
            name_beds,
            name_align
        ))
        # head_tail_common = ""
        
        dir_result = os.getcwd()
        dir_bed_spec = os.path.join(dir_result, "Bed.Spec")
        # RegionChange plot
        self.bed_change(
            file_beds=file_beds,
            dir_result=dir_bed_spec,
            logger=logger,
            palette=palette,
            name_beds=name_beds,
            head_tail_common=head_tail_common,
            name_align=name_align,
            title=title if title != "no" else ""
        )
        
        if (name_align not in name_beds) and (num_beds == 2):
            file_dict = dict()
            file_dict_num = dict()
            file_dict["common"] = os.path.join(dir_bed_spec, "{}.{}.Common.bed".format(
                "_".join(name_beds),
                head_tail_common
            ).strip(".").replace("..", "."))
            for name_temp in name_beds:
                file_dict[name_temp] = os.path.join(dir_bed_spec, "{}.{}.Spec.bed".format(
                    name_temp,
                    head_tail_common
                ).strip(".").replace("..", "."))

            for k in file_dict:
                file_dict_num[k] = self.region_num(file_dict[k])
            
            self.venn2(
                area1_ori=file_dict_num[name_beds[0]],
                area2_ori=file_dict_num[name_beds[1]],
                area_cross_ori=file_dict_num["common"],
                labels=name_beds,
                title="{}_vs_{}".format(*name_beds)
            )
            if len(file_bws) >= 2:
                file_bws_dict = dict(zip(name_beds, file_bws))
                file_bws_dict.update(dict(map(
                    lambda d: (os.path.basename(d)[:-3], d),
                    file_bws[2:]
                )))
                self.plot_heatmap(
                    file_dict,
                    file_dict_num,
                    file_bws_dict,
                    dict(zip(name_beds, file_summits)),
                    dir_result,
                    logger,
                    extend_bp=5,
                    split_count=200,
                    palette=palette
                )
        
        if len(file_bws) == 0:
            logger.warn("BigWigs files not exists, no plot signal !")
            return
        
        # BigWig Signal box plot
        file_bed_all_num = os.path.join(dir_bed_spec, "{}.{}.All.bed".format(
            "_".join(name_beds),
            head_tail_common
        ).strip(".").replace("..", "."))
        self.plot_box(
            file_bed=file_bed_all_num,
            file_bws=file_bws,
            dir_result=dir_result,
            logger=logger,
            palette=palette,
            name_beds=name_beds,
            head_tail_common=head_tail_common,
            title=title if title != "no" else ""
        )

        if calc_cor.lower() == "false":
            logger.warn("No calc cor !")
            return

        # 根据bed画Cor
        file_bed_all = tuple(map(
            lambda d: os.path.join(dir_bed_spec, d),
            filter(
                lambda dd: dd.endswith(".bed"),
                os.listdir(dir_bed_spec)
            )
        ))

        func_list = list()
        arg_list = list()
        for file_bed_temp in file_bed_all:
            func_list.append(self.cor_heatmap)
            arg_list.append(dict(
                file_bed_region=file_bed_temp,
                name_bed_region=os.path.basename(file_bed_temp)[:-4],
                file_bw=file_bws,
                dir_result=os.path.join(dir_result, "Cor"),
                logger=logger
            ))
        self.multi_task(func_list, arg_list, logger, 8)

        logger.info("done")


TaskSrc.__doc__ = """ SpecificRegion

计算不同Region之间的区别

### 一、输入参数说明

#### 1. 参数：`Region名字`

每类regin的名字，每行为一个name，对应每个bed文件，auto为自动根据bed文件命名，利用.-_等分隔，去除前后相同的，即为名字

#### 2. 参数：`标准Region类别`

以某一类region为标准，其余的做对比。no标准region则需要只有两类region，相互对比

#### 3. 参数：`结果名字`

结果名字，如果为无则自动根据类别名字计算

#### 4. 参数：`颜色列表`

每行一个颜色

#### 5. 数据：`file_beds`

各个bed文件，使用英文逗号间隔


### 二、输出文件说明


### 三、任务作者

Dong LIU
"""