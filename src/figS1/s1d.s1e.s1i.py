#!/usr/bin/env python
# cython: language_level=3
# -*- coding: UTF-8 -*-
'''
@File    :   CorrelationForSamples.py
@Modif   :   2024/01/25 17:01:21
@Author  :   winter
@Version :   0.1
@Contact :   winter_lonely@126.com
@License :   (C)Copyright winter
@Desc    :   
'''
# %%
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import matplotlib as mpl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pyg2plot import Plot
from itertools import combinations
from umap.umap_ import UMAP
import pandas as pd
import numpy as np
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
        
    def rm_batch_effect(self, file_fpkm: str, file_metadata: str, result_dir: str):
        """ 根据fpkm和metadata去除批次效应
        """
        df_metadata = pd.read_excel(file_metadata)
        df_fpkm = pd.read_table(file_fpkm, index_col=0)

        batch_dict = dict(map(
            lambda d: (d[1], d[0]),
            enumerate(df_metadata['experiment'].unique(), start=1)
        ))
        batch_dict = dict(map(
            lambda d: (d[0], batch_dict[d[1]]),
            zip(
                df_metadata["sample"],
                df_metadata["experiment"],
            )
        ))

        df_fpkm = df_fpkm[df_metadata['sample'].to_list()]
        df_fpkm.to_csv("fpkm.tmp.txt", sep="\t")

        batch_data = df_fpkm.columns.map(lambda d: batch_dict[d]).to_list()

        res_file_name = os.path.join(
            result_dir,
            os.path.basename(file_fpkm).replace(".txt", ".rm.txt")
        )

        r_str = """
library(edgeR)


experiment_batch <- as.factor(c(""" + ", ".join(map(str, batch_data)) + """))


data <- read.table("fpkm.tmp.txt",header=T,row.names=1,sep="\\t")

data_rm <- removeBatchEffect(data,batch=experiment_batch)

write.table(data_rm, file = \"""" + res_file_name + """\", sep = "\\t")
"""

        file_r = "removeBatchEffect.r"
        with open(file_r, "w", encoding="UTF-8") as f:
            f.write(r_str)

        os.system("/usr/bin/Rscript-4.2 {}".format(file_r))
        os.system("rm -rf {}".format(file_r))
        os.system("rm -rf fpkm.tmp.txt")
        
        return res_file_name

    def sample_cor(self, file_fpkm: str, file_meta: str, data_dir: str, img_dir: str, by_sample: bool, sample_order: list, logger: logging.Logger):

        if by_sample:
            groupby = "sample"
        else:
            groupby = "tag"

        logger.info("read table...")
        
        df_fpkm = pd.read_table(file_fpkm, index_col=0)
        df_meta = pd.read_excel(file_meta)

        cols_all = list(df_fpkm.columns)
        sample_dict = dict()
        df_meta = df_meta[df_meta["sample"].isin(cols_all)]
        for i, df_temp in df_meta.groupby(groupby):
            sample_dict[i.strip()] = df_temp["sample"].to_list()
        
        for i in cols_all:
            df_fpkm[i] = df_fpkm[i].astype("float")
        

        cmap = mpl.colors.LinearSegmentedColormap.from_list("www", ("darkblue", "lightblue", "white", "red", "darkred"))
        def plot_single(df_res: pd.DataFrame, annot: bool, title: str):
            nonlocal logger, cmap, sample_order, img_dir
            logger.info("ploting {}...".format("annot " if annot else ""))
            unit_size = 0.7 if annot else 0.5
            fig = sns.clustermap(
                df_res,
                cmap=cmap,
                annot=annot,
                figsize=(
                    len(df_res.columns)*unit_size+3,
                    len(df_res.index)*unit_size+3
                )
            )
            # fig.ax_col_dendrogram.set_title("Cor")
            fig.savefig(os.path.join(img_dir, "Cor.{}.cluster.{}pdf".format(title, "annot." if annot else "")), dpi=500, bbox_inches='tight')
            plt.close()

            if "sample" in title:
                cols_order = sample_order
            else:
                cols_order = list(df_res.columns)
                cols_order.sort()
            fig = plt.figure(figsize=(
                len(df_res.columns)*unit_size+3,
                len(df_res.index)*unit_size+3
            ))
            sns.heatmap(
                df_res.loc[cols_order][cols_order],
                cmap=cmap,
                annot=annot
            )
            # fig.ax_col_dendrogram.set_title("Cor")
            plt.savefig(os.path.join(img_dir, "Cor.{}.{}pdf".format(title, "annot." if annot else "")), dpi=500, bbox_inches='tight')
            plt.close()

        def calc_single(fpkm_func, title):
            nonlocal sample_dict, df_fpkm
            logger.info("calc cor, {} ...".format(title))
            res = dict()
            logger.info("calc cor, {}, mean ...".format(title))
            for i in sample_dict.keys():
                logger.info("calc cor, {}, mean, {} ...".format(title, i))
                df_fpkm["temp_tag_{}".format(i)] = df_fpkm[sample_dict[i]].mean(axis=1)
            logger.info("calc cor, {}, corrcoef ...".format(title))
            for i, j in combinations(sample_dict.keys(), 2):
                logger.info("calc cor, {}, corrcoef, {} & {} ...".format(title, i, j))
                if i not in res:
                    res[i] = dict()
                    res[i][i] = 1
                if j not in res:
                    res[j] = dict()
                    res[j][j] = 1
                cor_temp = np.corrcoef(
                    fpkm_func(df_fpkm["temp_tag_{}".format(i)]).to_numpy(),
                    fpkm_func(df_fpkm["temp_tag_{}".format(j)]).to_numpy()
                )[0][1]
                # cor_temp = cor_temp * cor_temp
                res[i][j] = cor_temp
                res[j][i] = cor_temp
            df_res = pd.DataFrame(res)
            df_res.to_csv(os.path.join(data_dir, "Cor.{}.txt".format(title)), sep="\t")

            plot_single(df_res, False, title)
            plot_single(df_res, True, title)

        fpkm_log = lambda d: np.log2(d+1)
        calc_single(fpkm_log, "{}.log".format(groupby))
        fpkm_ori = lambda d: d
        calc_single(fpkm_ori, groupby)


    def plot_tsne_or_pca(self, file_data: str, plot_type: str, group: str, result_file: str, palette: list, logger: logging.Logger):
        
        df = pd.read_table(file_data)
        title = os.path.basename(result_file)

        pca_cols = list(filter(
            lambda d: plot_type in d,
            df.columns
        ))
        pca_cols.sort()

        def sp(x, y, ax_now):
            nonlocal pca_cols, df, palette, group
            sns.scatterplot(
                df,
                x=pca_cols[x],
                y=pca_cols[y],
                hue=group,
                linewidth=0,
                palette=palette,
                zorder=10,
                ax=ax_now
            )
            
        fig = plt.figure(figsize=(5, 5))
        grid = plt.GridSpec(16, 16, hspace=1, wspace=1)
        main_ax = fig.add_subplot(grid[:, :])
        sp(0, 1, main_ax)
        main_ax.xaxis.set_ticklabels([])
        main_ax.yaxis.set_ticklabels([])
        main_ax.xaxis.set_tick_params(left=False)
        main_ax.yaxis.set_tick_params(left=False)
        
        main_ax.set_xlabel("t{}".format(pca_cols[0][1:]) if plot_type.lower() == "tsne" else pca_cols[0])
        main_ax.set_ylabel("t{}".format(pca_cols[1][1:]) if plot_type.lower() == "tsne" else pca_cols[1])

        legend_t = main_ax.legend()
        legend_t_labels = list(map(
            lambda d: d.get_text(),
            legend_t.texts
        ))
        legend_t_handlers = legend_t.legendHandles
        main_ax.legend(
            legend_t_handlers,
            legend_t_labels,
            frameon=False,
            bbox_to_anchor=(1, 0.5),
            loc="center left"
        )
        plt.savefig("{}.{}.{}.pdf".format(result_file, pca_cols[0], pca_cols[1]), bbox_inches='tight')
        plt.close()
        
        if len(pca_cols) == 3:
            fig = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(16, 16, hspace=1, wspace=1)
            main_ax = fig.add_subplot(grid[:8, 8:])
            y_ax = fig.add_subplot(grid[:8, :8], xticklabels=[], sharey=main_ax)
            x_ax = fig.add_subplot(grid[8:, 8:], yticklabels=[], sharex=main_ax)

            
            sp(0, 2, x_ax)
            sp(2, 1, y_ax)
            sp(0, 1, main_ax)
            main_ax.xaxis.set_ticklabels([])
            main_ax.yaxis.set_ticklabels([])
            main_ax.xaxis.set_tick_params(left=False)
            main_ax.yaxis.set_tick_params(left=False)
            main_ax.xaxis.set_label_position("top")
            main_ax.yaxis.set_label_position("right")
            y_ax.xaxis.set_ticklabels([])
            y_ax.yaxis.set_ticklabels([])
            y_ax.xaxis.set_tick_params(left=False)
            y_ax.yaxis.set_tick_params(left=False)
            y_ax.xaxis.set_label_position("top")
            y_ax.set_ylabel("")
            x_ax.xaxis.set_ticklabels([])
            x_ax.yaxis.set_ticklabels([])
            x_ax.xaxis.set_tick_params(left=False)
            x_ax.yaxis.set_tick_params(left=False)
            x_ax.set_xlabel("")
            x_ax.yaxis.set_label_position("right")

            x_ax.legend_ = None
            y_ax.legend_ = None
            legend_t = main_ax.legend()
            legend_t_labels = list(map(
                lambda d: d.get_text(),
                legend_t.texts
            ))
            legend_t_handlers = legend_t.legendHandles
            main_ax.legend(
                legend_t_handlers,
                legend_t_labels,
                frameon=False,
                bbox_to_anchor=(-0.55, -0.55),
                loc=10
            )
            plt.savefig("{}.pdf".format(result_file), bbox_inches='tight')
            plt.close()

        with open(os.path.join(os.path.dirname(__file__), "plotly-2.24.1.min.js")) as f:
            plotly_str = f.read()

        data = [ dict(d) for _, d in df.iterrows() ]
        sample = "sample"
        for x, y in combinations(pca_cols, 2):
            title_now = "{}, {} * {}".format(title, x, y)
            scatter = Plot("Scatter")
            scatter.set_options({
                "appendPadding": 10,
                "data": data,
                "xField": x,
                "yField": y,
                "shape": 'circle',
                "colorField": group,
                "tooltip": {
                    "fields": list(df.columns)
                },
                "legend": {
                    "flipPage": False
                },
                "color": palette,
                "size": 4,
                "yAxis": {
                    "nice": True,
                    "label": y,
                    "line": {
                        "style": {
                            "stroke": '#aaa',
                        },
                    },
                },
                "xAxis": {
                    "label": x,
                    "grid": {
                        "line": {
                            "style": {
                                "stroke": '#eee',
                            },
                        },
                    },
                    "line": {
                        "style": {
                            "stroke": '#aaa',
                        },
                    },
                },
            })
            result_file_html = "{}.{}.{}.g2plot.html".format(result_file, x, y)
            # scatter.render(result_file_html)
            self.render_pyg2plot(scatter, result_file_html)
            with open(result_file_html, "a") as f:
                f.write("""
                    <style>
                    h3{
                        text-align: center;
                        position: absolute;
                        top: 10px;
                        right: 30px;
                    }
                    </style>
                    <script>
                        document.title = \"""" + title_now + """\";
                        document.body.insertAdjacentHTML("beforeEnd", "<h3>""" + title_now + """</h3>");
                    </script>
                """)
            
            scatter_list = list()
            group_num = len(df[group].unique())
            for color_temp, (group_name, df_temp), index in zip(
                palette * 5,
                df.groupby(group),
                range(group_num)
            ):
                plot_data_temp = (
                    str(index),
                    ",".join(map(str, df_temp[x].to_list())),
                    ",".join(map(str, df_temp[y].to_list())),
                    group_name,
                    "','".join(map(str, df_temp[sample].to_list())),
                    color_temp,
                    x,
                    y,
                    sample
                )
                scatter_list.append("""
const scatter@@@ = {
    x: [@@@],
    y: [@@@],
    mode: "markers",
    type: "scatter",
    name: "@@@",
    text: ['@@@'],
    marker: { size: 6, color: '@@@' },
    hovertemplate: '<i><b>@@@</b>: %{x:.2f}' +
        '<br><b>@@@</b>: %{y:.2f}<br>' +
        '<b>@@@</b>: %{text}',
};
""")
                for i in range(len(plot_data_temp)):
                    scatter_list[-1] = scatter_list[-1].replace("@@@", plot_data_temp[i], 1)



            plot_data = (
                title,
                plotly_str,
                "\n".join(scatter_list),
                ",".join(map(lambda d: "scatter{}".format(d), range(group_num))),
                title,
                x,
                y
            )

            html_str = """
<!DOCTYPE html>
<head>
    <meta charset="UTF-8" />
    <title>@@@</title>
</head>
<body style="height:750px;">
	<div id='myDiv' style="height:600px;width:850px;margin:0 auto;"><!-- Plotly chart will be drawn inside this DIV --></div>
</body>
<script>
@@@
</script>
<script>

@@@

const data = [ @@@ ];

const layout = {
    title:'@@@',
    xaxis: { title: '@@@', zeroline: false },
    yaxis: { title: '@@@', zeroline: false },
};

Plotly.newPlot('myDiv', data, layout);
</script>
</html>
"""
            for i in range(len(plot_data)):
                html_str = html_str.replace("@@@", plot_data[i], 1)
            with open(result_file_html.replace("g2plot", "plotly"), "w", encoding="UTF-8") as f:
                f.write(html_str)
            


    def run_tsne_or_pca(self, file_fpkm: str, file_meta: str, data_dir: str, img_dir: str, palette: list, logger: logging.Logger, ext_name: str):
        df_fpkm = pd.read_table(file_fpkm, index_col=0).T
        
        df_meta = pd.read_excel(file_meta)

        if len(df_fpkm.index) > 30:
            func_all = [TSNE, UMAP, PCA]
        else:
            func_all = [PCA]

        
        def run_single(func, df_fpkm, title_ext):
            nonlocal data_dir, df_meta, palette

            func_name = func.__name__
            
            n_components = 3 if func_name == "PCA" else 2

            res_func = func(n_components=n_components)
            res = res_func.fit_transform(df_fpkm)

            file_result_now = os.path.join(data_dir, "{}{}.txt".format(func_name, title_ext))

            columns = list()
            for i in range(1, n_components+1):
                columns.append("{}{}{}".format(
                    func_name,
                    i,
                    "_{:.1f}%".format(res_func.explained_variance_ratio_[i-1]*100) if func_name == "PCA" else ""
                ))

            df = pd.DataFrame(res, columns=columns)
            df['sample'] = df_fpkm.index
            df = pd.merge(
                left=df,
                right=df_meta,
                how="left",
                on="sample"
            )
            df.to_csv(file_result_now, sep="\t", index=False)

            self.plot_tsne_or_pca(
                file_data=file_result_now,
                plot_type=func_name,
                group="tag",
                result_file=os.path.join(img_dir, "{}{}".format(func_name, title_ext)),
                palette=palette,
                logger=logger
            )


        for func in func_all:
            run_single(func, df_fpkm, ext_name)
            # 去批次后不取log
            if ".rm" in ext_name:
                continue
            df_fpkm_log = np.log2(df_fpkm + 1)
            run_single(func, df_fpkm_log, "{}.log".format(ext_name))


    def main_raw(self, fpkm_file: str, meta_file: str, result_dir: str, palette: list, sample_order: list, logger: logging.Logger):
        data_dir = os.path.join(result_dir, "data")
        img_dir = os.path.join(result_dir, "img")

        for i in [result_dir, data_dir, img_dir]:
            if os.path.exists(i):
                continue
            os.makedirs(i)

        logger.info("Check meta ...")
        
        df_meta = pd.read_excel(meta_file)
        df_fpkm = pd.read_table(fpkm_file, index_col=0)
        
        cols_gap = set(("experiment", "sample", "tag")) - set(df_meta.columns)
        assert len(cols_gap) == 0, "meta_file columns error: need {}".format(cols_gap)
        
        cols_gap = set(df_meta["sample"].to_list()) - set(df_fpkm.columns)
        assert len(cols_gap) == 0, "fpkm_file samples error: need {}".format(cols_gap)
        need_rm_batch = len(df_meta["experiment"].unique()) > 1

        # 去除不需要的列
        logger.info("Remove samples in fpkm file ...")
        cols_all = df_meta["sample"].to_list()
        df_fpkm = df_fpkm[cols_all]
        df_fpkm.to_csv(fpkm_file, sep="\t")

        del cols_gap
        del df_meta
        del df_fpkm

        if len(cols_all) > 100:
            logger.warn("samples num({}) > 100, skip cor by sample".format(len(cols_all)))
        else:
            logger.info("Cor task by sample ...")
            self.sample_cor(
                file_fpkm=fpkm_file,
                file_meta=meta_file,
                data_dir=data_dir,
                img_dir=img_dir,
                by_sample=True,
                sample_order=sample_order,
                logger=logger
            )

        logger.info("Cor task by tag ...")

        self.sample_cor(
            file_fpkm=fpkm_file,
            file_meta=meta_file,
            data_dir=data_dir,
            img_dir=img_dir,
            by_sample=False,
            sample_order=sample_order,
            logger=logger
        )

        logger.info("pca task ...")
        
        self.run_tsne_or_pca(
            file_fpkm=fpkm_file,
            data_dir=data_dir,
            file_meta=meta_file,
            img_dir=img_dir,
            palette=palette,
            logger=logger,
            ext_name=""
        )

        if need_rm_batch:

            logger.info("pca task with remove batch effect ...")
            
            self.run_tsne_or_pca(
                file_fpkm=self.rm_batch_effect(
                    file_fpkm=fpkm_file,
                    file_metadata=meta_file,
                    result_dir=result_dir
                ),
                data_dir=data_dir,
                file_meta=meta_file,
                img_dir=img_dir,
                palette=palette,
                logger=logger,
                ext_name=".rm"
            )
        else:
            logger.info("Don't run pca with remove batch effect")

        logger.info("Done")

    def main(
            self,
            palette: list,
            sample_order: str,
            **kwargs
        ):

        logger: logging.Logger = kwargs['logger']
        
        gene_all = set(kwargs['return'][2].split(",")) - set(("all",))
        result_dir_base = os.getcwd()

        result_dir_all = os.path.join(result_dir_base, "all")
        if not os.path.exists(result_dir_all):
            os.makedirs(result_dir_all)

        df_list = list()
        for file_single in kwargs['return'][0].split(","):
            assert os.path.exists(file_single), "fpkm file not exists: {}".format(file_single)
            df_fpkm = pd.read_table(file_single, low_memory=False)
            df_fpkm_columns = set(df_fpkm.columns)
            try:
                if ("refseq" in df_fpkm_columns) and ("gene_symbol" in df_fpkm_columns):
                    df_fpkm["gene_symbol"] = df_fpkm.apply(lambda d: "{}({})".format(d["gene_symbol"], d["refseq"]), axis=1)
                    df_fpkm.set_index("gene_symbol", drop=True, inplace=True)
                    df_fpkm.drop(["chr", "strand", "start", "end", "refseq", "num_exons", "length"], inplace=True, axis=1)
                else:
                    df_fpkm.set_index("gene_symbol", drop=True, inplace=True)
            except Exception as e:
                logger.error("{}: {}".format(file_single, str(e)))
                return
            df_fpkm.reset_index(inplace=True)
            df_list.append(df_fpkm)

        df_fpkm = df_list[0]
        for i in range(1, len(df_list)):
            logger.info("merge: {}".format(df_list[i].columns))
            df_fpkm = pd.merge(
                left=df_fpkm,
                right=df_list[i],
                on="gene_symbol",
                how="outer"
            )
        df_fpkm.dropna(inplace=True)
        df_fpkm.set_index("gene_symbol", drop=True, inplace=True)
        
        fpkm_all = os.path.join(result_dir_all, "fpkm.{}.txt".format(len(df_fpkm)))
        df_fpkm.to_csv(fpkm_all, sep="\t")
        

        if sample_order.lower() == "auto":
            df_meta = pd.read_excel(kwargs['return'][1])
            assert "sample" in df_meta.columns, "meta file no sample !"
            sample_order_new = list(map(str, df_meta["sample"].to_list()))
            sample_order_new.sort()
            del df_meta
        else:
            sample_order_new = self.handle_str_list(sample_order)

        # 所有基因
        self.main_raw(
            fpkm_file=fpkm_all,
            meta_file=kwargs['return'][1],
            result_dir=result_dir_all,
            palette=palette,
            sample_order=sample_order_new,
            logger=kwargs['logger']
        )
        # 给定的基因
        for file_single in gene_all:
            if not os.path.exists(file_single):
                continue
            result_dir_now = os.path.join(result_dir_base, os.path.basename(file_single)[:-4])
            if os.path.exists(result_dir_now):
                raise Exception("Duplicate gene file name !")
            os.makedirs(result_dir_now)
            with open(file_single) as f:
                genes_need_now = set(filter(
                    lambda d: d != "",
                    map(
                        lambda dd: dd.strip(),
                        f
                    )
                ))
            df_fpkm_now = df_fpkm[df_fpkm.index.map(lambda d: d.split("(",1)[0]).isin(genes_need_now)]
            assert len(df_fpkm_now) > 0, "{} gene no match !".format(os.path.basename(file_single))
            file_fpkm_now = os.path.join(result_dir_now, "fpkm.{}.txt".format(len(df_fpkm_now)))
            df_fpkm_now.to_csv(file_fpkm_now, sep="\t")
            del df_fpkm_now
            self.main_raw(
                fpkm_file=file_fpkm_now,
                meta_file=kwargs['return'][1],
                result_dir=result_dir_now,
                palette=palette,
                sample_order=sample_order_new,
                logger=kwargs['logger']
            )




TaskSrc.__doc__ = """ CorrelationForSamples

计算样本的相关性，会根据样本标签和样本分别做相关性，即每个样本之间的相关性和每类样本之间的相关性。

### 一、输入参数说明

#### 1. 数据：`file_fpkm`

fpkm文件。transcript fpkm时，前几列为chr、strand、start、end、refseq、num_exons、length、gene_symbol。genesymbol_fpkm时第一列为gene_symbol

#### 2. 数据：`file_meta`，[模板文件](/images/Correlation.meta.xlsx)

excel文件，存在experiment、sample和tag三列，experiment为实验名称，根据不同实验名称去批次效应，sample为样本id，tag为样本标签，注意样本id需要与fpkm文件的列名对应

metadata数据示例
![示例](/images/Correlation.meta.png)

#### 3. 数据：`file_genes`

利用给定的基因计算相关性，每个txt文件为一个基因列表，文件每行一个基因。默认为all，使用所有基因

### 二、输出文件说明

#### 1. 相关性的数据

顶层的每个文件夹为使用的基因计算的相关性，`.log`为使用log2(fpkm+1)作为计算数据，否则使用原始fpkm计算。

#### 2. 相关性的数据

`Cor`开头文件，`.tag`为根据标签分组均值做相关性，`.sample`为每个样本做相关性。`.annot`为显示相关性的数值，否则不显示。`.log`为使用log2(fpkm+1)作为计算数据，否则使用原始fpkm计算。

#### 3. 降维的数据

`PCA`、`tSNE`或`UMAP`开头文件。如果样本数量多（大于30个），则使用`tSNE`和`UMAP`降维，否则使用`PCA`降维。`.rm`为去除批次效应

### 三、任务作者

Dong LIU
"""