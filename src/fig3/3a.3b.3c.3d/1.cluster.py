# %%
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
import concurrent.futures as fu
import matplotlib.pyplot as plt
import matplotlib as mpl
from PyPDF2 import PdfFileMerger
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import threading
import traceback
import logging
import random
import pickle
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
cluster_num = 12
dir_result = "/mnt/liudong/task/20240116_HT/result.sort/fig3/3a.3b.3c"
os.makedirs(dir_result, exist_ok=True)
os.chdir(dir_result)
# %%
dir_cluster = os.path.join(dir_result, "fpkm_cluster")
dir_plot = os.path.join(dir_result, "fpkm_plot")
os.makedirs(dir_cluster, exist_ok=True)
os.makedirs(dir_plot, exist_ok=True)
# %%
file_fpkm = os.path.join(dir_result, "GeneSymbol_fpkm.final.filter_by_DEGs.0.5.txt")
assert os.path.exists(file_fpkm), "fpkm file not exists !"
# %%
df_fpkm = pd.read_table(file_fpkm, index_col=0)
# %%
cols_mef_esc = list(filter(
    lambda d: d.startswith(("ESC", "MEF")),
    df_fpkm.columns
))
# %%
df_fpkm[cols_mef_esc].to_csv("MEF_ESC.txt", sep="\t")
# %%
df_fpkm["MEF"] = df_fpkm[list(filter(
    lambda d: d.startswith("MEF"),
    cols_mef_esc
))].mean(axis=1)
df_fpkm["ESC"] = df_fpkm[list(filter(
    lambda d: d.startswith("ESC"),
    cols_mef_esc
))].mean(axis=1)
# %%
df_fpkm.drop(cols_mef_esc, axis=1, inplace=True)
# %%
samples_dict = dict()
for i, v in zip(
    tuple(map(
        lambda d: d.split("_"),
        df_fpkm.columns
    )),
    df_fpkm.columns
):
    if len(i) == 1:
        samples_dict[i[0]] = v
    else:
        if i[0] not in samples_dict:
            samples_dict[i[0]] = dict()
        if i[1] not in samples_dict[i[0]]:
            samples_dict[i[0]][i[1]] = list()
        samples_dict[i[0]][i[1]].append(v)
# %%
key_compare = list()
day_all = list()
for s in samples_dict:
    if isinstance(samples_dict[s], dict):
        key_compare.append(s)
        day_all.extend(list(samples_dict[s]))
    else:
        day_all.append(samples_dict[s])
day_all = list(set(day_all))

pat_d = re.compile("\d+")
def sort_func(d: str):
    global pat_d
    t = "".join(pat_d.findall(d))
    if len(t) == 0:
        if d == "MEF":
            return -100
        elif d == "ESC":
            return 100
        return ord(d[0])
    else:
        return int(t)
day_all.sort(key=sort_func)
# %%
col_need = list()
for s in samples_dict:
    if not isinstance(samples_dict[s], dict):
        col_need.append(samples_dict[s])
        continue
    for d in samples_dict[s]:
        v = "{}_{}".format(s, d)
        df_fpkm[v] = df_fpkm[samples_dict[s][d]].mean(axis=1)
        col_need.append(v)
df_fpkm = df_fpkm[col_need]
# %%
file_model = os.path.join(dir_result, "kmeans_model.pkl")
file_cluster_result = os.path.join(dir_result, "kmeanscluster.fpkm.txt")
dir_cluster_result = os.path.join(dir_result, "fpkm_cluster")

if not os.path.exists(dir_cluster_result):
    os.mkdir(dir_cluster_result)

random.seed(10)
class KMeans(object):
    """
    https://www.jb51.net/article/234623.htm
    """
    # k是分组数；tolerance‘中心点误差'；max_iter是迭代次数
    def __init__(self, n_clusters=10, tolerance=0.00000000001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.labels_ = []
 
    def fit(self, data):
        self.centers_ = {}
        self.labels_ = [ '' for _ in range(len(data)) ]
        for i in range(self.k_):
            self.centers_[i] = data[random.randint(0,len(data))]
        # print('center', self.centers_)
        for i in tqdm(range(self.max_iter_)):
            self.clf_ = {} #用于装归属到每个类中的点[k,len(data)]
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for index, feature in enumerate(data):
                distances = [] #装中心点到每个点的距离[k]
                for center in self.centers_:
                    # 欧拉距离
                    dis = np.linalg.norm(feature - self.centers_[center])
                    # 相关性
                    # dis = - np.corrcoef(feature, self.centers_[center])[0][1] + 1
                    # print(dis, feature.shape, self.centers_[center].shape, feature.shape, self.centers_[center])
                    distances.append(dis)
                classification = distances.index(min(distances))
                self.labels_[index] = classification
                self.clf_[classification].append(feature)
 
            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
 
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)
 
            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers) > self.tolerance_:
                    optimized = False
            if optimized:
                break
 
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
# %%
df_fpkm.fillna(0.0, inplace=True)

df_fpkm = df_fpkm[df_fpkm.apply(
    lambda d: np.any(d >= 1),
    axis=1
)]

df_now = df_fpkm.copy()

for i in df_now.columns:
    df_now[i] = np.log2(df_now[i] + 1)

k = KMeans(
    n_clusters=cluster_num
)

k.fit(df_now.to_numpy())

with open(file_model, "wb") as f:
    pickle.dump(k, f)

df_fpkm['cluster'] = k.labels_

df_fpkm.to_csv(
    file_cluster_result,
    sep="\t"
)

for cluster, df_temp in df_fpkm.groupby("cluster"):
    df_temp.drop("cluster", axis=1).to_csv(
        os.path.join(dir_cluster_result, "Cluster{}.txt".format(int(cluster) + 1)),
        sep="\t"
    )

# %%
def box_plot(data: list, dir_result: str, title: str, result_dir: str = ".", ylabel: str = "log2(fpkm+1)"):
    file_name_jpg = "{}.box.pdf".format(os.path.join(dir_result, title))

    data_now = list()
    keys_compare = list()
    for i in range(len(data)):
        for v in np.log2(np.asarray(data[i]["value"]) + 1):
            data_now.append({
                "day": data[i]["day"],
                "type": data[i]["type"],
                ylabel: v
            })
            if data[i]["type"] not in key_compare:
                key_compare.append(data[i]["type"])
    df_now = pd.DataFrame(data_now)
    
    def box_outliers(ser):
        # 对待检测的数据集进行排序
        new_ser = ser.sort_values()
        # 判断数据的总数量是奇数还是偶数
        if new_ser.count() % 2 == 0:
            # 计算Q3、Q1、IQR
            Q3 = new_ser[int(len(new_ser) / 2):].median()
            Q1 = new_ser[:int(len(new_ser) / 2)].median()
        elif new_ser.count() % 2 != 0:
            Q3 = new_ser[int((len(new_ser)-1) / 2):].median()
            Q1 = new_ser[:int((len(new_ser)-1) / 2)].median()
        IQR = round(Q3 - Q1, 1)
        ma = round(Q3+1.5*IQR, 1)
        mi = round(Q1-1.5*IQR, 1)
        return (ser.quantile(.95), ser.quantile(.05))
    
    f = plt.figure(figsize=(18, 4))
    col = "k"
    f.add_subplot(1,2,1)
    if len(keys_compare) == 3:
        colors = ["#CE0013","#16557A","#C7A609"]
    else:
        colors = ["#16557A","#C7A609"]
    fig = sns.boxplot(x="day", y=ylabel, hue="type", data=df_now, palette=colors, showfliers=False)
    y_max, y_min = box_outliers(df_now[ylabel])
    y_unit = (df_now[ylabel].max() - df_now[ylabel].min()) / 12
    # fig.set_ylim(y_min - 0.5 * y_unit, y_max + (len(keys_compare)*3 - 4) * y_unit)
    fig.set_title(file_name_jpg.split(".",1)[0])
    fig.set_xlabel("")

    def t_test(df_now_day, type_list, day):
        nonlocal ylabel
        d1 = df_now_day[(df_now_day['type'] == type_list[0]) & (df_now_day['day'] == day)][ylabel]
        d2 = df_now_day[(df_now_day['type'] == type_list[1]) & (df_now_day['day'] == day)][ylabel]
        t = stats.ttest_rel(d1, d2).pvalue
        assert t >= 0, "{} {} {}".format(type_list, day, t)
        if t < 0.001:
            return "***"
        if t < 0.01:
            return "**"
        if t < 0.05:
            return "*"
        return "-"

    # statistical annotation
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
            y, h = box_outliers(df_now_day[ylabel])[0] + 1 * y_unit, 0.1 * y_unit
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            # Oct4Nanog与Oct4NanogN70
            t = t_test(df_now_day, (keys_compare[1], keys_compare[2]), day)
            x1, x2 = x, x + 0.27   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = box_outliers(df_now_day[ylabel])[0] + 1.5 * y_unit, 0.1 * y_unit
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
            # Oct4Nanog与Oct4NanogN70
            t = t_test(df_now_day, (keys_compare[1], keys_compare[2]), day)
            x1, x2 = x - 0.27, x + 0.27   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = box_outliers(df_now_day[ylabel])[0] + 2.5 * y_unit, 0.1 * y_unit
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
        if len(keys_compare) == 2:
            t = t_test(df_now_day, keys_compare, day)
            x1, x2 = x - 0.2, x + 0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
            y, h = box_outliers(df_now_day[ylabel])[0] + 1 * y_unit, 0.1 * y_unit
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
            plt.text((x1+x2)*.5, y+h*0.2, t, ha='center', va='bottom', color=col, fontsize="large")
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
    plt.savefig(os.path.join(result_dir, file_name_jpg), bbox_inches='tight', dpi=1000)
    plt.close()
    # plt.show()


def line_plot(data: list, x: str, y: str, series: str, title: str, result_dir: str = ".", ylabel: str = "log2(fpkm+1)"):
    """ 画线图    
    """
    data = deepcopy(data)
    for i in range(len(data)):
        data[i][y] = np.log2(np.asarray(data[i][y]) + 1)
    colors = {
        "NanogN70.Oct4": "#C7A609",
        "Nanog.Oct4": "#16557A"
    }
    colors_head = {
        "MEF": "#8B7E75",
        "ESC": "#0F231F",
    }

    file_name_jpg = os.path.join(result_dir, "{}.line.pdf".format(title.split("(", 1)[0]))
    y_min = 10000
    y_max = 0
    for i in range(len(data)):
        # y_min = min(data[i][y], y_min)
        # y_max = max(data[i][y], y_max)
        y_min = min(data[i][y].mean(), y_min)
        y_max = max(data[i][y].mean(), y_max)
    y_gap = (y_max - y_min) * 0.1
    y_min -= y_gap
    y_max += y_gap

    
    f = plt.figure(figsize=(8, 3))
    f.subplots_adjust(hspace=0.5)
    ax = f.add_subplot(1,1,1)

    data_now = list()
    for i in range(len(data)):
        for j in data[i][y]:
            data_now.append({
                x: data[i][x],
                series: data[i][series],
                y: j,
            })
    df = pd.DataFrame(data_now)
    df_uniq = df[df[series] == "uniq"]
    sx = dict()
    for xt, df_temp in df_uniq.groupby(x):
        sx[xt] = (
            np.std(df_temp[y].values) / np.sqrt(len(df_temp[y].values)) * 2,
            df_temp[y].mean(),
        )
    def plot_head_tail(k: str):
        nonlocal ax, sx, colors_head
        ax.scatter(
            [k],
            [sx[k][1]],
            color=colors_head[k],
        )
        ax.plot(
            [k, k],
            [sx[k][1] - sx[k][0], sx[k][1] + sx[k][0]],
            color=colors_head[k],
        )
    
    plot_head_tail("MEF")
    sns.lineplot(
        df[df[series] != "uniq"],
        x=x,
        y=y,
        hue=series,
        palette=colors,
        ax=ax
    )
    plot_head_tail("ESC")


    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(.5, -0.1), ncol=3, title=None, frameon=False,
    )
    plt.savefig(file_name_jpg, dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close()

# %%
samples_dict = dict()
for i, v in zip(
    tuple(map(
        lambda d: d.split("_"),
        df_fpkm.columns
    )),
    df_fpkm.columns
):
    if v == "cluster":
        continue
    if len(i) == 1:
        samples_dict[i[0]] = v
    else:
        if i[0] not in samples_dict:
            samples_dict[i[0]] = dict()
        if i[1] not in samples_dict[i[0]]:
            samples_dict[i[0]][i[1]] = list()
        samples_dict[i[0]][i[1]].append(v)
# %%
for file_single in os.listdir(dir_cluster):
    file_single = os.path.join(dir_cluster, file_single)
    df = pd.read_table(file_single)
    data_line = list()
    for s in samples_dict:
        if isinstance(samples_dict[s], dict):
            for k in samples_dict[s]:
                m = df["{}_{}".format(s, k)].to_list()
                data_line.append({
                    "day": k,
                    "type": s,
                    "value": m
                })
        else:
            m = df[s].to_list()
            data_line.append({
                "day": s,
                "type": "uniq",
                "value": m
            })

    data_line.sort(key=lambda d: sort_func(d["day"]))

    line_plot(
        data_line,
        x="day",
        y="value",
        series="type",
        title="{}(n={})".format(os.path.basename(file_single)[:-4], len(df)),
        result_dir=dir_plot
    )

# %%
def pdf_merge(pdf_dir: str, pdf_end: str, pdf_result: str):
    file_res = os.path.basename(pdf_result)
    pdf_list = list(map(
        lambda d: os.path.join(pdf_dir, d),
        filter(
            lambda d: d.endswith(pdf_end) and (d != file_res),
            os.listdir(pdf_dir)
        )
    ))
    pdf_list.sort(key=lambda d: int(os.path.basename(d)[7:-9]))
    pdf_res = PdfFileMerger()
    for i in pdf_list:
        pdf_res.append(i)
    pdf_res.write(pdf_result)

pdf_merge(
    dir_plot,
    ".line.pdf",
    os.path.join(dir_plot, "..", "fpkm.line.merge.pdf")
)
# %%



