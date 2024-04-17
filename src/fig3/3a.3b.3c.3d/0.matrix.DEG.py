# %%
import warnings
warnings.filterwarnings('ignore')

import concurrent.futures as fu
import pandas as pd
import subprocess
import threading
import traceback
import logging
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


# %%
dir_base = "/mnt/liudong/task/20240116_HT/result.sort/fig3/3a.3b.3c"
os.makedirs(dir_base, exist_ok=True)
os.chdir(dir_base)
# %%
logger.info("Calc compare pair ...")
file_meta = "/mnt/liudong/task/20230731_HT_RNA_all/result/Correlation.meta.Sall4Oct4.20230628.xlsx"
# %%
df = pd.read_excel(file_meta)
# %%
df = df[df["sample"].str.startswith("Nanog")]
# %%
df["time"] = df["tag"].map(lambda d: d.rsplit("_", 1)[1])
df["type"] = df["tag"].map(lambda d: d.rsplit("_", 1)[0])
# %%
compare_all = list()
time_all = list(df["time"].unique())
time_all.sort(key=lambda d: int(d[1:]))
time_all = tuple(time_all)
type_all = tuple(df["type"].unique())
# %%
def get_sample(time_single: str, type_single: str):
    global df
    return ",".join(df[(df["time"] == time_single) & (df["type"] == type_single)]["sample"].to_list())
# %%
for time_single in time_all:
    compare_all.append(tuple(map(
        lambda d: get_sample(time_single, type_all[d]),
        range(2)
    )))
# %%
for type_single in type_all:
    for i in range(1, len(time_all)):
        compare_all.append(tuple(map(
            lambda d: get_sample(time_all[i-d], type_single),
            range(2)
        )))
# %%
for i in (
    (0, ("MEF1_p2", "MEF2_p2")),
    (-1, ("ESC_OG2_p20", )),
):
    s2 = ",".join(i[1])
    for type_single in type_all:
        s1 = get_sample(time_all[i[0]], type_single)
        compare_all.append((
            s1,
            s2
        ))
# %%
compare_all.append((
    "MEF1_p2,MEF2_p2",
    "ESC_OG2_p20"
))
# %%
samples_all = set(df["sample"].to_list()).union(("MEF1_p2", "MEF2_p2", "ESC_OG2_p20"))
# %%
# task_all = list()
logger.info("Check file ...")
dir_deg = os.path.join(dir_base, "DEG")
os.makedirs(dir_deg, exist_ok=True)
# %%
dir_count = os.path.join(dir_deg, "count")
dir_gfold_ori = os.path.join(dir_deg, "gfold_ori")
for i in [dir_count, dir_gfold_ori]:
    os.makedirs(i, exist_ok=True)
# %%
samples_all_has = list()
dir_datas = [
    "/mnt/liudong/task/20230731_HT_RNA_all/result/20230731.RNA1",
    "/mnt/liudong/task/20230731_HT_RNA_all/result/20230731.RNA2",
    "/mnt/liudong/task/20230731_HT_RNA_all/result/20230731.RNA3.MEF_ESC"
]
for dir_data in dir_datas:
    for root, _, files in os.walk(dir_data):
        for file_single in files:
            if not file_single.endswith(".gfold_count"):
                continue
            exec_cmd("ln -s {} {}".format(
                os.path.join(root, file_single),
                os.path.join(dir_count, file_single)
            ))
            samples_all_has.append(file_single.rsplit(".", 1)[0])
# %%
assert len(samples_all - set(samples_all_has)) == 0, "样本{}的bam或sam文件不存在".format(samples_all - set(samples_all_has))
# %%
def task_done(f: fu.Future):
    global logger
    # 线程异常处理
    e = f.exception()
    if e:
        logger.error(traceback.format_exc())

# gfold
logger.info("gfold ...")
gfold_files = list()
task_all = list()
with fu.ThreadPoolExecutor(max_workers=64) as executor:
    for s1, s2 in compare_all:
        gfold_files.append(os.path.join(dir_gfold_ori, "{}_VS_{}.txt".format(
            s1.replace(",", "_"),
            s2.replace(",", "_")
        )))
        cmd = [
            "cd",
            dir_count,
            "&&",
            "gfold",
            "diff",
            "-s1",
            s2,
            "-s2",
            s1,
            "-suf",
            ".gfold_count",
            "-o",
            gfold_files[-1],
            ">>",
            os.path.join(dir_gfold_ori, "gfold.log"),
            "2>&1"
        ]
        # 提交任务
        task_temp = executor.submit(
            exec_cmd,
            " ".join(cmd)
        )
        # 任务回调
        task_temp.add_done_callback(task_done)
        task_all.append(task_temp)
    fu.wait(task_all)

# %%
logger.info("get degs & filtered fpkm ...")
def get_skiprows(file_name: str):
    skiprows = -1
    with open(file_name) as f:
        for line in f:
            if not line.startswith("#"):
                break
            skiprows += 1
    return skiprows
# %%
genes_all = list()
for file_single in os.listdir(dir_gfold_ori):
    if not file_single.endswith(".txt"):
        continue
    print(file_single)
    file_name = os.path.join(dir_gfold_ori, file_single)
    df = pd.read_table(
        file_name, 
        skiprows=range(get_skiprows(file_name))
    )
    genes_all.extend(df[df["GFOLD(0.01)"].abs() > 0.5]["#GeneSymbol"].to_list())
genes_all = set(genes_all)
# %%
def read_genes(file_gene: str):
    with open(file_gene) as f:
        genes = set(filter(
            lambda d: d != "",
            map(
                lambda dd: dd.strip(),
                f
            )
        ))
    return genes
# %%
def write_genes(genes: set, file_write: str):
    with open(file_write, "w", encoding="UTF-8") as f:
        f.write("\n".join(map(str, genes)))
# %%
def flattern(data):
    data_now = list()
    for data_single in data:
        if isinstance(data_single, (str, int, float)):
            data_now.append(data_single)
        else:
            data_now.extend(flattern(data_single))
    return data_now
# %%
write_genes(genes_all, os.path.join(dir_deg, "DEGs.0.5.txt"))
# %%
















# %%
df = pd.read_table(
    "/mnt/liudong/task/20230731_HT_RNA_all/result/GeneSymbol_fpkm.final.txt",
    index_col=0
)
# %%
df[df.index.isin(genes_all)].to_csv(
    os.path.join(dir_base, "GeneSymbol_fpkm.final.filter_by_DEGs.0.5.txt"),
    sep="\t"
)
# %%
