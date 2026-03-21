"""
The base/pretraining dataset is a set of parquet files.
基础/预训练数据集是一组 parquet 文件。
This file contains utilities for:
此文件包含以下工具：
- iterating over the parquet files and yielding documents from it
- 遍历 parquet 文件并从中生成文档
- download the files on demand if they are not on disk
- 如果文件不在磁盘上则按需下载

For details of how the dataset was prepared, see `repackage_data_reference.py`.
有关数据集如何准备的详细信息，请参阅 `repackage_data_reference.py`。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset
# 当前预训练数据集的具体信息

# The URL on the internet where the data is hosted and downloaded from on demand
# 数据托管的互联网 URL，按需下载
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
                 # 最后一个数据分片是 shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
                                                               # 文件名格式
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported
# 这些函数对其他模块有用的工具，可以/应该被导入

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    """ 查看数据目录并返回所有 parquet 文件的完整路径。 """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # 由于从 FinewebEdu-100B 升级到 ClimbMix-400B 而保留的旧版支持代码
    # This code will eventually be deleted.
    # 此代码最终将被删除。
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("  警告：需要升级数据集")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print(f"  找不到：{data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  nanochat 最近从 FinewebEdu-100B 切换到 ClimbMix-400B。")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  截至 2026 年 3 月 4 日执行 `git pull` 的每个人都应该看到此消息。")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print("  要升级到新的 ClimbMix-400B 数据集，请运行这两个命令：")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m nanochat.dataset -n 170     # 下载约 170 个分片，足够 GPT-2 使用，根据需要调整")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print("    python -m scripts.tok_train           # 在新的 ClimbMix 数据上重新训练分词器")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("  暂时回退到您旧的 FinewebEdu-100B 数据集...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        # 尝试回退到旧版数据目录
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    遍历数据集，以底层 row_groups 为批次以提高效率。
    - split can be "train" or "val". the last parquet file will be val.
    - split 可以是 "train" 或 "val"。最后一个 parquet 文件将作为验证集。
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - start/step 用于在 DDP 中跳过行。例如 start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """
    """ 下载单个文件索引，带有退避机制 """

    # Construct the local filepath for this file and skip if it already exists
    # 构造此文件的本地路径，如果已存在则跳过
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    # 构造此文件的远程 URL
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    # 带重试的下载
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            # 先写入临时文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                                                                   # 1MB 块
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            # 将临时文件移动到最终位置
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            # 清理任何部分文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            # 使用指数退避重试几次：2^attempt 秒
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # Prepare the output directory
    # 准备输出目录
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # 这个的工作方式是用户通过 -n 标志指定要下载的训练分片数量。
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    # 此外，验证分片*始终*被下载，并固定为最后一个分片。
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard
                                      # 始终下载验证分片

    # Download the shards
    # 下载分片
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    # 报告结果
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
