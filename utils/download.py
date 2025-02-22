import requests
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm

from utils.config import URL, COLS

def download_page(index, max_retries):
    url = f"{URL}{index}"
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download page {index}: {e}")
            retries += 1
            continue

        if response.status_code == 200:
            return index, response.text

    return index, None

# 使用线程池并行下载
def download_process_all(
    download_and_process: callable,
    num_workers: int,
    min_index: int,
    max_index: int,
) -> pd.DataFrame:
    satellite_data =pd.DataFrame(columns=COLS) 

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 使用线程池并行执行下载任务
        results = executor.map(download_and_process, range(min_index, max_index + 1))

        # 将结果保存到字典中
        for index, content in tqdm(results, total=max_index - min_index + 1):
            if content and content[0]:
                satellite_data.loc[len(satellite_data)] = content
            elif not content[0]:
                logging.error(f"Page idx {index} doesn't exist.")
            else:
                logging.error(f"Failed to parse page idx {index}.")

    return satellite_data
