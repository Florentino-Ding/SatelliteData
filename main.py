import argparse
import logging
import pandas as pd

from src.process_webpage import parse_page
from utils.config import CACHE_DIR, LOG_DIR
from utils.download import download_page, download_process_all


logging.basicConfig(
    filename=f"{LOG_DIR}/info.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download satellite data.")

    parser.add_argument("--download", action="store_true", help="Download the data.")
    parser.add_argument("--threads", type=int, default=80, help="Number of threads.")
    parser.add_argument("--min-index", type=int, default=1, help="Minimum index")
    parser.add_argument("--max-index", type=int, default=62878, help="Maximum index")
    parser.add_argument("--max-retries", type=int, default=8, help="Maximum retries")
    parser.add_argument("--cache-path", default=CACHE_DIR, help="Output file name")

    args = parser.parse_args()

    cache_path = args.cache_path

    if args.download:
        pipline = lambda index: (
            index,
            parse_page(download_page(index, args.max_retries)[1]),
        )

        satellite_data = download_process_all(
            pipline, args.threads, args.min_index, args.max_index
        )

        # Save the data to the cache
        satellite_data.to_csv(cache_path, index=False)
    else:
        satellite_data = pd.read_csv(cache_path)


if __name__ == "__main__":
    main()
