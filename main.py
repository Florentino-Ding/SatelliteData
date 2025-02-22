import argparse

from src.process_webpage import parse_page
from utils.download import download_page, download_process_all

def main() -> None:
    parser = argparse.ArgumentParser(description="Download satellite data.")

    parser.add_argument("-j", type=int, default=80, help="Number of threads.")
    parser.add_argument("-o", default="satellite_data.csv", help="Output file name")
    parser.add_argument("--min_index", type=int, default=1, help="Minimum index")
    parser.add_argument("--max_index", type=int, default=62878, help="Maximum index")
    parser.add_argument("--max_retries", type=int, default=8, help="Maximum retries")

    args = parser.parse_args()

    pipline = lambda index: (
        index,
        parse_page(download_page(index, args.max_retries)[1]),
    )

    satellite_data = download_process_all(pipline, args.j, args.min_index, args.max_index)

    # Save the data to a CSV file
    satellite_data.to_csv(args.o, index=False)


if __name__ == "__main__":
    main()
