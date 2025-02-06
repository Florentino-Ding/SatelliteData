import argparse
import logging
import requests
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

URL = "https://www.n2yo.com/database/?q="
COLS = [
    "Code",
    "NORAD ID",
    "Int'l Code",
    "Perigee(km)",
    "Apogee(km)",
    "Inclination(deg)",
    "Period(minutes)",
    "Semi major axis(km)",
    "RCS(m^2)",
    "Launch date",
    "Source",
    "Launch site",
    "Decay date",
    "Type",
    "TLE",
]
OUTPUT = pd.DataFrame(columns=COLS)

# Define the regexes
CODE_PATTERN = re.compile(r"<H1>.+<\/H1>")
NORAD_PATTERN = re.compile(r"<B>NORAD ID<\/B>: [0-9]+ ")
INTL_CODE_PATTERN = re.compile(r"<B>Int'l Code<\/B>: [0-9A-Z\-]+ ")
PERIGEE_PATTERN = re.compile(r"<B>Perigee<\/B>: [0-9.]+ km ")
APOGEE_PATTERN = re.compile(r"<B>Apogee<\/B>: [0-9.]+ km ")
INCLINATION_PATTERN = re.compile(r"<B>Inclination<\/B>: [0-9.]+ &deg ")
PERIOD_PATTERN = re.compile(r"<B>Period<\/B>: [0-9.]+ minutes ")
SEMI_MAJOR_AXIS_PATTERN = re.compile(r"<B>Semi major axis<\/B>: [0-9]+ km ")
RCS_PATTERN = re.compile(r"<B>RCS<\/B>: [a-zA-Z0-9\ ]+ ")
LAUNCH_DATE_PATTERN = re.compile(
    r'<B>Launch date<\/B>: <a href="[a-zA-Z0-9\/\?=\&]+">[0-9a-zA-Z ,]+<\/a>'
)
SOURCE_PATTERN = re.compile(r"<B>Source<\/B>: [a-zA-Z \'\(\)]+")
LAUNCH_SITE_PATTERN = re.compile(r"<B>Launch site<\/B>: [a-zA-Z \(\)]+")
DECAY_DATE_PATTERN = re.compile(r"<B>Decay date<\/B>: [\d-]+")
TYPE_NAME_PATTERN = re.compile(r"<b>Note: This is [a ]?.+<\/b>")
TLE_PATTERN = re.compile(r"<pre>[\w .\n\-]+<\/pre>")


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


def parse_page(content: str) -> tuple:
    code = CODE_PATTERN.search(content)
    norad = NORAD_PATTERN.search(content)
    intl_code = INTL_CODE_PATTERN.search(content)
    perigee = PERIGEE_PATTERN.search(content)
    apogee = APOGEE_PATTERN.search(content)
    inclination = INCLINATION_PATTERN.search(content)
    period = PERIOD_PATTERN.search(content)
    semi_major_axis = SEMI_MAJOR_AXIS_PATTERN.search(content)
    rcs = RCS_PATTERN.search(content)
    launch_date = LAUNCH_DATE_PATTERN.search(content)
    source = SOURCE_PATTERN.search(content)
    launch_site = LAUNCH_SITE_PATTERN.search(content)
    decay_date = DECAY_DATE_PATTERN.search(content)
    type_name = TYPE_NAME_PATTERN.search(content)
    tle = TLE_PATTERN.search(content)

    # Process the regex results
    code = code.group(0).split("<H1>")[1].split("</H1>")[0] if code else None
    norad = norad.group(0).split(": ")[1] if norad else None
    intl_code = intl_code.group(0).split(": ")[1] if intl_code else None
    perigee = perigee.group(0).split(": ")[1].split(" ")[0] if perigee else None
    apogee = apogee.group(0).split(": ")[1].split(" ")[0] if apogee else None
    inclination = (
        inclination.group(0).split(": ")[1].split(" ")[0] if inclination else None
    )
    period = period.group(0).split(": ")[1].split(" ")[0] if period else None
    semi_major_axis = (
        semi_major_axis.group(0).split(": ")[1].split(" ")[0]
        if semi_major_axis
        else None
    )
    rcs = rcs.group(0).split(": ")[1] if rcs else None
    launch_date = (
        launch_date.group(0).split(": ")[1].split(">")[1].split("<")[0]
        if launch_date
        else None
    )
    source = source.group(0).split(": ")[1] if source else None
    launch_site = launch_site.group(0).split(": ")[1] if launch_site else None
    decay_date = decay_date.group(0).split(": ")[1] if decay_date else "ACTIVE"
    type_name = (
        type_name.group(0).split("This is ")[1].split("</b>")[0].split("a ")[-1]
        if type_name
        else "SATELLITE"
    )
    tle = tle.group(0).split("<pre>")[1].split("</pre>")[0] if tle else None

    return (
        norad,
        code,
        intl_code,
        perigee,
        apogee,
        inclination,
        period,
        semi_major_axis,
        rcs,
        launch_date,
        source,
        launch_site,
        decay_date,
        type_name,
        tle,
    )


# 使用线程池并行下载
def download_process_all(
    download_and_process: callable,
    num_workers: int,
    min_index: int,
    max_index: int,
) -> None:
    global OUTPUT

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 使用线程池并行执行下载任务
        results = executor.map(download_and_process, range(min_index, max_index + 1))

        # 将结果保存到字典中
        for index, content in tqdm(results, total=max_index - min_index + 1):
            if content and content[0]:
                OUTPUT.loc[len(OUTPUT)] = content
            elif not content[0]:
                logging.error(f"Page idx {index} doesn't exist.")
            else:
                logging.error(f"Failed to parse page idx {index}.")


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

    download_process_all(pipline, args.j, args.min_index, args.max_index)

    # Save the data to a CSV file
    OUTPUT.to_csv(args.o, index=False)


if __name__ == "__main__":
    main()
