# Satellite Data Crawl
## Usage
```bash
python main.py -j <threads> -o <output_file> --min_index <min_index> --max_index <max_index> --max_retries <max_retries>
```
OPTIONS

    -j <int>        Number of threads (default: 80)
    -o <str>        Output file name (default: satellite_data.csv)
    --min_index <int>
                    Minimum index to process (default: 1)
    --max_index <int>
                    Maximum index to process (default: 62878)
    --max_retries <int>
                    Maximum number of retry attempts (default: 8)