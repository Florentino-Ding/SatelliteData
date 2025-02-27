import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
LOG_DIR = os.path.join(PROJECT_ROOT, ".logs")

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

print(PROJECT_ROOT)
