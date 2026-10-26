from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/hsmanik/Student_performance_factors/main/StudentPerformanceFactors.csv"
DEST = Path("data/StudentPerformanceFactors.csv")

DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading dataset to {DEST} ...")
with urlopen(URL, timeout=30) as response:
    DEST.write_bytes(response.read())

print("Done.")
