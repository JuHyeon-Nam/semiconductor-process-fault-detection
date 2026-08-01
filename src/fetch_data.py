from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = {
    "secom.data": f"{BASE_URL}/secom.data",
    "secom_labels.data": f"{BASE_URL}/secom_labels.data",
}


def main() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        target = raw_dir / filename
        if target.exists():
            print(f"exists: {target}")
            continue
        print(f"download: {url}")
        urlretrieve(url, target)
        print(f"saved: {target}")


if __name__ == "__main__":
    main()
