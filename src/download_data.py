"""Download BANKING77 (CC BY 4.0) from PolyAI's GitHub at a pinned commit and verify checksums.

Source: Casanueva et al. (2020), "Efficient Intent Detection with Dual Sentence Encoders".
Usage: python -m src.download_data
"""
import hashlib
import urllib.request

from src.config import DATA_RAW

COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BASE_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}/banking_data"
FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = DATA_RAW / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual}")
        print(f"{name}: OK")


if __name__ == "__main__":
    main()
