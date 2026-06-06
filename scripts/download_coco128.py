from pathlib import Path
import os
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent

url = "https://ultralytics.com/assets/coco128.zip"
dest = str(ROOT / "data" / "coco128.zip")

urllib.request.urlretrieve(url, dest)
with zipfile.ZipFile(dest, "r") as z:
    z.extractall(str(ROOT / "data"))
os.remove(dest)
print("Extracted to data/coco128/")
