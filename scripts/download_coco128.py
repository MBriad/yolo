import os
import urllib.request
import zipfile

url = "https://ultralytics.com/assets/coco128.zip"
dest = "data/coco128.zip"

urllib.request.urlretrieve(url, dest)
with zipfile.ZipFile(dest, "r") as z:
    z.extractall("data/")
os.remove(dest)
print("Extracted to data/coco128/")
