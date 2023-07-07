import requests
from zipfile import ZipFile
import os

URL = 'https://files.catbox.moe/d4w7q6.zip'

r = requests.get(URL)
os.makedirs('tmp', exist_ok=True)
with open('tmp/dataset.zip', 'wb') as f:
    f.write(r.content)
  
os.makedirs('data', exist_ok=True)
with ZipFile("tmp/dataset.zip", 'r') as zObject:
    zObject.extractall(path="data")

os.system('rm -rf tmp')