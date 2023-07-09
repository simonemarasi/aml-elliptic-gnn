import requests
from zipfile import ZipFile
import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--path", dest="path", help="Path where save folder", default="/")
command_line_args = parser.parse_args()
data_path = command_line_args.path

# URL = 'https://archive.org/download/elliptic-dataset/d4w7q6.zip'
URL = 'https://www.4sync.com/web/directDownload/fQErng3L/5YfHxh7W.cc4f36f14c07d75ced4bf1fcfa1a0772'

r = requests.get(URL)
os.makedirs(os.path.join(data_path, 'tmp'), exist_ok=True)
with open(os.path.join(data_path, 'tmp/dataset.zip'), 'wb') as f:
    f.write(r.content)
  
os.makedirs(os.path.join(data_path, 'data'), exist_ok=True)
with ZipFile(os.path.join(data_path, 'tmp/dataset.zip'), 'r') as zObject:
    zObject.extractall(path=os.path.join(data_path, 'data'))

shutil.rmtree(os.path.join(data_path, 'tmp'))