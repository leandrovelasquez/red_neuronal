import os
from pathlib import Path
import urllib.request

urls = [
    'https://raw.githubusercontent.com/srjoglekar/PlantVillage-Dataset/master/README.md'
]
out = Path('data/sample_downloads')
out.mkdir(parents=True, exist_ok=True)
for u in urls:
    fname = u.split('/')[-1]
    dest = out / fname
    if not dest.exists():
        urllib.request.urlretrieve(u, dest)
print('downloaded sample files to', out)
