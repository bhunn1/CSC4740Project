### Run this script once to clean the dataset programmatically
from pathlib import Path
import pandas as pd 
import os
import re

if __name__ == '__main__':
    image_dir = Path('data/resized/resized')
    paths = os.listdir(image_dir)
    path_artists = list(map(lambda x: re.search(r'.+_\d', x).group()[:-2], paths))
    
    artists = list(pd.unique(path_artists))
    duplicate = list(filter(lambda x: 'Albrecht' in x, artists))
    
    if len(duplicate) == 2:
        print('Removing duplicate Albrecht paintings from resized')
        destroy = duplicate[0]
        for path in paths:
            if destroy in path:
                os.remove(path=image_dir / Path(path))
