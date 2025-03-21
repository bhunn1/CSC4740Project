from pyspark.sql import SparkSession
import pandas as pd
import os
from pathlib import Path

# Opens and manages pyspark context, will run on localhost
class SparkManager:
    def __init__(self, name:str, master:str='local'):
        self.name = name
        self.context = SparkSession.builder.master(master).appName(name).getOrCreate()
        
    def __enter__(self) -> SparkSession:
        return self.context
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.stop()
       
 
def read_csv(sc: SparkSession, path: os.PathLike):
    df = sc.read.csv(path=path, inferSchema=True, header=True)
    return df

def read_images(sc: SparkSession, path: os.PathLike):
    df = sc.read.format("image").load(path)
    return df

# Data is too large to put on git
# the archive is unzipped in ./data/
# link here: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

if __name__ == '__main__':
    with SparkManager(name='art') as sc:        
        # Example from the artists file
        csv = read_csv(sc, Path('data/artists.csv'))
        csv.show(10)
        csv.printSchema()
        
        # Example from images
        images = read_images(sc, Path('data/images/images/Raphael/*'))
        
        # Turns struct values into columns
        images = images.select('image.*')
        images.show(3)        
        images.printSchema()
        
        
