import pyspark
import pyspark.conf
import pyspark.sql
from pyspark.sql import SparkSession
import pyspark.ml.torch.distributor as distributer
import pandas as pd
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import glob
from petastorm.spark import SparkDatasetConverter, make_spark_converter

# Opens and manages pyspark context, will run on localhost
class SparkManager:
    def __init__(self, name:str, master:str='local', cache_dir:os.PathLike='cache'):
        cache_dir = Path(cache_dir)
        cache_dir = str(cache_dir.resolve())
        
        self.name = name
        self.context = (SparkSession.builder
            .master(master)
            .appName(name)
            .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f"file://{cache_dir}")
            .getOrCreate()
        )
        
    def __enter__(self) -> SparkSession:
        return self.context
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.stop()
       
       
def read_csv(sc: SparkSession, path: os.PathLike):
    path = str(path)
    
    df = sc.read.csv(path=path, inferSchema=True, header=True)
    return df

def read_images(sc: SparkSession, path: os.PathLike):
    path = str(path)
    
    df = sc.read.format("image").load(path)
    return df
    
# Data is too large to put on git
# the archive is unzipped in ./data/
# link here: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time

# To run on local, driver memory limit needs to increase to accomodate file size
# otherwise you might get a java lang out of memory error on the executor

# To do this, go to your spark install conf 
# (hopefully /opt/spark/conf/spark-defaults.conf, if the file isn't there just create it)
# and add/modify the line to "spark.driver.memory 8g" to give the driver node 8GBs memory to work with
# iirc this dataset took ~6GB when running

if __name__ == '__main__':    
    with SparkManager(name='art') as sc:        
        # Example from the artists file
        csv = read_csv(sc, Path('data/artists.csv'))
        csv.show(10)
        csv.printSchema()
                
        # Example from images
        images = read_images(sc, Path('data/images/images/El_Greco/*'))
        
        # Turns struct values into columns
        images = images.select('image.*')
        images.show(10)        
        images.printSchema()
        
        images_train, images_val = images.randomSplit([.9, .1])
        convert_train, convert_val = make_spark_converter(images_train), make_spark_converter(images_val)
        
        trainloader = convert_train.make_torch_dataloader(batch_size=16)
        valloader = convert_val.make_torch_dataloader(batch_size=16)
        