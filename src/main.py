import numpy as np
from pyspark.sql import SparkSession
from petastorm.reader import make_batch_reader
from petastorm.pytorch import DataLoader
import os
from pathlib import Path
import torchvision
import torch
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.codecs import NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql.types import FloatType
from model import train_generative


# Opens and manages pyspark context, will run on localhost.
# Essentially just a wrapper class to run pyspark and do config
# without cluttering main
class SparkManager:
    def __init__(self, name:str, master:str='local', cache_dir:os.PathLike='cache'):
        cache_dir = Path(cache_dir)
        cache_dir = str(cache_dir.resolve())
        
        self.name = name
        self.context = (SparkSession.builder
            .master(master)
            .appName(name)
            .getOrCreate()
        )
        self.context.sparkContext.addPyFile('./src/model.py')
    
    # These are python dunders that are executed with the with statement
    # It just handles shutting down the context manager and declutters main
    def __enter__(self) -> SparkSession:
        return self.context
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.stop()
       
# Makes a spark DF from a csv path
def read_csv(sc: SparkSession, path: os.PathLike):
    path = str(path)
    
    df = sc.read.csv(path=path, inferSchema=True, header=True)
    return df

# Makes a spark DF from an images path
def read_images(sc: SparkSession, path: os.PathLike):
    path = str(path)
    
    df = sc.read.format("image").load(path)
    return df
    

def transform_image(image_bytes, dims, channels):
    raw_img = torch.frombuffer(image_bytes, dtype=torch.uint8).reshape((*dims, channels)).permute(2, 0, 1)
    if channels == 1:
        raw_img = raw_img.repeat(3, 1, 1)
        
    img_resized = torchvision.transforms.functional.resize(raw_img, size=(256, 256)).float()
    img = img_resized / 255
    img = img * 2 - 1
    return img.numpy().astype(np.float32)

def train_partition(_):
                    
    file='file:///' + os.getcwd() + '/data/parquet'
    batch_size=32
    load_checkpoint=False
    save_checkpoint=True
    epochs_per_node=100
    
    reader = make_batch_reader(file)
    loader = DataLoader(reader, batch_size)
    train_generative(loader, 
                     epochs=epochs_per_node, 
                     load_checkpoint=load_checkpoint,
                     save_checkpoint=save_checkpoint)
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
        
        output_prepend = 'file:///'
        output_file = os.getcwd() + '/data/parquet'
        output_url = output_prepend + output_file
        
        if not os.path.exists(output_file):
            images = read_images(sc, Path('data/resized/resized/*'))
            
            # Turns struct values into columns
            images = images.select('image.*')
            
            schema = Unischema('sparkData', [
                UnischemaField('images', np.float32, (3, None, None), NdarrayCodec(), False),
            ])
            
            rows_rdd = images.rdd.map(
                lambda x: dict_to_spark_row(schema, {
                    'images': transform_image(
                        x['data'],
                        dims=(x['width'], x['height']),
                        channels=x['nChannels']
                    )})
            )
            
            df = sc.createDataFrame(rows_rdd, schema=schema.as_spark_schema())
            df.printSchema()
            
            with materialize_dataset(sc, output_url, schema):
                df.write.mode('overwrite').parquet(output_url)

        workers = int(sc.sparkContext.getConf().get("spark.executor.instances", "1"))
        processes = sc.sparkContext.parallelize([[] for _ in range(workers)], workers)
        processes.foreach(train_partition)
        