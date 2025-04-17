import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
import pyspark.ml.torch.distributor as distributer
import os
from pathlib import Path
from petastorm.spark import SparkDatasetConverter, make_spark_converter
import torchvision

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
            .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, f"file://{cache_dir}")
            .getOrCreate()
        )
    
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
    
def spark_to_dataloader(sc: SparkSession, df: pyspark.sql.DataFrame, 
                        test_size = .1, train_batches=16, test_batches=16):
    num_workers = sc.getConf().get("spark.executor.instances")

    train_size = 1 - test_size
    train, test = df.randomSplit([train_size, test_size])
    
    train.repartition(num_workers)
    test.repartition(num_workers)
    
    storm_train, storm_test = make_spark_converter(train), make_spark_converter(test)
    trainloader = storm_train.make_torch_dataloader(batch_size=train_batches)
    testloader = storm_test.make_torch_dataloader(batch_size=test_batches)
    
    return trainloader, testloader

def transform_images(image_bytes):
    raw_img = torchvision.io.decode_image(image_bytes, mode='RGB')
    img_resized = torchvision.transforms.functional.resize(raw_img, size=(256, 256)).float()
    img = img_resized / 255
    img = img * 2 - 1
    
    return img
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
        images = read_images(sc, Path('data/resized/resized/*'))
        
        # Turns struct values into columns
        images = images.select('image.*')
        images = images['data'].map(lambda x: Image.open)
        images.show(10)        
        images.printSchema()
        
        # Train/test split for images
        #trainloader, valloader = spark_to_dataloader(sc, images)
        
        