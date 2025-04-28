import numpy as np
from pyspark.sql import SparkSession
from petastorm.reader import make_reader
from petastorm.pytorch import DataLoader
import os
from pathlib import Path
import torchvision
import torch
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.codecs import NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql.types import FloatType
import model as net
from petastorm.transform import TransformSpec
from torchvision import transforms
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


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
    

# Preprocess the data into normalized format for training
def transform_image(image_bytes, dims, channels):
    raw_img = torch.frombuffer(image_bytes, dtype=torch.uint8).reshape((*dims, channels)).permute(2, 0, 1)
    if channels == 1:
        raw_img = raw_img.expand(3, -1, -1)
    img_resized = torchvision.transforms.functional.resize(raw_img, size=(256, 256)).float()
    img = img_resized / 255
    img = img * 2 - 1
    return img.permute(1, 2, 0).numpy().astype(np.float32)

# Petastorm parquet format to tensor transformation
def load_tensor(row):
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    return {'images': transform(row['images'])}

# Host the torch multiprocess training routine
def distributed_train(load=False, save=True, devices=1):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train_worker, nprocs=devices, args=(load, save, devices))

def train_worker(rank, load, save, devices, prefix='file:///'):
    # Setup distributed process group
    dist.init_process_group("nccl", rank=rank, world_size=devices)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)

    transform_spec = TransformSpec(load_tensor)
    file = prefix + os.getcwd() + '/data/parquet'
    
    # If resuming training, load the state values from the training checkpoint
    if not load:
        model = net.Denoiser()
        optim = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5)
        epochs = 0
    else:
        model, optim, scheduler, _ = net.load_training_checkpoint()
    
    # Using torch multiprocessing
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    epochs = 1
    
    # Mean Squared Error metric
    loss_fn = nn.MSELoss()
    
    # Noise generator
    diffuser = net.ForwardDiffusion()

    # Shard_id is the rank of the current reader and shard_count
    with DataLoader(
        make_reader(
            file,
            num_epochs=epochs,
            transform_spec=transform_spec,
            shuffle_rows=True,
            shard_count=devices,
            cur_shard=rank
        ),
        batch_size=16,
    ) as trainloader:
        net.train_one_epoch(model, optim, loss_fn, scheduler, diffuser, trainloader)

    # Save the model to one device
    if save and rank == 0:
        model.module.save_weights()  # model.module because of DistributedDataParallel
        net.save_training_checkpoint(model.module, optim, scheduler, epochs)

    # Multiprocessing cleanup
    dist.destroy_process_group()
        
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
        
        # Parquet file location
        output_prepend = 'file:///'
        output_file = os.getcwd() + '/data/parquet'
        output_url = output_prepend + output_file
        
        # If the dataset hasn't been processed, create the parquet file
        if not os.path.exists(output_file):
            # Load images from folder
            images = read_images(sc, Path('data/resized/resized/*'))
            
            # Turns struct values into columns
            images = images.select('image.*')
            
            # Define parquet schema as a 3 channel multisized numpy array
            schema = Unischema('sparkData', [
                UnischemaField('images', np.float32, (None, None, 3), NdarrayCodec(), False),
            ])
            
            # Turn the numpy arrays into spark row data
            rows_rdd = images.rdd.map(
                lambda x: dict_to_spark_row(schema, {
                    'images': transform_image(
                        x['data'],  
                        dims=(x['width'], x['height']),
                        channels=x['nChannels']
                    )})
            )
            
            # Create a spark dataframe from the processed data and write it as parquet
            df = sc.createDataFrame(rows_rdd, schema=schema.as_spark_schema())
            
            with materialize_dataset(sc, output_url, schema):
                df.write.mode('overwrite').parquet(output_url)
        
        # Train the model on all avaliable gpu devices in cluster
        workers =int(sc.sparkContext.getConf().get("spark.executor.instances", "1"))
        distributed_train(devices=workers)
        
        