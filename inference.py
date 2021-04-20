 
# imports
import ast
import pathlib
 
import neptune
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from api_key_neptune import get_api_key
from datasets import ObjectDetectionDatasetSingle, ObjectDetectionDataSet
from transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, ComposeDouble, FunctionWrapperDouble
from utils import get_filenames_of_path, collate_single
from IPython import get_ipython

 
# parameters
params = {'EXPERIMENT': 'heads',
          'INPUT_DIR': './heads/test', # files to predict
          'PREDICTIONS_PATH': './predictions', # where to save the predictions
          'MODEL_DIR': './epoch=3-step=499.ckpt', # load model from checkpoint
          'DOWNLOAD': False, # wether to download from neptune
          'DOWNLOAD_PATH': './savedModel', # where to save the model
          'OWNER': 'luca76485743',
          'PROJECT': 'Heads',
          'MIN_SIZE': 657,
          'FPN': 'False',
          'MAX_SIZE': 876,
          'IMG_MEAN': '[0.485, 0.456, 0.406]',
          'IMG_STD': '[0.229, 0.224, 0.225]',
          'CLASSES': 2, # 5 for all classes
          'BACKBONE': 'resnet34',
          'ANCHOR_SIZE': '((32, 64, 128, 256, 512),)',
          'ASPECT_RATIOS': '((0.5, 1.0, 2.0),)'
          }
 
# input files
inputs = get_filenames_of_path(pathlib.Path(params['INPUT_DIR']))
inputs.sort()
 
# transformations
transforms = ComposeSingle([
    FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
    FunctionWrapperSingle(normalize_01)
])
 
# create dataset and dataloader
dataset = ObjectDetectionDatasetSingle(inputs=inputs,
                                       transform=transforms,
                                       use_cache=False,
                                       )
 
dataloader_prediction = DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_single)
 
 
# view dataset
from visual import DatasetViewerSingle
from torchvision.models.detection.transform import GeneralizedRCNNTransform
 
transform = GeneralizedRCNNTransform(min_size=int(params['MIN_SIZE']),
                                     max_size=int(params['MAX_SIZE']),
                                     image_mean=ast.literal_eval(params['IMG_MEAN']),
                                     image_std=ast.literal_eval(params['IMG_STD']))
 
 
datasetviewer = DatasetViewerSingle(dataset, rccn_transform=None)
datasetviewer.napari()
 
# download model from neptune or load from checkpoint
if params['DOWNLOAD']:
    download_path = pathlib.Path(params['DOWNLOAD_PATH'])
    model_name = properties['checkpoint_name'] # logged when called log_model_neptune()
    if not (download_path / model_name).is_file():
        experiment.download_artifact(path=model_name, destination_dir=download_path)  # download model
 
    model_state_dict = torch.load(download_path / model_name)
else:
    checkpoint = torch.load(params['MODEL_DIR'])
    model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()
 
# model init
from faster_RCNN import get_fasterRCNN_resnet
model = get_fasterRCNN_resnet(num_classes=int(params['CLASSES']),
                              backbone_name=params['BACKBONE'],
                              anchor_size=ast.literal_eval(params['ANCHOR_SIZE']),
                              aspect_ratios=ast.literal_eval(params['ASPECT_RATIOS']),
                              fpn=ast.literal_eval(params['FPN']),
                              min_size=int(params['MIN_SIZE']),
                              max_size=int(params['MAX_SIZE'])
                              )
 
# load weights
model.load_state_dict(model_state_dict)
 
# inference
model.eval()
for sample in dataloader_prediction:
    x, x_name = sample
    with torch.no_grad():
        pred = model(x)
        pred = {key: value.numpy() for key, value in pred[0].items()}
        name = pathlib.Path(x_name[0])
        torch.save(pred, pathlib.Path(params['PREDICTIONS_PATH']) / name.with_suffix('.pt'))
 
# create prediction dataset
predictions = get_filenames_of_path(pathlib.Path(params['PREDICTIONS_PATH']))
predictions.sort()
 
transforms_prediction = ComposeDouble([
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])
 
dataset_prediction = ObjectDetectionDataSet(inputs=inputs,
                                            targets=predictions,
                                            transform=transforms_prediction,
                                            use_cache=False)
 
# visualize predictions
from visual import DatasetViewer
 
color_mapping = {
    1: 'red',
}
 
datasetviewer_prediction = DatasetViewer(dataset_prediction, color_mapping)
datasetviewer_prediction.napari()

