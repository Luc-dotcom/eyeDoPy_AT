# Imports
import pathlib

import albumentations as A
import numpy as np
from torch.utils.data import DataLoader

from datasets import ObjectDetectionDataSet
from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01
from utils import get_filenames_of_path, collate_double, convertLabelsToDict
import torch
from utils import log_model_neptune


# Activate this function only the first time to create labels.py
#convertLabelsToDict(str("labels_txt"), str("heads/target/"))

# hyper-parameters
params = {'BATCH_SIZE': 16,
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': 2, # 5 for all classes
          'SEED': 42,
          'PROJECT': 'Heads',
          'EXPERIMENT': 'heads',
          'MAXEPOCHS': 800,
          'BACKBONE': 'resnet34',
          'FPN': False,
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 657,
          'MAX_SIZE': 876,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }

# root directory
root = pathlib.Path("./heads")

# input and target files
inputs = get_filenames_of_path(root / 'input')
targets = get_filenames_of_path(root / 'target')

inputs.sort()
targets.sort()

# mapping
mapping = {
    '1': 1,
    #'2': 2,
    #'3': 3,
    #'4': 4,
}

# training transformations and augmentations
transforms_training = ComposeDouble([
    Clip(),
    AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# validation transformations
transforms_validation = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# test transformations
transforms_test = ComposeDouble([
    Clip(),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])

# random seed
from pytorch_lightning import seed_everything

seed_everything(params['SEED'])

# training validation test split
inputs_train, inputs_valid, inputs_test = inputs[:700], inputs[701:1000], inputs[1001:2000]
targets_train, targets_valid, targets_test = targets[:700], targets[701:1000], targets[1001:2000]

# dataset training
dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
                                       targets=targets_train,
                                       transform=transforms_training,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=mapping)

# dataset validation
dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
                                       targets=targets_valid,
                                       transform=transforms_validation,
                                       use_cache=True,
                                       convert_to_format=None,
                                       mapping=mapping)

# dataset test
dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                      targets=targets_test,
                                      transform=transforms_test,
                                      use_cache=True,
                                      convert_to_format=None,
                                      mapping=mapping)

# dataloader training
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=params['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=6,
                              collate_fn=collate_double
                              )

# dataloader validation
dataloader_valid = DataLoader(dataset=dataset_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=6,
                              collate_fn=collate_double
                              )

# dataloader test
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=6,
                             collate_fn=collate_double
                              )


# neptune logger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from api_key_neptune import get_api_key

# api_key_neptune.py
#
# def get_api_key():
#     return 'your_super_long_API_token'


api_key = get_api_key()

neptune_logger = NeptuneLogger(
    api_key=api_key,
    project_name=f'luca76485743/{params["PROJECT"]}',
    experiment_name=params['EXPERIMENT'],
    params=params
)

# model init
from faster_RCNN import get_fasterRCNN_resnet

# Aggiunte per caricare checkpoint
#checkpoint = torch.load('./Experiments/heads/HEAD-35/checkpoints/epoch=198-step=6367.ckpt')
#model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE']
                            )
# Caricamento pesi
#model.load_state_dict(model_state_dict)

# lightning init
from faster_RCNN import FasterRCNN_lightning

task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])


# callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

# trainer init
from pytorch_lightning import Trainer

trainer = Trainer(tpu_cores=8,
                  precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                  callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                  default_root_dir="./Experiments",  # where checkpoints are saved to
                  logger=neptune_logger,
                  log_every_n_steps=1,
                  num_sanity_val_steps=0,
                  enable_pl_optimizer=True,  # False seems to be necessary for half precision # FALSE BEFORE
                  )

# %% start training
trainer.max_epochs = params['MAXEPOCHS']
trainer.fit(task,
            train_dataloader=dataloader_train,
            val_dataloaders=dataloader_valid)

# start testing
trainer.test(ckpt_path='best', test_dataloaders=dataloader_test)

# log model
checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
log_model_neptune(checkpoint_path=checkpoint_path,
                  save_directory=pathlib.Path.home(),
                  name='best_model.pt',
                  neptune_logger=neptune_logger)
