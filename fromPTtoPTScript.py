# Imports

import torch

save_path = './'

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


# model init
from faster_RCNN import get_fasterRCNN_resnet

# Aggiunte per caricare checkpoint
checkpoint = torch.load('./epoch=3-step=499.ckpt', map_location=torch.device('cpu'))
model_state_dict = checkpoint['hyper_parameters']['model'].state_dict()

model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                              backbone_name=params['BACKBONE'],
                              anchor_size=params['ANCHOR_SIZE'],
                              aspect_ratios=params['ASPECT_RATIOS'],
                              fpn=params['FPN'],
                              min_size=params['MIN_SIZE'],
                              max_size=params['MAX_SIZE']
                            )
# Caricamento pesi
model.load_state_dict(model_state_dict)

model.eval()

#input_tensor = torch.rand(1, 3, 576, 768)
# conversion
script_model = torch.jit.script(model)

# OPTIONAL: 
#uncomment to switch on the optimizations for mobile
#script_model = optimize_for_mobile(script_model)
#uncomment for Quantization (int-8) Dynamic
#net = torch.quantization.quantize_dynamic(net, {torch.nn.Linear,torch.nn.Sequential,net.modules}, dtype=torch.qint8)  # the target dtype for quantized weights

# saving
script_model.save(save_path + "ObjDetresnet34.pt")


#SANITY CHECK
script_model.eval()
#image = Image.open("testImage.JPG")
#image = image.resize((768,576))

#image = np.transpose(image, (2, 0, 1)) # Comment in case of normalization
        
#normalize image
#image = transforms.functional.to_tensor(image)
#image = transforms.functional.normalize(image, mean = [120.56737612047593, 119.16664454573734, 113.84554638827127], std=[66.32028460114392, 65.09469952002551, 65.67726614496246])

#image = torch.FloatTensor([image]) # Comment in case of normalization

