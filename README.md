# ZaloAIchallenge2018 - Landmark Identification
**The result should be on top-2 of competition with error=0.62**
## Prerequisite:
+ Caffe installation https://github.com/BVLC/caffe/releases/tag/1.0 
+ Data folder https://challenge.zalo.ai/portal/landmark/data 

## Preparation
### Data
+ Delete corrupted files (0kb)
+ Delete file with wrong format (.png, .bmp but renamed as .jpg in dataset)
+ Delete duplicate files
+ Separate train/val folder with ratio of 92%/8%
+ No data augmentation

## Training
### Models
1) googlenet_reduce (forward time ~ 20ms)
2) resnet_152 (ft~60ms)
+ Training time for model 1 is about 16hours (200000 iters), model 2 is 36 hours (400000 iters)

## Testing
### error:
1) googlenet_reduce: 0.17
2) Resnet_152: 0.62
