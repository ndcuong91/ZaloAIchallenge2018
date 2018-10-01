# ZaloAIchallenge2018 - Landmark Identification
**Error=0.06429 on Public dataset** 
## Prerequisite:
+ Caffe installation [v1.0](https://github.com/BVLC/caffe/releases/tag/1.0)
+ [Dataset](https://challenge.zalo.ai/portal/landmark/data) 

## Preparation
### Data
+ Delete corrupted files (0kb)
+ Delete file with wrong format (.png, .bmp but renamed as .jpg in dataset)
+ Delete duplicate files
+ Use *preprocessing_data.py* to do all of above
+ Use *01.create_train_val* to separate train/val folder with ratio of 92%/8%
+ No data augmentation

## Training
+ Use *04.1.train.sh* or "04.2.train_resume.sh* to train from scratch (googlenet_reduce) or fine-tunning model (resnet_152)
### Models
1) googlenet_reduce (forward time ~ 20ms)
2) resnet_152 (ft~60ms) [Caffemodel](https://drive.google.com/drive/u/0/folders/1PYXLmVz0jFPRdQwtm62pkZoUgm5T6Hzq)
+ Training time for model 1 is about 16hours (200000 iters), model 2 is 36 hours (400000 iters)

## Testing
+ Use *eval_val_test.py* to evaluate accuracy on val data and make submission
### error:
1) googlenet_reduce: 0.17
2) Resnet_152: 0.06429
