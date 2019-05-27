# ZaloAIchallenge2018 - Landmark Identification
**Repo có chứa source code cho cuộc thi Zalo Landmark do Zalo tổ chức năm 2018.
Đề bài yêu cầu phân loại các bức ảnh cho sẵn vào 103 địa điểm tương ứng với 103 địa danh nổi tiếng ở Việt Nam như vịnh Hạ Long, chùa Bái Đính, ga Hà Nội... Phương pháp đánh giá kết quả là dựa trên top-3 accuracy**

**Resnet_152-Caffe:**
+ top5: 243/6968  = 0.0349
+ top3: 427/6968  = 0.0613
+ top1: 1004/6968 = 0.1828
+ public_dataset(top3): 0.06429

**Resnet_152-Mxnet:**
+ top5: 125/6968  = 0.01794
+ top3: 164/6968  = 0.0235
+ top1: 485/6968 = 0.0696
+ public_dataset(top3): 0.01846

**Resnext50_32x4d-Mxnet:**
+ top5: 65/6968  = 0.00933
+ top3: 118/6968  = 0.01693
+ top1: 429/6968 = 0.06157
+ public_dataset(top3): 0.01665

## Environments:
+ Ubuntu 16.04
+ Cuda 9.0
+ Cudnn 7
+ OpenCV 3
+ Python 2.7
+ Caffe, MXnet, GluonCV
+ Caffe installation [v1.0](https://github.com/BVLC/caffe/releases/tag/1.0)
+ [Dataset](https://challenge.zalo.ai/portal/landmark/data) 

## EDA (Exploratory Data Analysis)
+ Xóa những file bị corrupt (0kb)
+ Xóa file bị nhầm format (.png, .bmp, .tiff nhưng được format thành .jpg)
+ Có nhiều file giống hệt nhau xuất hiện trong nhiều class khác nhau, mình chọn giải pháp xóa hết các file đó đi  
+ dùng script *preprocessing_data.py* để làm tất cả các bước trên. Sau đó, tập TrainVal sẽ còn lại 86608 imgs
+ Tiếp theo, dùng script *01.create_train_val* để chia thành 2 tập Train và Validation với tỉ lệ 92%/8%
+ Số ảnh tập Train: 79640 imgs
+ Số ảnh tập Validation: 6968 imgs
Dưới đây là distribution của tập TrainVal, qua đó có thể thấy đây là **imbalance data**

![TrainVal1_distribution](https://user-images.githubusercontent.com/17918935/58390109-be05b300-8059-11e9-8edf-15f82e6ca6b2.jpg)

+ Tập Public test có 14356 imgs
Dưới đây là distribution **dự đoán** của tập public test, so sánh với tập TrainVal

![TrainVal_origin compare public_test_top3_prob_distribution](https://user-images.githubusercontent.com/17918935/58390395-5486a400-805b-11e9-9e4c-ec70030f0402.jpg)

+ Tập Private test có 14759 imgs
Dưới đây là distribution **dự đoán** của tập private test, so sánh với tập public test

![public_test_top3_prob_vs_private_test_top3_prob_distribution](https://user-images.githubusercontent.com/17918935/58390472-a16a7a80-805b-11e9-94e7-424d11915d14.jpg)

Qua đó có thể thấy distribution của tập Public test giống với tập TrainVal, và khác khá nhiều tập Private Test. Đó là lý do tại sao accuracy trong LB thì cao nhưng trong Final LB thì lại thấp hơn nhiều.

+ augmentation (Flip vertical, Color Jitter, Lighting)

## Training
+ Use *04.1.train.sh* or "04.2.train_resume.sh* to train from scratch (googlenet_reduce) or fine-tunning model (resnet_152)
### Models
1) googlenet_reduce (forward time ~ 20ms)
2) resnet_152 (ft~60ms) [Caffemodel](https://drive.google.com/drive/u/0/folders/1PYXLmVz0jFPRdQwtm62pkZoUgm5T6Hzq)
+ Training time for model 1 is about 16hours (200000 iters), model 2 is 36 hours (400000 iters)

## Testing
+ Use *eval_val_test.py* to evaluate accuracy on val data and make submission
