# ZaloAIchallenge2018 - Landmark Identification
**Repo có chứa source code cho cuộc thi Zalo Landmark do Zalo tổ chức năm 2018.
Đề bài yêu cầu phân loại các bức ảnh cho sẵn vào 103 địa điểm tương ứng với 103 địa danh nổi tiếng ở Việt Nam như vịnh Hạ Long, chùa Bái Đính, ga Hà Nội... Phương pháp đánh giá kết quả là dựa trên top-3 accuracy**

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

## Data Augmentation + TTA (Test Time Augment)
+ Flip vertical
+ Random crop (training), Center crop (testing)
+ Color Jitter
+ Lighting Adjustment

## Training
### Models
+ resnet_152 [Caffemodel](https://drive.google.com/drive/u/0/folders/1PYXLmVz0jFPRdQwtm62pkZoUgm5T6Hzq)
+ ResNext50_32x4D
+ Use "04.2.train_resume.sh* to  fine-tunning model resnet_152 in Caffe

### Model Ensemble
+ 5 TTA with amean
+ Resnet_152+ResNext50_32x4D with gmean

## Prediction
+ Use *eval_val_test.py* to evaluate accuracy on val data and make submission

## Feature Embedding using t-SNE

Mình có visualize lại các cái embedding feature của 1 subset trong Train set với mạng ResNext50_32x4D. Qua đó ta thấy bên cạnh 1 số class được phân tách rất tốt như 49,86,102 thì có một số class có các embedded rất gần nhau như 58, 65 hoặc 64,51. Ngoài ra vẫn có khá nhiều nhiễu trong 1 số cụm chứng tỏ dataset vẫn còn nhiều nhiễu hoặc model phân loại chưa tốt. 

![val_github](https://user-images.githubusercontent.com/17918935/58447786-c7f9e580-812f-11e9-9ded-dbaef280b492.gif)

## Một số ý tưởng trong tương lai
+ Phân tích t-SNE để biết được class nào dễ nhầm với nhau. Từ đó thử thêm các chiến thuật khác phân tách chúng.
+ Training với input size lớn hơn. Hiện tại mình chỉ dùng 224 do phần cứng giới hạn.
+ Phân nhóm lại các class tương tự nhau (nhà thờ, biển, vườn quốc gia...), classify các nhóm tương tự đó rồi mới tách thành các class nhỏ hơn

## Result
**LB: 0.0147 (top 30)**
