# SUPERRESOLUTION

## TODO
* model compare


## DONE
* [Tensorboard](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard)
* make Inference script
* addition metric ssim
* implement psnr to skimage
* inference 3 dataset (set5,set14,BSD100)
* drawing info result
* drawing background
* bmp !
* valid dataset
* matlab code h5 train code

## h5
* input label 96
* stride 14

## model
* model1: basic model __ESPCN__
* model2: basic model __ESPCN__ just adapting depthwise seperable
* model3: baseline model
    * input kernel 3*3 depthwise 5*1 recursive conv
* model4: model3 + residual connection conv 1 to conv 4 
* model5: residual + recursive + invert residual 
* model6: mad model..
* model7: residual light
* model8: residual more light
* model9 : model 23이 이상치 너무 좋음 입력 파라미터줄임
* model10 : model 23이 이상치 너무 좋음 입력 파라미터줄임+ recurcive x2
* model11 : model 23이 이상치 너무 좋음 입력 파라미터줄임+ recurcive x3
* model20: reduce feature map
* model21: bias on 
* model22: input kernal 5x5
* mdoel23: 5x1 conv to 3x1 cov
* model24: 5x1 conv 3x3 conv    
* model25: add feature
* model26 add depth
* model27 3x3 to 1x1
