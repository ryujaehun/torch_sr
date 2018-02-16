# subpixel SRCNN

___
## about 
기존의 SRCNN은 학습시에 원본이 이미지를 가우시안필터를 이용하여 축소시킨다음  bicubic interpolation을 사용하여
원본 크기 만큼 확대시킨후 이것을 원본에 가깝게 만드는 필터를 학습시킨다.

그러나 위의 방법은 다음과 같은 문제점이 있다.

1. bicubilc interpolation은 ill-posed problem을해결할  어떠한 추가적인 정보를 제공하지 않는다.
2. 확대된 이미지를 이용하는 만큼 더 많은 연산이 필요로 한다.

이러한 문제를 다음과 같은 아키텍쳐로 해결하고자 한다.
1. cnn을 두 가지 부분으로 나눈다.
    1. 전반부는 Non-linear mapping
    2. 후반부는 deconvolution 과 같은 upsampling을 이용 pixel의 크기를 키운다.
2. 모바일넷에서 제안한 depthwise separable 연산을 이용하여 연산량을 줄인다.

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--model MODELS] [--data DATA]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --data                train data path
  --model               choose a model
```


## Example Usage:

### Train

`python3 main.py --upscale_factor 2 --batchSize 32 --testBatchSize 100 --nEpochs 30 --lr 0.001 --cuda`

### inference


`python3 inference.py --upscale_factor 2 --testBatchSize 100 --weight model_epoch_30 --cuda`

### Super Resolve
`python3 super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png`
