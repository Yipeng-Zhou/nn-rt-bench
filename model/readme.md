## TFLite Models for Image Classification
```
vgg16_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012 (index starts from 1)
input tensor size: 224x224x3
model size: 553.4 MB
preprocess: [-128, 127], BGR
source: (conversion from) tf.keras.applications.vgg16.VGG16()
```
```
vgg19_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012 (index starts from 1)
input tensor size: 224x224x3
model size: 574.7 MB
preprocess: [-128, 127], BGR
source: (conversion from) tf.keras.applications.vgg19.VGG19()
```
```

```
```
mobilenet_v1_1.0_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 16.9 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v1_100_224/fp32/1
```
```
mobilenet_v1_1.0_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 4.3 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v1_100_224/uint8/1
```
```
mobilenet_v2_1.0_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 14.0 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v2_100_224/fp32/1
```
```
mobilenet_v2_1.0_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 3.6 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v2_100_224/uint8/1
```
```
mobilenet_v3_large_1.0_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 21.9 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v3_large_100_224/fp32/1
```
```
mobilenet_v3_large_1.0_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 5.6 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/iree/lite-model/mobilenet_v3_large_100_224/uint8/1
```
```

```
```
inception_v1_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 26.6 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/inception_v1/classification/5
```
```
inception_v1_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 6.7 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v1_quant/1/default/1
```
```
inception_v2_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 44.8 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/inception_v2/classification/5
```
```
inception_v2_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 11.3 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v2_quant/1/default/1
```
```
inception_v3_299_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 299x299x3
model size: 95.3 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v3/1/default/1
```
```
inception_v3_299_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 299x299x3
model size: 23.9 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v3_quant/1/default/1
```
```
inception_v4_299_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 299x299x3
model size: 170.7 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v4/1/default/1
```
```
inception_v4_299_uint8.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 299x299x3
model size: 42.9 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/default/1
```
```
inception_resnet_v2_299_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 299x299x3
model size: 121.0 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/tensorflow/lite-model/inception_resnet_v2/1/default/1
```
```

```
```
resnet50_v1_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 102.2 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5
```
```
resnet101_v1_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 178.1 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v1_101/classification/5
```
```
resnet152_v1_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 240.7 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v1_152/classification/5
```
```
resnet50_v2_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 102.3 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5
```
```
resnet101_v2_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 178.4 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5
```
```
resnet152_v2_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012
input tensor size: 224x224x3
model size: 241.1 MB
preprocess: [0, 1], RGB
source: (conversion from) https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5
```
```

```
```
efficientnet_b0_224_fp32.tflite
train dataset: 1,000 objects ILSVRC2012 (index starts from 1)
input tensor size: 224x224x3
model size: 18.6 MB
preprocess: [-1, 1], RGB
source: https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2
```
```
efficientnet_b0_224_uint8.tflite
train dataset: 1,000 objects ILSVRC2012 (index starts from 1)
input tensor size: 224x224x3
model size: 5.4 MB
preprocess: [0, 255], RGB
source: https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/uint8/2
```
<br/>

## TFLite Models for Object Detection
All models for object detection are YOLOv3-tiny with different depths. 
They are built, trained and converted by the repository "https://github.com/Yipeng-Zhou/yolov3-tf2".