# Benchmarking Real-time Features of Deep Neural Network Inferences
This repository provides the source code of Paper "__Benchmarking Real-time Features of Deep Neural Network Inferences__". 

Based on TensorFlow Lite and RT-Bench, this repository supports benchmarking real-time features of image classification and object detection models. 
All models need to be provided as .tflite format. The real-time features involved include Inference Time, Memory Usage and Accuracy.

The YOLOv3-tiny models with different depths used in this repository are built, trained and converted by the repository "https://github.com/Yipeng-Zhou/yolov3-tf2". The accuracy of these object detection models is calculated separately by the repository "https://github.com/Yipeng-Zhou/mAP" after benchmarking.
<br/>

## Usage
1. Clone this repository and the repository of RT-Bench under the same path. 

   The version of RT-Bench used by our paper has the commit hash "c5ae6e2f55c9ad6ba7034a2ba78b2690053eed95".
   <br/>

2. Go to the folder "./nn-rt-bench/src/image_classification" or "./nn-rt-bench/src/object_detection" according to the models to be benchmarked. 
    <br/>

3. Choose the right "libtensorflowlite.so" according to your platform and whether XNNPACK need to be used. 

   Besides, the selected "libtensorflowlite.so" needs to be added into the folder "./user/lib".
   <br/>

4. Compile the source code on Cortex-A53: `make CORE=CORTEX_A53`

   Compile the source code on other platforms: `make`

   Clear the compilation outputs: `make clean`
   <br/>

5. Go to the folder "./nn-rt-bench/scripts" and start benchmarking.

   For image classification: `sudo bash test_image_classification_loops.sh`

   For object detection: `sudo bash test_object_detection_loops.sh`
<br/>

6. You can find all benchmark results under the folder "./nn-rt-bench/benchmark_results". 

   Besides, the folder "./nn-rt-bench/data_processing" provides the methods to analyse these results.
