#! /bin/bash

PROJECT_PATH=../src/image_classification
EXECUTABLE=${PROJECT_PATH}/image_classification
MODEL_PATH=../model/tflite_image_classification
LABEL=../label/ImageNetLabels.txt
IMAGES_PATH=../image/test_images_ILSVRC2012_val_bmp_resized_224
number_of_threads=1

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# pre_time=(0.6 3.5 0.8 0.9 2.1 4.0 0.4 0.3 0.3 2.6 2.7 3.7 3.8 1.5 1.6 5.1 6.4)
pre_time=(0.6 0.5 4.9 1.0 0.6 1.2 0.9 2.8 1.9 5.4 3.5 0.5 0.4 0.4 0.3 0.4 0.3 3.4 3.5 4.9 5.0 1.9 2.0 6.7 8.2)

index=0

for MODEL in `ls ${MODEL_PATH}`
do  
    if [[ ${MODEL} == "inception_v3_299_fp32.tflite" || ${MODEL} == "inception_v3_299_uint8.tflite" || ${MODEL} == "inception_v4_299_fp32.tflite" || ${MODEL} == "inception_v4_299_uint8.tflite" || ${MODEL} == "inception_resnet_299_fp32.tflite" ]]; then
        IMAGES_PATH=../image/test_images_ILSVRC2012_val_bmp_resized_299
    else
        IMAGES_PATH=../image/test_images_ILSVRC2012_val_bmp_resized_224
    fi

    echo ${MODEL}
    # echo ${IMAGES_PATH}
    ./${EXECUTABLE} -p ${pre_time[${index}]} -d ${pre_time[${index}]} -t 1000 -c 1 -l 2 -b ". --tflite_model ${MODEL_PATH}/${MODEL} --labels ${LABEL} --images_path ${IMAGES_PATH} --threads ${number_of_threads}"
    mkdir ../benchmark_results/image_classification/${MODEL}
    mv timing.csv ../benchmark_results/image_classification/${MODEL}
    mv prediction_output.csv ../benchmark_results/image_classification/${MODEL}
    let index+=1
    
done