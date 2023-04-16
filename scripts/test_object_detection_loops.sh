#! /bin/bash

PROJECT_PATH=../src/object_detection
EXECUTABLE=${PROJECT_PATH}/object_detection
MODEL_PATH=../model/tflite_object_detection
LABEL=../label/coco.names
IMAGES_PATH=../image/test_images_coco2017_val_bmp_resized
number_of_threads=1

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

for MODEL in `ls ${MODEL_PATH}`
do  
    #if [ ${MODEL} == "yolov3-tiny.tflite" ]; then
        echo ${MODEL}
        ./${EXECUTABLE} -p 3.5 -d 3.5 -t 5 -c 1 -l 2 -b ". --tflite_model ${MODEL_PATH}/${MODEL} --labels ${LABEL} --images_path ${IMAGES_PATH} --threads ${number_of_threads}"
        mkdir ../benchmark_results/object_detection/${MODEL}
        mv timing.csv ../benchmark_results/object_detection/${MODEL}
        mkdir ../benchmark_results/object_detection/${MODEL}/detection_results
        mv results/* ../benchmark_results/object_detection/${MODEL}/detection_results
    #fi
done
