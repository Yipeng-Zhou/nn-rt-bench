PROJECT_PATH=../src/image_classification
EXECUTABLE=${PROJECT_PATH}/image_classification
MODEL=../model/tflite_image_classification/mobilenet_v1_1.0_224_fp32.tflite
LABEL=../label/ImageNetLabels.txt
IMAGES_PATH=../image/test_images_ILSVRC2012_val_bmp_resized_224
number_of_threads=1

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
./${EXECUTABLE} -p 0.8 -d 0.8 -t 100 -c 1 -l 3 -b ". --tflite_model ${MODEL} --labels ${LABEL} --images_path ${IMAGES_PATH} --threads ${number_of_threads}"
