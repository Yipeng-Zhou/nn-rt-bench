PROJECT_PATH=../src/object_detection
EXECUTABLE=${PROJECT_PATH}/object_detection
MODEL=../model/tflite_object_detection/yolov3-tiny.tflite
LABEL=../label/coco.names
IMAGES_PATH=../image/test_images_coco2017_val_bmp_resized
number_of_threads=1

echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
./${EXECUTABLE} -p 2.8 -d 2.8 -t 4942 -c 1 -l 2 -b ". --tflite_model ${MODEL} --labels ${LABEL} --images_path ${IMAGES_PATH} --threads ${number_of_threads}"
