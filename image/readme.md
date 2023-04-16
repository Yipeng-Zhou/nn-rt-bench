## Test Images for Image Classification
The dataset for image classification contains the first 1000 images from the validation dataset of ILSVRC2012 (50,000 images in total). Because our program does not accept the images with one channel, we replace all 18 images with one channel in the first 1000 images with the last 18 images with three channels (RGB) of the validation dataset of ILSVRC2012. We also provide the ground truth of these 1000 images in the folder "nn-rt-bench/label". The replaced images are listed below:
```
ILSVRC2012_val_00000031.bmp replaced by ILSVRC2012_val_50000.bmp
ILSVRC2012_val_00000107.bmp replaced by ILSVRC2012_val_49999.bmp
ILSVRC2012_val_00000118.bmp replaced by ILSVRC2012_val_49998.bmp
ILSVRC2012_val_00000126.bmp replaced by ILSVRC2012_val_49997.bmp
ILSVRC2012_val_00000141.bmp replaced by ILSVRC2012_val_49996.bmp
ILSVRC2012_val_00000296.bmp replaced by ILSVRC2012_val_49995.bmp
ILSVRC2012_val_00000317.bmp replaced by ILSVRC2012_val_49994.bmp
ILSVRC2012_val_00000377.bmp replaced by ILSVRC2012_val_49993.bmp
ILSVRC2012_val_00000392.bmp replaced by ILSVRC2012_val_49992.bmp
ILSVRC2012_val_00000429.bmp replaced by ILSVRC2012_val_49991.bmp
ILSVRC2012_val_00000532.bmp replaced by ILSVRC2012_val_49990.bmp
ILSVRC2012_val_00000560.bmp replaced by ILSVRC2012_val_49989.bmp
ILSVRC2012_val_00000636.bmp replaced by ILSVRC2012_val_49988.bmp
ILSVRC2012_val_00000704.bmp replaced by ILSVRC2012_val_49987.bmp
ILSVRC2012_val_00000760.bmp replaced by ILSVRC2012_val_49986.bmp
ILSVRC2012_val_00000872.bmp replaced by ILSVRC2012_val_49985.bmp
ILSVRC2012_val_00000889.bmp replaced by ILSVRC2012_val_49984.bmp
ILSVRC2012_val_00000896.bmp replaced by ILSVRC2012_val_49983.bmp
```
we pre-resized all 1000 images in this dataset to 224x224x3 and 299x299x3 according to the needs of models’ input size. You can find them under the folder "test_images_ILSVRC2012_val_bmp_resized_224" and "test_images_ILSVRC2012_val_bmp_resized_299".
<br/>

## Test Images for Object Detection
We use the images from the validation dataset of COCO2017 for object detection and remove the images that don’t contain any objects of the 80 classes used by COCO2017. This processing is done by the repository "https://github.com/Yipeng-Zhou/yolov3-tf2". Furthermore, we only kept the images with three channels (RGB) in this dataset. The removed 10 images with one channel are listed below:
```
000000007888.bmp                      000000024021.bmp
000000061418.bmp                      000000130465.bmp
000000141671.bmp                      000000205289.bmp
000000209222.bmp                      000000274219.bmp
000000353180.bmp                      000000431848.bmp
```
In the end, 4942 images were used for benchamrk in total. You can find them under the folder "test_images_coco2017_val_bmp_resized". The annotations (ground truths) are also filtered accordingly by the repository "https://github.com/Yipeng-Zhou/yolov3-tf2" and "https://github.com/Yipeng-Zhou/mAP". In addition, we also pre-resized all images to 416x416x3. Because the true positions of objects in the annotations (ground truths) are based on the size of the unresized images, we also recorded the size of the original images in order to restore the positions of the objects detection boxes in the inference’s results.