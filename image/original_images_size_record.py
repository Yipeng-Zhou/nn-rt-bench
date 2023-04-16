import os
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('images_path', '/home/yipeng/Repository/nn-rt-bench/image/test_images_coco2017_val_bmp', 'path to imgaes to be resized')
flags.DEFINE_string('output_path', '/home/yipeng/Repository/nn-rt-bench/label/size_images_coco2017_val.txt', 'saving path of original size of images')

def main(_argv):
    with open(FLAGS.output_path, 'w') as f:
        images = os.listdir(FLAGS.images_path)
        images.sort()
        for image in images:
            image_fullname = os.path.join(FLAGS.images_path, image)
            image_raw = cv2.imread(image_fullname)
            size = image_raw.shape
            f.write(image)
            f.write(' ')
            f.write(str(size[0])) # height
            f.write(' ')
            f.write(str(size[1])) # width
            f.write('\n')
    f.close()
    logging.info('finish')

if __name__ == '__main__':
    app.run(main)