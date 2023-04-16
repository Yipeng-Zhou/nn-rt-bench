import os
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('images_path', '/home/yipeng/Repository/nn-rt-bench/image/test_images_ILSVRC2012_val_bmp', 'path to imgaes to be resized')
flags.DEFINE_string('outputs_path', '/home/yipeng/Repository/nn-rt-bench/image/test_images_ILSVRC2012_val_bmp_resized_299', 'saving path of resized images')

def main(_argv):
    for image in os.listdir(FLAGS.images_path):
        logging.info(image)
        image_fullname = os.path.join(FLAGS.images_path, image)
        image_raw = cv2.imread(image_fullname)
        image_resize = cv2.resize(image_raw, (299,299), interpolation=cv2.INTER_LINEAR)
        saving = os.path.join(FLAGS.outputs_path, image)
        cv2.imwrite(saving, image_resize)

if __name__ == '__main__':
    app.run(main)