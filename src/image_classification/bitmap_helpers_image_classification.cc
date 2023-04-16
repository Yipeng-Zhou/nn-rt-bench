/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "bitmap_helpers_image_classification.h"
#include "log_image_classification.h"

#include <unistd.h>  // NOLINT(build/include_order)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace tflite {
namespace label_image {

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down, Settings* s) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];

          // BGR -> RGB
          if ((s->model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v1_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v1_224_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v2_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v2_224_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v3_299_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v3_299_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v4_299_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_v4_299_uint8.tflite") ||
              (s->model_name == "../model/tflite_image_classification/inception_resnet_299_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet50_v1_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet101_v1_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet152_v1_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet50_v2_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet101_v2_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/resnet152_v2_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/efficientnet_b0_224_fp32.tflite") ||
              (s->model_name == "../model/tflite_image_classification/efficientnet_b0_224_uint8.tflite")) {  
            output[dst_pos] = input[src_pos + 2];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos];
          }
          // BGR -> BGR
          else if ((s->model_name == "../model/tflite_image_classification/vgg16_224_fp32.tflite") ||
                   (s->model_name == "../model/tflite_image_classification/vgg19_224_fp32.tflite")) {
            output[dst_pos] = input[src_pos];
            output[dst_pos + 1] = input[src_pos + 1];
            output[dst_pos + 2] = input[src_pos + 2];
          }

          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          exit(-1);
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels, Settings* s) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << input_bmp_name << " not found";
    exit(-1);
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down, s);
}

}  // namespace label_image
}  // namespace tflite
