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

#ifndef IMAGE_CLASSIFICATION_H_
#define IMAGE_CLASSIFICATION_H_

#include "tensorflow/lite/model.h"
//#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace label_image {

struct Settings {
  TfLiteType input_type = kTfLiteFloat32;
  bool allow_fp16 = false;
  float input_mean = 127.5f;
  float input_std = 127.5f;
  tflite::string model_name = "./mobilenet_v1_1.0_224_quant.tflite";
  tflite::FlatBufferModel* model;
  tflite::string images_path = "./test_images";
  tflite::string labels_file_name = "./ImageNetLabels.txt";
  int number_of_threads = 4;
  int number_of_results = 1;
};

} // namespace label_image
} // namespace tflite

#endif  //IMAGE_CLASSIFICATION_H_
