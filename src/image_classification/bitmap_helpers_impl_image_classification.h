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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H_

#include "image_classification.h"
#include "log_image_classification.h"

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace label_image {

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, Settings* s) {
  // int number_of_pixels = image_height * image_width * image_channels;
  // std::unique_ptr<Interpreter> interpreter(new Interpreter);

  // int base_index = 0;

  // // two inputs: input and new_sizes
  // interpreter->AddTensors(2, &base_index);
  // // one output
  // interpreter->AddTensors(1, &base_index);
  // // set input and output tensors
  // interpreter->SetInputs({0, 1});
  // interpreter->SetOutputs({2});

  // // set parameters of tensors
  // TfLiteQuantizationParams quant;
  // interpreter->SetTensorParametersReadWrite(
  //     0, kTfLiteFloat32, "input",
  //     {1, image_height, image_width, image_channels}, quant);
  // interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
  //                                           quant);
  // interpreter->SetTensorParametersReadWrite(
  //     2, kTfLiteFloat32, "output",
  //     {1, wanted_height, wanted_width, wanted_channels}, quant);

  // ops::builtin::BuiltinOpResolver resolver;
  // const TfLiteRegistration* resize_op =
  //     resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
  // auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
  //     malloc(sizeof(TfLiteResizeBilinearParams)));
  // params->align_corners = false;
  // params->half_pixel_centers = false;
  // interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
  //                                    nullptr);

  // interpreter->AllocateTensors();

  // // fill input image
  // // in[] are integers, cannot do memcpy() directly
  // auto input = interpreter->typed_tensor<float>(0);
  // for (int i = 0; i < number_of_pixels; i++) {
  //   input[i] = in[i];
  // }

  // // fill new_sizes
  // interpreter->typed_tensor<int>(1)[0] = wanted_height;
  // interpreter->typed_tensor<int>(1)[1] = wanted_width;

  // interpreter->Invoke();

  // auto output = interpreter->typed_tensor<float>(2);

  if ((image_height != wanted_height) || (image_width != wanted_width) || (image_channels != wanted_channels)) {
    LOG(ERROR) << "image is not resized correctly!";
    exit(-1);
  }

  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    switch (s->input_type) {
      case kTfLiteFloat32:
        out[i] = (in[i] - s->input_mean) / s->input_std; // [-1, 1]

        if ((s->model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_fp32.tflite") ||
            (s->model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_fp32.tflite") ||
            (s->model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_fp32.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_v3_299_fp32.tflite") ||  
            (s->model_name == "../model/tflite_image_classification/inception_v4_299_fp32.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_resnet_299_fp32.tflite") ||
            (s->model_name == "../model/tflite_image_classification/efficientnet_b0_224_fp32.tflite")) {
          out[i] = (in[i] - s->input_mean) / s->input_std; // [-1, 1]
        }
        else if ((s->model_name == "../model/tflite_image_classification/inception_v1_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/inception_v2_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet50_v1_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet101_v1_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet152_v1_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet50_v2_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet101_v2_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/resnet152_v2_224_fp32.tflite")) {
          out[i] = in[i] / 255.0; // [0, 1]
        }
        else if ((s->model_name == "../model/tflite_image_classification/vgg16_224_fp32.tflite") ||
                 (s->model_name == "../model/tflite_image_classification/vgg19_224_fp32.tflite")) {
          out[i] = in[i] - 128; // [-128, 127]
        }

       // out[i] = in[i]; // [0, 255]
  
        break;

      case kTfLiteInt8:
        out[i] = static_cast<int8_t>(in[i] - 128); // [-128, 127]
        break;

      case kTfLiteUInt8:
        out[i] = static_cast<uint8_t>(in[i]); // [0, 255]

        if ((s->model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_v1_224_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_v2_224_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_v3_299_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/inception_v4_299_uint8.tflite") ||
            (s->model_name == "../model/tflite_image_classification/efficientnet_b0_224_uint8.tflite")) {
          out[i] = static_cast<uint8_t>(in[i]); // [0, 255]
        }
        break;
        
      default:
        break;
    }
  }
}

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_BITMAP_HELPERS_IMPL_H_
