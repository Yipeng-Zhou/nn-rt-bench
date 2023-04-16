#include "image_classification.h"
#include "bitmap_helpers_image_classification.h"
#include "get_top_n_image_classification.h"
#include "log_image_classification.h"

//#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>       // NOLINT(build/include_order)
//#include <sys/types.h>  // NOLINT(build/include_order)
//#include <sys/uio.h>    // NOLINT(build/include_order)
//#include <unistd.h>     // NOLINT(build/include_order)

//#include <cstdarg>
//#include <cstdio>
//#include <cstdlib>
#include <fstream>
//#include <iomanip>
#include <iostream>
//#include <map>
//#include <memory>
//#include <sstream>
#include <string>
//#include <unordered_set>
#include <vector>
#include <dirent.h>

// Libraries used by tensorflow/lite
//#include "absl/memory/memory.h"
//#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
//#include "tensorflow/lite/examples/label_image/get_top_n.h"
//#include "tensorflow/lite/examples/label_image/log.h"
//#include "tensorflow/lite/kernels/register.h"
//#include "tensorflow/lite/optional_debug_tools.h"
//#include "tensorflow/lite/string_util.h"
//#include "tensorflow/lite/tools/command_line_flags.h"

// Libraries used by rt-bench
#include "logging.h"
#include "periodic_benchmark.h"

// Global Variables
 tflite::label_image::Settings s;
 std::vector<std::string> images_files;
 std::vector<std::string> ground_truth;
 std::vector<tflite::string> labels;
 FILE *prediction_output = NULL;
 int image_index = 0;
 std::unique_ptr<tflite::Interpreter> interpreter;
 std::unique_ptr<tflite::FlatBufferModel> model;
 tflite::ops::builtin::BuiltinOpResolver resolver;
//tflite::ErrorReporter error_reporter;

// Config "Settings" with options from command line
void ConfigSettings(int argc, char** argv) {
  while (true) {
    static struct option long_options[] = {
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"images_path", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"threads", required_argument, nullptr, 't'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"num_results", required_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;
    
    int c;
    c = getopt_long(argc, argv, "b:f:i:l:m:r:s:t",
                    long_options, &option_index);

    /* detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'b':
        s.input_mean = strtod(optarg, nullptr); 
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);
        break;
      case 'i':
        s.images_path = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(
            optarg, nullptr, 10);
        break;
      default:
        exit(-1);
    }
  }
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const tflite::string& file_name,
                            std::vector<tflite::string>* result) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(ERROR) << "Labels file " << file_name << " not found";
    return kTfLiteError;
  }
  result->clear();
  tflite::string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

// Read the file names in a folder and sort
void getAllFiles(std::string path, std::vector<std::string>& files) {
  DIR *pDir;
  struct dirent* ptr;

  if(!(pDir = opendir(path.c_str()))){
    LOG(ERROR) << "Folder doesn't Exist!";
    exit(-1);
  }

  while((ptr = readdir(pDir))!=0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
      files.push_back(path + "/" + ptr->d_name);
    }
  }

  closedir(pDir);
  sort(files.begin(), files.end());
}

// Read ground truth from txt file
void readGroundTruth(std::string path, std::vector<std::string>& ground_truth) {
  std::ifstream fp(path);

  if(!fp.is_open()){
    LOG(ERROR) << "Could not open this file!";
    exit(-1);
  }

  std::string truth;
  while(!fp.eof()){
    std::getline(fp, truth);
    if(truth.length()){
      ground_truth.push_back(truth);  
    }
  }

  fp.close();
}

extern "C" int benchmark_init(int argc, void** argv) {
  // config "Settings" with options from command line
  ConfigSettings(argc, (char**) argv);

  // load model and install interpreter
  if (!s.model_name.c_str()) {
    LOG(ERROR) << "no model file name";
    exit(-1);
  }

  model = tflite::FlatBufferModel::BuildFromFile(s.model_name.c_str());
  if (!model) {
    LOG(ERROR) << "Failed to mmap model " << s.model_name;
    exit(-1);
  }
  s.model = model.get();
  model->error_reporter();

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(ERROR) << "Failed to construct interpreter";
    exit(-1);
  }

  interpreter->SetAllowFp16PrecisionForFp32(s.allow_fp16);

  if (s.number_of_threads != -1) {
    interpreter->SetNumThreads(s.number_of_threads);
  }

  // allocate tensors according to input tensor required by model
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    exit(-1);
  }

  // read all names of image files
  getAllFiles(s.images_path, images_files);

  // read ground truth
  std::string ground_truth_path = "../label/ground_truth_1000.txt";
  readGroundTruth(ground_truth_path, ground_truth);

  // read labels/classes
  if (ReadLabelsFile(s.labels_file_name, &labels) != kTfLiteOk) {
    LOG(ERROR) << "Failed to read labels!";
    exit(-1);
  }
    
  // open file to record the prediction result
  prediction_output = fopen("prediction_output.csv", "w");
  flogf(LOG_LEVEL_FILE, prediction_output, "prediction_flag,confidence\n");

  return 0;
}

extern "C" void benchmark_execution(int argc, void** argv) {
  // determine which image been processing 
  image_index = image_index % images_files.size();

  // read bmp image
  int image_width = 0;
  int image_height = 0;
  int image_channels = 0;
  std::vector<uint8_t> in = tflite::label_image::read_bmp(images_files[image_index], &image_width,
                                     &image_height, &image_channels, &s);

  // get input tensor dimension of the model
  int input = interpreter->inputs()[0];

  // const std::vector<int> inputs = interpreter->inputs();
  // const std::vector<int> outputs = interpreter->outputs();

  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  // LOG(INFO) << wanted_height;
  // LOG(INFO) << wanted_width;
  // LOG(INFO) << wanted_channels;

  // resize input image to adapt the size of input tensor of the model
  s.input_type = interpreter->tensor(input)->type;
  switch (s.input_type) {
    case kTfLiteFloat32:
      // LOG(INFO) << "input_type: " << "kTfLiteFloat32";
      tflite::label_image::resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, &s);
      break;
    case kTfLiteInt8:
      // LOG(INFO) << "input_type: " << "kTfLiteInt8";
      tflite::label_image::resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                     image_height, image_width, image_channels, wanted_height,
                     wanted_width, wanted_channels, &s);
      break;
    case kTfLiteUInt8:
      // LOG(INFO) << "input_type: " << "kTfLiteUInt8";
      tflite::label_image::resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, &s);
      break;
    default:
      LOG(ERROR) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  // inference
  if (interpreter->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Failed to invoke tflite!";
    exit(-1);
  }

  // handle output tensor
  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims; // 1x1001
  // LOG(INFO) << output_dims->size; // 2
  // LOG(INFO) << output_dims->data[0]; // 1
  // LOG(INFO) << output_dims->data[1]; // 1001
  // TfLiteTensor* output_data = interpreter->tensor(output);
  // auto output_value = output_data->data.f;
  // for (int i = 0; i < 1001; i++) {
  //   std::cout << output_value[i] << "\n";
  // }

  /* assume output dims to be something like (1, 1, ... ,size) */
  auto output_size = output_dims->data[output_dims->size - 1]; // 1001
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      tflite::label_image::get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                       s.number_of_results, threshold, &top_results, s.input_type);
      break;
    case kTfLiteInt8:
      tflite::label_image::get_top_n<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
                        output_size, s.number_of_results, threshold,
                        &top_results, s.input_type);
      break;
    case kTfLiteUInt8:
      tflite::label_image::get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, s.number_of_results, threshold,
                         &top_results, s.input_type);
      break;
    default:
      LOG(ERROR) << "cannot handle output type "
                 << interpreter->tensor(output)->type << " yet";
      exit(-1);
  }

  // match output with label
  for (const auto& result : top_results) {
    const float confidence = result.first;
    int index = result.second;
    
    // background's label is 0 or 1
    if ((s.model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/mobilenet_v1_1.0_224_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/mobilenet_v2_1.0_224_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/mobilenet_v3_large_1.0_224_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v1_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v1_224_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v2_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v2_224_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v3_299_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v3_299_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v4_299_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_v4_299_uint8.tflite") ||
        (s.model_name == "../model/tflite_image_classification/inception_resnet_299_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet50_v1_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet101_v1_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet152_v1_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet50_v2_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet101_v2_224_fp32.tflite") ||
        (s.model_name == "../model/tflite_image_classification/resnet152_v2_224_fp32.tflite")) {                                                                         
      index = result.second;
    }
    else if ((s.model_name == "../model/tflite_image_classification/vgg16_224_fp32.tflite") ||
             (s.model_name == "../model/tflite_image_classification/vgg19_224_fp32.tflite") ||
             (s.model_name == "../model/tflite_image_classification/efficientnet_b0_224_fp32.tflite") ||
             (s.model_name == "../model/tflite_image_classification/efficientnet_b0_224_uint8.tflite")) {
      index = result.second + 1;
    }
    std::string prediction;
    int prediction_flag;
    prediction = labels[index].substr(0, labels[index].find(":"));
    if(prediction == ground_truth[image_index]){
      prediction_flag = 1;
    } 
    else{
      prediction_flag = 0;
    }

    flogf(LOG_LEVEL_FILE, prediction_output, "%d,%f\n", prediction_flag, confidence);
    //LOG(INFO) << "image_" << image_index + 1 << ": " << labels[index] << " " << confidence;
  }

  image_index++;

}

extern "C" void benchmark_teardown(int argc, void** argv) {
  // destory the interpreter
  interpreter.reset();

  // close the file that record the prediction results
  fclose(prediction_output);
}


// int main(int argc, char** argv) {
//   if (benchmark_init(argc, argv) == 0) {
//     benchmark_execution();
//     benchmark_teardown();
//   }

//   return 0;
// }