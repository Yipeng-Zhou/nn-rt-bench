#include "object_detection.h"
#include "bitmap_helpers_object_detection.h"
#include "log_object_detection.h"

// #include "jpeg_helpers_structured_process_images.h"

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
 std::vector<tflite::string> class_names;
 std::vector<int> original_height;
 std::vector<int> original_width;
 int image_index = 0;
 std::string image_name;
 int num_detections_nms = 0;
 std::vector<int> classes;
 std::vector<float> scores;
 std::vector<int> boxes;
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
// returns a vector of the strings.
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
  
  return kTfLiteOk;
}

// Takes a txt file, and load different columns into different vectors
void getOriginalSize(std::string file_name, std::vector<int>& height, std::vector<int>& width) {
  std::ifstream file(file_name);
  if (!file.is_open()) {
    LOG(ERROR) << "File about images' original size: " << file_name << " not found";
    exit(-1);
  }

  std::vector<std::string> item;
  std::string temp;

  while (std::getline(file, temp)) {
    item.push_back(temp);
  }

  file.close();

  for (auto it = item.begin(); it != item.end(); it++) {
    std::istringstream istr(*it);
    std::string str;
    int count = 0;

    while (istr >> str) {
      if (count == 1) {
        int h = std::atoi(str.c_str());
        height.push_back(h);
      }
      else if (count == 2) {
        int w = std::atoi(str.c_str());
        width.push_back(w);
      }
      count++;
    }
  }
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

  // read class names
  if (ReadLabelsFile(s.labels_file_name, &class_names) != kTfLiteOk) {
    LOG(ERROR) << "Can not read class names!";
    exit(-1);
  }

  // read original images' size
  getOriginalSize("../label/size_images_coco2017_val.txt", original_height, original_width);

  return 0;
}

extern "C" const char *benchmark_log_header() {
  return ",num_detections";
}

extern "C" void benchmark_execution(int argc, void** argv) {
  // determine which image been processing 
  image_index = image_index % images_files.size();

  // read bmp image
  int image_width = 0;
  int image_height = 0;
  int image_channels = 0;
  // LOG(INFO) << "image: " << images_files[image_index];

  int pos_1 = images_files[image_index].find_last_of('/');
  int pos_2 = images_files[image_index].find_last_of('.');
  image_name = images_files[image_index].substr(pos_1 + 1, pos_2 - pos_1 - 1);

  // LOG(INFO) << image_name;

  // std::vector<uint8_t> in = decode_jpeg(images_files[image_index], &image_width, &image_height, &image_channels);

  std::vector<uint8_t> in = tflite::label_image::read_bmp(images_files[image_index], &image_width,
                                     &image_height, &image_channels, &s);

  // LOG(INFO) << "bmp_size: " << in.size();
  // LOG(INFO) << "image_height: " << image_height;
  // LOG(INFO) << "image_width: " << image_width;
  // LOG(INFO) << "image_channels: " << image_channels;

  // LOG(INFO) << ".........................";

  // LOG(INFO) << "decode_bmp: ";
  // for (int i = 0; i < 5; i++){
  //   for (int j = 0; j < 3; j++){
	// 			std::cout << (uint)in[3*i+j] << " ";
  //   }
  //   std::cout << "\n";
	// }

  // LOG(INFO) << ".........................";

  // get input tensor dimension of the model
  int input = interpreter->inputs()[0];

  // const std::vector<int> inputs = interpreter->inputs();
  // const std::vector<int> outputs = interpreter->outputs();

  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  // LOG(INFO) << "wanted_height: " << wanted_height;
  // LOG(INFO) << "wanted_width: " << wanted_width;
  // LOG(INFO) << "wanted_channels: " << wanted_channels;

  // transfer decoded image to input tensor, since image has been resized ahead.
  s.input_type = interpreter->tensor(input)->type;
  switch (s.input_type) {
    case kTfLiteFloat32:
      tflite::label_image::resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, &s);
      break;
    case kTfLiteInt8:
      tflite::label_image::resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                     image_height, image_width, image_channels, wanted_height,
                     wanted_width, wanted_channels, &s);
      break;
    case kTfLiteUInt8:
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

  // LOG(INFO) << ".........................";

  // LOG(INFO) << "input layer: ";
  // input = interpreter->inputs()[0]; // index of input layer
  // LOG(INFO) << input;

  // TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
  // LOG(INFO) << input_dims->size; // input 1x416x416x3

  // auto input_size_0 = input_dims->data[0];
  // LOG(INFO) << input_size_0; // 1
  // auto input_size_1 = input_dims->data[1];
  // LOG(INFO) << input_size_1; // 416
  // auto input_size_2 = input_dims->data[2];
  // LOG(INFO) << input_size_2; // 416
  // auto input_size_3 = input_dims->data[3];
  // LOG(INFO) << input_size_3; // 3

  // TfLiteTensor* resized_image = interpreter->tensor(interpreter->inputs()[0]);
  // auto resized_image_data = resized_image->data.f;
  
  // LOG(INFO) << "resized_bmp: ";
  // for (int i = 0; i < 5; i++){
  //   for (int j = 0; j < 3; j++){
	// 			std::cout << resized_image_data[3*i+j] << " ";
  //   }
  //   std::cout << "\n";
	// }

  // LOG(INFO) << ".........................";

  // LOG(INFO) << "output layer: ";

  // // index of output layers
  // int output_0 = interpreter->outputs()[0];
  // LOG(INFO) << output_0;
  // int output_1 = interpreter->outputs()[1];
  // LOG(INFO) << output_1;
  // int output_2 = interpreter->outputs()[2];
  // LOG(INFO) << output_2;
  // int output_3 = interpreter->outputs()[3];
  // LOG(INFO) << output_3;

  // LOG(INFO) << ".........................";

  // TfLiteIntArray* output_dims_0 = interpreter->tensor(output_0)->dims;
  // LOG(INFO) << output_dims_0->size; // classes 2 (1x100)
  // TfLiteIntArray* output_dims_1 = interpreter->tensor(output_1)->dims;
  // LOG(INFO) << output_dims_1->size; // num_detections 1 (1)
  // TfLiteIntArray* output_dims_2 = interpreter->tensor(output_2)->dims;
  // LOG(INFO) << output_dims_2->size; // boxes 3 (1x100x4)
  // TfLiteIntArray* output_dims_3 = interpreter->tensor(output_3)->dims;
  // LOG(INFO) << output_dims_3->size; // scores(confidence) 2 (1x100)

  // LOG(INFO) << ".........................";

  // auto output_size_0_0 = output_dims_0->data[0];
  // LOG(INFO) << output_size_0_0; // classes 1
  // auto output_size_0_1 = output_dims_0->data[1];
  // LOG(INFO) << output_size_0_1; // classes 100

  // auto output_size_1_0 = output_dims_1->data[0];
  // LOG(INFO) << output_size_1_0; // num_detections 1

  // auto output_size_2_0 = output_dims_2->data[0]; // boxes 1
  // LOG(INFO) << output_size_2_0;
  // auto output_size_2_1 = output_dims_2->data[1]; // boxes 100
  // LOG(INFO) << output_size_2_1;
  // auto output_size_2_2 = output_dims_2->data[2]; // boxes 4
  // LOG(INFO) << output_size_2_2;

  // auto output_size_3_0 = output_dims_3->data[0]; // scores(confidence) 1
  // LOG(INFO) << output_size_3_0;
  // auto output_size_3_1 = output_dims_3->data[1]; // scores(confidence) 100
  // LOG(INFO) << output_size_3_1;

  // LOG(INFO) << ".........................";

  // process outputs
	TfLiteTensor* output_classes = interpreter->tensor(interpreter->outputs()[0]);
	TfLiteTensor* output_num_detections = interpreter->tensor(interpreter->outputs()[1]);
	TfLiteTensor* output_boxes = interpreter->tensor(interpreter->outputs()[2]);
  TfLiteTensor* output_scores = interpreter->tensor(interpreter->outputs()[3]);

  auto output_classes_data = output_classes->data.i32;
  auto output_num_detections_data = output_num_detections->data.i32;
  auto output_boxes_data = output_boxes->data.f;
  auto output_scores_data = output_scores->data.f;

  // reset detection vectors
  num_detections_nms = 0;
  classes.clear();
  scores.clear();
  boxes.clear();

  int num_detections = output_num_detections_data[0];

  for (int i = 0; i < num_detections; i++) {
    if (output_scores_data[i] == 0) {
      break;
    }
    else {
      int cls     = output_classes_data[i*2];
      float score = output_scores_data[i];
      int box_x1  = output_boxes_data[4*i+0] * original_width[image_index];
      int box_y1  = output_boxes_data[4*i+1] * original_height[image_index];
      int box_x2  = output_boxes_data[4*i+2] * original_width[image_index];
      int box_y2  = output_boxes_data[4*i+3] * original_height[image_index];

      classes.push_back(cls);
      scores.push_back(score);
      boxes.push_back(box_x1);
      boxes.push_back(box_y1);
      boxes.push_back(box_x2);
      boxes.push_back(box_y2);
      ++num_detections_nms;
    }
  }

  // std::vector<int> classes;
  // for (int i = 0; i < num_detections; i++){
	// 			int cls = output_classes_data[i*2]; // ???
  //      std::cout << cls << " ";
	// 			classes.push_back(cls);
	// 		}

  // LOG(INFO) << ".........................";

  // std::vector<float> boxes;
  // for (int i = 0; i < num_detections; i++){
  //   for (int j = 0; j < 4; j++){
	// 			float box = output_boxes_data[4*i+j];
  //      std::cout << box << " ";
	// 			boxes.push_back(box);
  //   }
  //   std::cout << "\n";
	// }

  // LOG(INFO) << ".........................";

  // std::vector<float> scores;
  // for (int i = 0; i < num_detections; i++){
	// 			float score = output_scores_data[i];
  //      std::cout << score << " ";
	// 			scores.push_back(score);
	// 		}

  image_index++;

}

extern "C" float benchmark_log_data() {
  // open file to record the prediction result
  std::string record_path = "./results/";
  std::string record_form = ".txt";
  std::string record_name = record_path + image_name + record_form;
  std::ofstream record(record_name.c_str());
  if(!record) {
    LOG(INFO) << "Can not create record!";
    exit(-1);
  }

  // record prediction
  for (int i = 0; i < num_detections_nms; i++) {
    record << class_names[classes[i]] << " " << scores[i] << " " << boxes[4*i+0] << " " << boxes[4*i+1] << " " << boxes[4*i+2] << " " << boxes[4*i+3] << std::endl;
  }

  record.close();

  return (float)num_detections_nms;
}

extern "C" void benchmark_teardown(int argc, void** argv) {
  // destory the interpreter
  interpreter.reset();
}

// int main(int argc, void** argv) {
//   if(benchmark_init(argc, argv) == 0) {
//     while(image_index < 4942) {
//       benchmark_execution(argc, argv);
//     }
//     benchmark_teardown(argc, argv);
//   }

//   return 0;
// }