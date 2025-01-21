#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "ObjectDetector.h"

DEFINE_string(images_folder, "../images", "图片文件夹路径");
DEFINE_string(model_path, "../../best.torchscript", "模型路径");
DEFINE_string(output_path, "../output", "绘制矩形框后的图片保存路径");
DEFINE_int32(ifdraw, 1, "是否绘制矩形框，1表示绘制，0表示不绘制");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    std::string output_path = FLAGS_output_path;
    std::string images_folder = FLAGS_images_folder;
    std::string model_path = FLAGS_model_path;

    // yolov8s.pt的分类
    // std::vector<std::string> classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    //                                   "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    //                                   "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    //                                   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    //                                   "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    //                                   "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    //                                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    // 自定义best.pt的分类
    std::vector<std::string> classes {"bus","traffic light","traffic sign","person","bike","truck","motor","car","train","rider"};
    ObjectDetector detector(model_path, classes);

    if(FLAGS_ifdraw == 1) {
        int count_image = 0;
        for (const auto & entry : std::filesystem::directory_iterator(images_folder)){
            if (entry.is_regular_file()) {
                count_image++;

                std::string image_path = entry.path().string();
                detector.processImage(image_path);

                // 在图像上绘制矩形框,并且保存到 output 文件夹
                cv::Mat output_image = cv::imread(image_path);
                for (int i = 0; i < detector.keep_boxes_.size(0); i++) {
                    int x1 = static_cast<int>(detector.keep_boxes_[i][0].item<float>());
                    int y1 = static_cast<int>(detector.keep_boxes_[i][1].item<float>());
                    int x2 = static_cast<int>(detector.keep_boxes_[i][2].item<float>());
                    int y2 = static_cast<int>(detector.keep_boxes_[i][3].item<float>());
                    float conf = detector.keep_boxes_[i][4].item<float>();
                    int cls = static_cast<int>(detector.keep_boxes_[i][5].item<float>());
                    const std::string& class_name = detector.class_names_[cls];

                    cv::rectangle(output_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2); // 绿色边框，粗细为 2
                    // 添加类别标签和置信度
                    std::string label = class_name + ": " + std::to_string(conf).substr(0, 4);
                    int baseline;
                    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrg(x1, y1 - 5); // 标签位置在矩形框上方

                    // 绘制标签背景
                    cv::rectangle(output_image, textOrg + cv::Point(0, textSize.height), textOrg + cv::Point(textSize.width, -textSize.height), cv::Scalar(0, 255, 0), cv::FILLED);
                    // 绘制标签文本
                    cv::putText(output_image, label, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1); // 黑色文本

                    // std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << class_name << std::endl;
                }
                // 显示绘制了矩形框的图像
                image_path.compare(0, images_folder.size(), images_folder);
                // 去除文件夹路径，只保留文件名
                std::string file_name = image_path.substr(images_folder.size());
                // std::cout << "file_name: " << file_name << std::endl;
                cv::imwrite(output_path + file_name , output_image);

            }
        }
                
    } else {
        detector.processFolds(images_folder, true);
    }

    return 0;
}
