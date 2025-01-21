#ifndef ObjectDetector_H
#define ObjectDetector_H

// C++ 标准库
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <filesystem>
// OpenCV 库
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
// libtorch 库
#include <torch/torch.h>
#include <torch/script.h>
// gflags 和 glog 库
#include <gflags/gflags.h>
#include <glog/logging.h>

class ObjectDetector {
public:
    ObjectDetector(const std::string& model_path, const std::vector<std::string>& class_names);

    // 处理一个文件夹下的所有图片
    void processFolds(const std::string& image_folder, bool show_process_time_ = false);

    // 处理一张图片
    void processImage(const std::string& image_path);

    class Timer {
    public:
        void start();
        void stop();
        double elapsedMilliseconds() const;
        double elapsedSeconds() const;
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    };

private:
    float generateScale(cv::Mat& image, const std::vector<int>& target_size);
    float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
    torch::Tensor xyxy2xywh(const torch::Tensor& x);
    torch::Tensor xywh2xyxy(const torch::Tensor& x);
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);
    torch::Tensor nonMaxSuppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);
    torch::Tensor clipBoxes(torch::Tensor& boxes, const std::vector<int>& shape);
    torch::Tensor scaleBoxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
    bool isImageFile(const std::string& filename);
    void loadModel();
    
public:
    std::vector<std::string> class_names_;
    torch::Device device_;
    torch::Tensor keep_boxes_;
private:
    torch::jit::script::Module yolo_model_;
    std::string model_path_;
};

#endif // ObjectDetector_H