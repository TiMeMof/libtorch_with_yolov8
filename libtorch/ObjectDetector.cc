#include "ObjectDetector.h"

using torch::indexing::Slice;
using torch::indexing::None;

// 实现 Timer 类的方法
void ObjectDetector::Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

void ObjectDetector::Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
}

double ObjectDetector::Timer::elapsedMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

double ObjectDetector::Timer::elapsedSeconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
}

// 实现 ObjectDetector 类的构造函数
ObjectDetector::ObjectDetector(const std::string& model_path, const std::vector<std::string>& class_names)
    : model_path_(model_path), class_names_(class_names),
    device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    keep_boxes_(torch::empty(0)) 
{
    LOG(INFO) << "Device: " << device_ ;
    loadModel();
}

// 实现 ObjectDetector 类的 processFolds 方法
void ObjectDetector::processFolds(const std::string& image_folder, bool show_process_time_) {
    Timer timer;
    if (show_process_time_) {
        timer.start();
    }
    int count_image = 0;

    for (const auto& entry : std::filesystem::directory_iterator(image_folder)) {
        if (entry.is_regular_file() && isImageFile(entry.path().string())) {
            count_image++;
            std::string image_path = entry.path().string();
            processImage(image_path);
        }
    }

    if (show_process_time_) {
        timer.stop();
        LOG(INFO) << "Load image and preprocess and NMS: " << timer.elapsedMilliseconds() << " ms" ;
        if (count_image > 0) {
            LOG(INFO) << "per img: " << timer.elapsedMilliseconds() / count_image << " ms" ;
        }
        LOG(INFO) << "total img: " << count_image ;
    }
}

// ... 其他 ObjectDetector 类的方法的实现 ...
float ObjectDetector::generateScale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

float ObjectDetector::letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generateScale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}

torch::Tensor ObjectDetector::xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor ObjectDetector::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

// Reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor ObjectDetector::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
        auto j = order[_j];
        if (suppressed[j] == 1)
            continue;
        auto xx1 = std::max(ix1, x1[j]);
        auto yy1 = std::max(iy1, y1[j]);
        auto xx2 = std::min(ix2, x2[j]);
        auto yy2 = std::min(iy2, y2[j]);

        auto w = std::max(static_cast<float>(0), xx2 - xx1);
        auto h = std::max(static_cast<float>(0), yy2 - yy1);
        auto inter = w * h;
        auto ovr = inter / (iarea + areas[j] - inter);
        if (ovr > iou_threshold)
            suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor ObjectDetector::nonMaxSuppression(torch::Tensor& prediction, float conf_thres /*= 0.25*/, float iou_thres /*= 0.45*/, int max_det /*= 300*/) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({0, 6 + nm}, prediction.device()));
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]});
        auto x_split = x.split({4, nc, nm}, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({Slice(), Slice{5, 6}}) * 7680;
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
        auto scores = x.index({Slice(), 4});
        auto i = nms(boxes, scores, iou_thres);
        i = i.index({Slice(None, max_det)});
        output[xi] = x.index({i});
    }

    return torch::stack(output);
}

torch::Tensor ObjectDetector::clipBoxes(torch::Tensor& boxes, const std::vector<int>& shape) {
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}).clamp(0, shape[1]));
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}).clamp(0, shape[0]));
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}).clamp(0, shape[1]));
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}).clamp(0, shape[0]));
    return boxes;
}

torch::Tensor ObjectDetector::scaleBoxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {

    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}

bool ObjectDetector::isImageFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp"); // 添加你支持的图片格式
}

void ObjectDetector::loadModel() {
    try {
        Timer timer;
        timer.start();
        yolo_model_ = torch::jit::load(model_path_);
        yolo_model_.eval();
        yolo_model_.to(device_, torch::kFloat32);
        timer.stop();
        LOG(INFO) << "Load the model: " << timer.elapsedMilliseconds() << " ms" ;
    } catch (const c10::Error& e) {
        LOG(ERROR) << "Error loading the model: " << e.msg() ;
    }
}

void ObjectDetector::processImage(const std::string& image_path) {
    // Load image and preprocess
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        LOG(ERROR) << "Could not read the image: " << image_path ;
        return;
    }

    cv::Mat input_image;
    letterbox(image, input_image, { 640, 640 });
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    torch::Tensor image_tensor = torch::from_blob(input_image.data, { input_image.rows, input_image.cols, 3 }, torch::kByte).to(device_);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({ 2, 0, 1 });
    image_tensor = image_tensor.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs{ image_tensor };

    // Inference
    torch::Tensor output = yolo_model_.forward(inputs).toTensor().cpu();

    // NMS
    auto keep = nonMaxSuppression(output)[0];
    auto boxes = keep.index({ Slice(), Slice(None, 4) });
    keep.index_put_({ Slice(), Slice(None, 4) }, scaleBoxes({ input_image.rows, input_image.cols }, boxes, { image.rows, image.cols }));
    keep_boxes_ = keep;

}