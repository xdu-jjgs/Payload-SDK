#include "v5lite.hpp"
#include "yaml-cpp/yaml.h"
#include "common.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


V5lite::V5lite(const std::string &config_file) {
    YAML::Node config = YAML::LoadFile(config_file);
    std::cout << config_file << std::endl;

    BATCH_SIZE = 1;
    NUM_CHANNELS = config["num_channels"].as<int>();
    WIDTH = config["width"].as<int>();
    HEIGHT = config["height"].as<int>();
    LABELS = config["labels"].as<std::vector<std::string>>();
    NUM_CLASSES = LABELS.size();
    COLORS.resize(NUM_CLASSES);
    srand((int) time(nullptr));
    for (cv::Scalar &COLOR : COLORS)
        COLOR = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    CLS_THRESHOLD = config["cls_threshold"].as<float>();
    IOU_THRESHOLD = config["iou_threshold"].as<float>();
    STRIDES = config["strides"].as<std::vector<int>>();
    NUM_ANCHORS = config["num_anchors"].as<std::vector<int>>();
    assert(STRIDES.size() == NUM_ANCHORS.size());
    ANCHORS = config["anchors"].as<std::vector<std::vector<int>>>();
    int index = 0;
    for (const int &stride : STRIDES)
    {
        grids.push_back({NUM_ANCHORS[index], int(HEIGHT / stride), int(WIDTH / stride)});
    }
}

V5lite::~V5lite() = default;

void V5lite::LoadEngine(const std::string &engine_rgb_file) {
    // create and load engine
    std::fstream existEngine;
    //std::fstream existEngine_inf;
    //existEngine_inf.open(engine_inf_file, std::ios::in);
    existEngine.open(engine_rgb_file, std::ios::in);
    readTrtFile(engine_rgb_file, engine);
    //readTrtFile(engine_inf_file, engine_inf);
    assert(engine != nullptr);
}

std::vector<V5lite::DetectRes> V5lite::InferenceImage(cv::Mat &img_rgb) {
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        // std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = bufferSize[1] / sizeof(float);

    auto rects = EngineInference(img_rgb, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    // engine->destroy();

    return rects;
}



std::vector<V5lite::DetectRes> V5lite::EngineInference(cv::Mat &img,
                                                       const int &outSize,
                                                       void **buffers,
                                                       const std::vector<int64_t> &bufferSize,
                                                       cudaStream_t stream) {
    int index = 0;
    int batch_id = 0;
    std::vector<cv::Mat> vec_Mat(1);
    float total_time = 0;
    vec_Mat[0] = img.clone();

    // std::cout << "Processing" << std::endl;
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    // std::cout << "prepareImage" << std::endl;
    //std::vector<float>  curInput = prepareImage(vec_Mat);
    float* curInput = prepareImage(vec_Mat);
    auto t_end_pre = std::chrono::high_resolution_clock::now();
    float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    std::cout << "prepare image take: " << total_pre << " ms." << std::endl;
    total_time += total_pre;
    batch_id = 0;
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    // std::cout << "host2device" << std::endl;
    //cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[0], curInput, bufferSize[0], cudaMemcpyHostToDevice, stream);
    // do inference
    // std::cout << "execute" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    //if(flag == 0)
    //   context_inf->execute(BATCH_SIZE, buffers);
    context->execute(BATCH_SIZE, buffers);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Inference take: " << total_inf << " ms." << std::endl;
    total_time += total_inf;
    // std::cout << "execute success" << std::endl;
    // std::cout << "device2host" << std::endl;
    // std::cout << "post process" << std::endl;
    auto r_start = std::chrono::high_resolution_clock::now();
    auto *out = new float[outSize * BATCH_SIZE];
    cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto boxes = postProcess(vec_Mat, out, outSize);
    auto r_end = std::chrono::high_resolution_clock::now();
    float total_res = std::chrono::duration<float, std::milli>(r_end - r_start).count();
    std::cout << "Post process take: " << total_res << " ms." << std::endl;
    total_time += total_res;
    auto org_img = vec_Mat[0];
    auto rects = boxes[0];
    /*
    for(const auto &rect : rects)
    {
        char t[256];
        sprintf(t, "%.2f", rect.prob);
        std::string name = LABELS[rect.classes] + "-" + t;
        cv::putText(org_img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, COLORS[rect.classes], 2);
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(org_img, rst, COLORS[rect.classes], 2, cv::LINE_8, 0);
    }
    cv::imshow("result", org_img);
    cv::waitKey(1);
    */
    vec_Mat = std::vector<cv::Mat>(BATCH_SIZE);
    delete[] out;
    delete[] curInput;
    std::cout << "Processing time is " << total_time << "ms" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    return rects;
}

//std::vector<float> V5lite::prepareImage(std::vector<cv::Mat> &vec_img) {
float* V5lite::prepareImage(std::vector<cv::Mat> &vec_img) {
    auto t_start_pre1 = std::chrono::high_resolution_clock::now();
    //std::vector<float> result(BATCH_SIZE * WIDTH * HEIGHT * NUM_CHANNELS);
    float* result = new float[BATCH_SIZE * WIDTH * HEIGHT * NUM_CHANNELS];
    //float *data = result.data();
    float* data = result;
    
    
    
    int index = 0;
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = float(WIDTH) / float(src_img.cols) < float(HEIGHT) / float(src_img.rows) ? float(WIDTH) / float(src_img.cols) : float(HEIGHT) / float(src_img.rows);
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(WIDTH, HEIGHT), CV_8UC3);
        cv::Mat rsz_img;
        
        auto t_end_pre1 = std::chrono::high_resolution_clock::now();
    	float total_pre1 = std::chrono::duration<float, std::milli>(t_end_pre1 - t_start_pre1).count();
    	std::cout << "split image00000 take: " << total_pre1 << " ms." << std::endl;
        
        auto t_start_pre = std::chrono::high_resolution_clock::now();
        
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

        //HWC TO CHW
    
        int channelLength = WIDTH * HEIGHT;
        std::vector<cv::Mat> split_img = {
                cv::Mat(HEIGHT, WIDTH, CV_32FC1, data + channelLength * (index + 2)),
                cv::Mat(HEIGHT, WIDTH, CV_32FC1, data + channelLength * (index + 1)),
                cv::Mat(HEIGHT, WIDTH, CV_32FC1, data + channelLength * index)
        };
        index += 3;
        cv::split(flt_img, split_img);
        
        auto t_end_pre = std::chrono::high_resolution_clock::now();
    	float total_pre = std::chrono::duration<float, std::milli>(t_end_pre - t_start_pre).count();
    	std::cout << "split image take: " << total_pre << " ms." << std::endl;
    }
    return result;
}

std::vector<std::vector<V5lite::DetectRes>> V5lite::postProcess(const std::vector<cv::Mat> &vec_Mat, float *output,
                                                                const int &outSize) {
    std::vector<std::vector<DetectRes>> vec_result;
    int index = 0;
    for (const cv::Mat &src_img : vec_Mat)
    {
        std::vector<DetectRes> result;
        float ratio = float(src_img.cols) / float(WIDTH) > float(src_img.rows) / float(HEIGHT)  ? float(src_img.cols) / float(WIDTH) : float(src_img.rows) / float(HEIGHT);
        float *out = output + index * outSize;
        int position = 0;
        for (int n = 0; n < (int)grids.size(); n++)
        {
            for (int c = 0; c < grids[n][0]; c++)
            {
                std::vector<int> anchor = ANCHORS[n * grids[n][0] + c];
                for (int h = 0; h < grids[n][1]; h++)
                    for (int w = 0; w < grids[n][2]; w++)
                    {
                        float *row = out + position * (NUM_CLASSES + 5);
                        // std::cout << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << "," << row[4] << std::endl;
                        position++;
                        DetectRes box;
                        auto max_pos = std::max_element(row + 5, row + NUM_CLASSES + 5);
                        box.prob = row[4] * row[max_pos - row];
                        if (box.prob < CLS_THRESHOLD)
                            continue;
                        box.classes = max_pos - row - 5;
                        box.x = (row[0] * 2 - 0.5 + w) / grids[n][2] * WIDTH * ratio;
                        box.y = (row[1] * 2 - 0.5 + h) / grids[n][1] * HEIGHT * ratio;
                        box.w = pow(row[2] * 2, 2) * anchor[0] * ratio;
                        box.h = pow(row[3] * 2, 2) * anchor[1] * ratio;
                        // box.w = box.w < box.h ? box.w : box.h;
                        // box.h = box.w < box.h ? box.w : box.h;
                        result.push_back(box);
                    }
            }
        }
        NmsDetect(result);
        vec_result.push_back(result);
        index++;
    }
    return vec_result;
}

void V5lite::NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > IOU_THRESHOLD)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

float V5lite::IOUCalculate(const V5lite::DetectRes &det_a, const V5lite::DetectRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
