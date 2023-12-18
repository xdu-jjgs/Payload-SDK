#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class V5lite
{
public:
    struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

    V5lite(const std::string &config_file);
    ~V5lite();
    void LoadEngine(const std::string &engine_file);
    std::vector<DetectRes> InferenceImage(cv::Mat &img);

private:
    std::vector<DetectRes> EngineInference(cv::Mat &img, const int &outSize,void **buffers,
                                           const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void NmsDetect(std::vector <DetectRes> &detections);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);

    int BATCH_SIZE;
    int NUM_CHANNELS;
    int WIDTH;
    int HEIGHT;
    std::vector<std::string> LABELS;
    int NUM_CLASSES;
    std::vector<cv::Scalar> COLORS;
    float CLS_THRESHOLD;
    float IOU_THRESHOLD;
    std::vector<int> STRIDES;
    std::vector<int> NUM_ANCHORS;
    std::vector<std::vector<int>> ANCHORS;
    std::vector<std::vector<int>> grids;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
};
