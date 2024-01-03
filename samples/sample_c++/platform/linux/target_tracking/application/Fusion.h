//
// Created by 13328 on 2023/12/20.
//

#ifndef FUSION_FUSION_H
#define FUSION_FUSION_H

#include <iostream>
#include <queue>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include "yaml-cpp/yaml.h"
#include <queue>
#include "BYTETracker.h"

class Fusion {
public:
    V5lite *v5lite_inf;
    V5lite *v5lite_rgb;
    std::vector<std::string> label;
    std::vector<cv::Scalar> colors;
    std::string camera_mat;
    cv::Mat mtx_mat;
    cv::Mat dist_mat;
    std::vector<int> crop_dot;
    BYTETracker tracker;

    Fusion(std::string &inf_cfg, std::string &inf_eng,
           std::string &rgb_cfg, std::string &rgb_eng,
           std::string &camera_mat);

    static cv::Mat vec_mat(std::vector<float> dist, int w, int h);
    static cv::Mat crop_rgb(cv::Mat &image_rgb, std::vector<int> crop_dot);

    std::vector<V5lite::DetectRes> inference_rgb(cv::Mat &frame);
    std::vector<V5lite::DetectRes> inference_inf(cv::Mat &frame);
    std::vector<V5lite::DetectRes> inference_fusion(cv::Mat &frame_rgb, cv::Mat &frame_inf);

    float IOUCalculate(const V5lite::DetectRes &det_a, const V5lite::DetectRes &det_b);

    void NmsDetect(std::vector<V5lite::DetectRes> &detections);

    std::vector<STrack> track(std::vector<V5lite::DetectRes> &detections);

    ~Fusion(){
        delete v5lite_inf;
        v5lite_inf = nullptr;
        delete v5lite_rgb;
        v5lite_rgb = nullptr;
    };
};


#endif //FUSION_FUSION_H
