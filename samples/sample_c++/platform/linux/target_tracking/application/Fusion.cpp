//
// Created by 13328 on 2023/12/20.
//

#include "Fusion.h"

Fusion::Fusion(std::string &inf_cfg, std::string &inf_eng, std::string &rgb_cfg, std::string &rgb_eng,
               std::string &camera_mat) {
    this->v5lite_inf = new V5lite(inf_cfg);
    this->v5lite_inf->LoadEngine(inf_eng);
    this->v5lite_rgb = new V5lite(rgb_cfg);
    this->v5lite_rgb->LoadEngine(rgb_eng);

    this->tracker = BYTETracker(30, 120);

    YAML::Node config = YAML::LoadFile(camera_mat);
    std::vector<float> dist = config["dist"].as<std::vector<float>>();
    std::vector<float> mt = config["mtx"].as<std::vector<float>>();
    this->crop_dot = config["crop_dot"].as<std::vector<int>>();

    this->label.push_back("car");
    this->label.push_back("person");
    this->colors.resize(2);
    this->colors[0] = cv::Scalar(255, 0, 0);
    this->colors[1] = cv::Scalar(0, 0, 255);

    this->mtx_mat = vec_mat(mt, 3,3);
    this->dist_mat = vec_mat(dist, 1,5);

}


cv::Mat Fusion::vec_mat(std::vector<float> dist, int w, int h) {
    cv::Mat out_result(w,h, CV_32FC1, cv::Scalar(0));
    memcpy(out_result.data, dist.data(), dist.size() * sizeof(float));
    return out_result;
}

cv::Mat Fusion::crop_rgb(cv::Mat &image_rgb, std::vector<int> crop_dot){
    cv::Mat rgb_crop = image_rgb(cv::Rect(crop_dot[0], crop_dot[1], crop_dot[2] - crop_dot[0], crop_dot[3] - crop_dot[1]));
    return rgb_crop;
}

std::vector<V5lite::DetectRes> Fusion::inference_rgb(cv::Mat &frame) {
    std::vector<V5lite::DetectRes> bbox = this->v5lite_rgb->InferenceImage(frame);
    this->NmsDetect(bbox);
    return bbox;
}

std::vector<V5lite::DetectRes> Fusion::inference_inf(cv::Mat &frame) {
    std::vector<V5lite::DetectRes> bbox = this->v5lite_inf->InferenceImage(frame);
    this->NmsDetect(bbox);
    return bbox;
}

float Fusion::IOUCalculate(const V5lite::DetectRes &det_a, const V5lite::DetectRes &det_b) {
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

void Fusion::NmsDetect(std::vector<V5lite::DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const V5lite::DetectRes &left, const V5lite::DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > 0.2)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const V5lite::DetectRes &det)
    { return det.prob == 0; }), detections.end());
}

std::vector<V5lite::DetectRes> Fusion::inference_fusion(cv::Mat &frame_rgb, cv::Mat &frame_inf){
    cv::Mat temp;
    cv::undistort(frame_inf, temp, this->mtx_mat, this->dist_mat);
    cv::resize(temp, frame_inf, frame_rgb.size());
    frame_rgb = this->crop_rgb(frame_rgb, this->crop_dot);

    std::vector<V5lite::DetectRes> bbox_rgb = this->inference_rgb(frame_rgb);
    std::vector<V5lite::DetectRes> bbox_inf = this->inference_inf(frame_inf);
    std::vector<V5lite::DetectRes> rgb_and_inf;
    rgb_and_inf.insert(rgb_and_inf.end(),bbox_rgb.begin(), bbox_rgb.end());
    rgb_and_inf.insert(rgb_and_inf.end(),bbox_inf.begin(), bbox_inf.end());
    NmsDetect(rgb_and_inf);
    return rgb_and_inf;
}

std::vector<STrack> Fusion::track(std::vector<V5lite::DetectRes> &detections){
    std::vector<STrack> output_stracks = this->tracker.update(detections);
    return output_stracks;
}
