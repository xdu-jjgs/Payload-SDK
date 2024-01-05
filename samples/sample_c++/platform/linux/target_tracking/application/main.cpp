/**
 ********************************************************************
 * @file    main.cpp
 * @brief
 *
 * @copyright (c) 2021 DJI. All rights reserved.
 *
 * All information contained herein is, and remains, the property of DJI.
 * The intellectual and technical concepts contained herein are proprietary
 * to DJI and may be covered by U.S. and foreign patents, patents in process,
 * and protected by trade secret or copyright law.  Dissemination of this
 * information, including but not limited to data and other proprietary
 * material(s) incorporated within the information, in any form, is strictly
 * prohibited without the express written consent of DJI.
 *
 * If you receive this source code without DJI’s authorization, you may not
 * further disseminate the information, and you must immediately remove the
 * source code and notify DJI of its removal. DJI reserves the right to pursue
 * legal actions against you for any loss(es) or damage(s) caused by your
 * failure to do so.
 *
 *********************************************************************
 */

/* Includes ------------------------------------------------------------------*/
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <liveview/dji_camera_stream_decoder.hpp>
#include <liveview/test_liveview.hpp>
#include <dji_low_speed_data_channel.h>
#include <dji_high_speed_data_channel.h>
#include <dji_mop_channel.h>
#include <dji_gimbal_manager.h>
#include <dji_logger.h>

#include "application.hpp"
#include "Fusion.h"

/* Private constants ---------------------------------------------------------*/
#define FPV_CAM          false
#define MAIN_CAM         true
#define VICE_CAM         false
#define TOP_CAM          false

#define TASK_STACK_SIZE 2048

#define PIPELINE_IS_RELIABLE false
#define PIPELINE_ID 49155

#define SEND_BUFFER_SIZE 10240
#define RECV_BUFFER_SIZE 10240

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/
bool connected = false;
int width, height;
std::mutex mtx;
int fd_cpp2py, fd_py2cpp;
int output_size;
uint8_t* send_buffer;
uint8_t* recv_buffer;
uint32_t send_size;
uint32_t recv_size;
// Fusion* fusion;
Fusion* fusion_inf;
Fusion* fusion_rgb;

static T_DjiMopChannelHandle s_MopChannelNormalHandle;
static T_DjiMopChannelHandle s_MopChannelNormalOutHandle;

static T_DjiTaskHandle s_ConnectThread;
static T_DjiTaskHandle s_SendThread;
static T_DjiTaskHandle s_RecvOutputThread;

static E_DjiMountPosition s_MountPosition = E_DjiMountPosition('1' - '0');
static E_DjiGimbalMode s_GimbalMode = DJI_GIMBAL_MODE_FREE;

/* Private functions declaration ---------------------------------------------*/
static void ShowRgbImageCallback(CameraRGBImage img, void*);
static void* RecvOutputTask(void*);
static void* ConnectTask(void*);
static T_DjiReturnCode RecvCallback(const uint8_t *data, uint16_t len);
static void* SendTask(void*);

static void bboxs2img(std::vector<STrack>& trk_rects, cv::Mat& img);
static std::string bboxs2json(std::vector<STrack>& trk_rects);

/* Exported functions definition ---------------------------------------------*/
int main(int argc, char **argv)
{
    Application application(argc, argv);

    std::string inf_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/cfgs/inf.yaml";
    std::string inf_eng = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/ckpts/yolov5s_hb2_inf_640.trt";
    std::string rgb_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/cfgs/rgb.yaml";
    std::string rgb_eng = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/ckpts/yolov5m_xd_rgb_1280.trt";
    std::string camera_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/configs/camera.yaml";
    fusion_inf = new Fusion(inf_cfg, inf_eng, rgb_cfg, rgb_eng, camera_cfg);
    fusion_rgb = new Fusion(inf_cfg, inf_eng, rgb_cfg, rgb_eng, camera_cfg);

    auto* sample = new LiveviewSample();
    if (FPV_CAM) {
      sample->StartFpvCameraStream(&ShowRgbImageCallback, NULL);
    } else if (MAIN_CAM) {
      sample->StartMainCameraStream(&ShowRgbImageCallback, NULL);
    } else if (VICE_CAM) {
      sample->StartViceCameraStream(&ShowRgbImageCallback, NULL);
    } else if (TOP_CAM) {
      sample->StartTopCameraStream(&ShowRgbImageCallback, NULL);
    }

    T_DjiOsalHandler *handler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode retcode;

    send_buffer = static_cast<uint8_t*>(handler->Malloc(SEND_BUFFER_SIZE));
    if (send_buffer == NULL) {
        USER_LOG_ERROR("malloc send buffer error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("malloc send buffer success");
    }

    retcode = handler->TaskCreate("connect", ConnectTask, TASK_STACK_SIZE, NULL, &s_ConnectThread);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("connect task create error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("connect task create success");
    }

    // recv
    /*
    E_DjiChannelAddress address = DJI_CHANNEL_ADDRESS_MASTER_RC_APP;
    retcode = DjiLowSpeedDataChannel_RegRecvDataCallback(address, RecvCallback);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("register receive callback error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("register receive callback success");
    }
    */

    // send
    /*
    retcode = handler->TaskCreate("send", SendTask, TASK_STACK_SIZE, NULL, &s_SendThread);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("send task create error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("send task create success");
    }
    */

    // gimbal
    retcode = DjiGimbalManager_Init();
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("init gimbal manager error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("init gimbal manager success");
    }
    retcode = DjiGimbalManager_SetMode(s_MountPosition, s_GimbalMode);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("set gimbal mode error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("set gimbal mode success");
    }
    retcode = DjiGimbalManager_Rotate(s_MountPosition, (T_DjiGimbalManagerRotation) {DJI_GIMBAL_ROTATION_MODE_RELATIVE_ANGLE, 5, 0, 0, 2.0});
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("rotate gimbal error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("rotate gimbal success");
    }

    handler->TaskSleepMs(99999999);

    if (FPV_CAM) {
      sample->StopFpvCameraStream();
    } else if (MAIN_CAM) {
      sample->StopMainCameraStream();
    } else if (VICE_CAM) {
      sample->StopViceCameraStream();
    } else if (TOP_CAM) {
      sample->StopTopCameraStream();
    }
    return 0;
}

static void bboxs2img(std::vector<STrack>& trk_rects, cv::Mat& img) {
    std::unordered_map<int, std::string> cls2label = {{0, "car"}, {1, "people"}};
    for (const auto& trk : trk_rects) {
        const std::vector<float>& tlwh = trk.tlwh;
        float x = tlwh[0];
        float y = tlwh[1];
        float w = tlwh[2];
        float h = tlwh[3];

        cv::putText(img, cls2label[trk.cls] + "-" + std::to_string(trk.track_id), cv::Point(x, y - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::Rect rst(x, y, w, h);
        cv::rectangle(img, rst, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
}

static std::string bboxs2json(std::vector<STrack>& trk_rects) {
    std::stringstream ss;
    ss << "{\"size\":[1920, 1080]}";
    if (!trk_rects.empty()) {
        ss << "{\"bboxs\":[";
    }
    for (int i = 0; i < trk_rects.size(); i++) {
        const auto& trk = trk_rects[i];

        const std::vector<float>& tlwh = trk.tlwh;
        float x = tlwh[0];
        float y = tlwh[1];
        float w = tlwh[2];
        float h = tlwh[3];
        float cx = x + w/2;
        float cy = y + h/2;

        ss << "[";
	ss << std::to_string((int)cx) << ",";
	ss << std::to_string((int)cy) << ",";
	ss << std::to_string((int)w) << ",";
	ss << std::to_string((int)h) << ",";
	ss << std::to_string((int)trk.cls) << ",";
	ss << std::to_string((int)trk.track_id) << "]";
        if (i != trk_rects.size() - 1) ss << ",";
    }
    if (!trk_rects.empty()) {
        ss << "]}";
    }
    ss << "\0";
    return ss.str();
}

static void* RecvOutputTask(void*) {
    while (1) {
        output_size = read(fd_py2cpp, send_buffer, SEND_BUFFER_SIZE);
        // USER_LOG_INFO("recv from py2cpp %s", &send_buffer[0]);
    }
    return NULL;
}

/* Private functions definition-----------------------------------------------*/
// cv::VideoCapture* cap = new cv::VideoCapture("/home/dji/Desktop/input.mp4");
static void ShowRgbImageCallback(CameraRGBImage img, void*)
{
    T_DjiOsalHandler *handler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode retcode;

    if (!mtx.try_lock()) return;

    cv::Mat mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width * 3);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    /*
    cap->read(mat);
    if (mat.size().width == 0) {
        cap = new cv::VideoCapture("/home/dji/Desktop/input.mp4");
        mtx.unlock();
        return;
    }
    */
    cv::Mat inf = mat(cv::Rect(0, 160, 960, 760));
    cv::Mat rgb = mat(cv::Rect(960, 160, 960, 760));
    width = mat.size().width;
    height = mat.size().height;
    USER_LOG_INFO("w=%d, h=%d", width, height);

    auto t0_det_inf = std::chrono::high_resolution_clock::now();
    auto rects_det_inf = fusion_inf->inference_inf(inf);
    auto t1_det_inf = std::chrono::high_resolution_clock::now();
    float t_det_inf = std::chrono::duration<float, std::milli>(t1_det_inf - t0_det_inf).count();
    USER_LOG_INFO("detect (inf): %f ms", t_det_inf);
    USER_LOG_INFO("number of det boxes (inf): %d", rects_det_inf.size());

    auto t0_trk_inf = std::chrono::high_resolution_clock::now();
    auto rects_trk_inf = fusion_inf->track(det_rects_inf);
    auto t1_trk_inf = std::chrono::high_resolution_clock::now();
    float t_trk_inf = std::chrono::duration<float, std::milli>(t1_trk_inf - t0_trk_inf).count();
    USER_LOG_INFO("track (inf): %f ms", t_trk_inf);
    USER_LOG_INFO("number of trk boxes (inf): %d", rects_trk_inf.size());

    auto t0_det_rgb = std::chrono::high_resolution_clock::now();
    auto rects_det_rgb = fusion_rgb->inference_rgb(rgb);
    auto t1_det_rgb = std::chrono::high_resolution_clock::now();
    float t_det_rgb = std::chrono::duration<float, std::milli>(t1_det_rgb - t0_det_rgb).count();
    USER_LOG_INFO("detect (rgb): %f ms", t_det_rgb);
    USER_LOG_INFO("number of det boxes (rgb): %d", rects_det_rgb.size());

    auto t0_trk_rgb = std::chrono::high_resolution_clock::now();
    auto rects_trk_rgb = fusion_rgb->track(det_rects_rgb);
    auto t1_trk_rgb = std::chrono::high_resolution_clock::now();
    float t_trk_rgb = std::chrono::duration<float, std::milli>(t1_trk_rgb - t0_trk_rgb).count();
    USER_LOG_INFO("track (rgb): %f ms", t_trk_rgb);
    USER_LOG_INFO("number of trk boxes (rgb): %d", rects_trk_rgb.size());

    bboxs2img(trk_rects_inf, inf);
    bboxs2img(trk_rects_rgb, rgb);
    cv::imshow("inf", inf);
    cv::imshow("rgb", rgb);
    cv::waitKey(10);

    for (auto& trk : trk_rects_inf) {
        trk.tlwh[0] += 0;
        trk.tlwh[1] += 160;
    }
    for (auto& trk : trk_rects_rgb) {
        trk.tlwh[0] += 960;
        trk.tlwh[1] += 160;
    }
    std::vector<STrack> trk_rects;
    trk_rects.reserve(trk_rects_inf.size() + trk_rects_rgb.size());
    trk_rects.insert(trk_rects.end(), trk_rects_inf.begin(), trk_rects_inf.end());
    trk_rects.insert(trk_rects.end(), trk_rects_rgb.begin(), trk_rects_rgb.end());
    std::string res = bboxs2json(trk_rects);

    USER_LOG_INFO("result: %s", &res[0]);
    memcpy(send_buffer, &res[0], res.size()+1);
    send_size = res.size();

    if (connected) {
        /*
        // data transmission lowspeed
        E_DjiChannelAddress address = DJI_CHANNEL_ADDRESS_MASTER_RC_APP;
        retcode = DjiLowSpeedDataChannel_SendData(address, send_buffer, send_size);
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("send data error");
        } else {
            USER_LOG_INFO("send data success: %s", &send_buffer[0]);
        }
        */
        // mop
        auto t0_mop = std::chrono::high_resolution_clock::now();
        retcode = DjiMopChannel_SendData(s_MopChannelNormalOutHandle, send_buffer, send_size, &send_size);
        auto t1_mop = std::chrono::high_resolution_clock::now();
        float t_mop = std::chrono::duration<float, std::milli>(t1_mop - t0_mop).count();
        USER_LOG_INFO("mop: %f ms", t_mop);
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("send data error, stat:0x%08llX", retcode);
            if (retcode == DJI_ERROR_MOP_CHANNEL_MODULE_CODE_CONNECTION_CLOSE) {
                USER_LOG_ERROR("mop channel is disconnected");
                connected = false;
                DjiMopChannel_Close(s_MopChannelNormalOutHandle);
                DjiMopChannel_Destroy(s_MopChannelNormalOutHandle);
            }
        } else {
            USER_LOG_INFO("send data success: %d bytes", send_size);
        }
    }

    // handler->TaskSleepMs(200);
    mtx.unlock();
}

static void* ConnectTask(void*) {
    T_DjiOsalHandler *handler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode retcode;

    /*
    // data transmission
    retcode = DjiLowSpeedDataChannel_Init();
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("init data transmission module error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("init data transmission module success");
        connected = true;
    }
    */

    // mop
    retcode = DjiMopChannel_Init();
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("init mop channel error, stat:0x%08llX", retcode);
        return NULL;
    } else {
        USER_LOG_INFO("init mop channel success");
    }
#if PIPELINE_IS_RELIABLE
    retcode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_RELIABLE);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX", retcode);
        return NULL;
    } else {
        USER_LOG_INFO("mop channel create send handle success");
    }
#else
    retcode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_UNRELIABLE);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX", retcode);
        return NULL;
    } else {
        USER_LOG_INFO("mop channel create send handle success");
    }
#endif
    retcode = DjiMopChannel_Bind(s_MopChannelNormalHandle, PIPELINE_ID);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop bind channel error, stat:0x%08llX", retcode);
        return NULL;
    } else {
        USER_LOG_INFO("mop bind channel success");
    }
    while (1) {
        if (connected) {
            handler->TaskSleepMs(1000);
            continue;
        }
        retcode = DjiMopChannel_Accept(s_MopChannelNormalHandle, &s_MopChannelNormalOutHandle);
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("mop accept channel error, stat:0x%08llX", retcode);
        } else {
            USER_LOG_INFO("mop accept channel success");
            handler->TaskSleepMs(3000);
            connected = true;
        }
    }
}

static T_DjiReturnCode RecvCallback(const uint8_t *data, uint16_t len)
{
    char* printData = NULL;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    printData = (char*)osalHandler->Malloc(len + 1);
    if (printData == NULL) {
        USER_LOG_ERROR("malloc memory for printData fail.");
        return DJI_ERROR_SYSTEM_MODULE_CODE_MEMORY_ALLOC_FAILED;
    }

    strncpy(printData, (const char *) data, len);
    printData[len] = '\0';
    USER_LOG_INFO("receive data from mobile: %s, len:%d.", printData, len);

    osalHandler->Free(printData);

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static void* SendTask(void*) {
    T_DjiOsalHandler *handler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode retcode;
    T_DjiDataChannelState state = {0};

    const uint8_t data[] = "[[100,200,1000,2000,0,1],[100,200,1000,2000,0,1],[100,200,1000,2000,0,1],[100,200,1000,2000,0,1]]";

    E_DjiChannelAddress address = DJI_CHANNEL_ADDRESS_MASTER_RC_APP;
    while (1) {
        retcode = DjiLowSpeedDataChannel_SendData(address, data, sizeof(data));
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("send data error");
            return NULL;
        } else {
            USER_LOG_INFO("send data success");
        }

        /*
        const T_DjiDataChannelBandwidthProportionOfHighspeedChannel bandwidthProportionOfHighspeedChannel = {10, 60, 30};
        retcode = DjiHighSpeedDataChannel_SetBandwidthProportion(bandwidthProportionOfHighspeedChannel);
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("set data channel bandwidth proportion error");
            return NULL;
        } else {
            USER_LOG_INFO("set data channel bandwidth proportion success");
        }

        retcode = DjiHighSpeedDataChannel_SendDataStreamData(data, sizeof(data));
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("send data error");
            return NULL;
        } else {
            USER_LOG_INFO("send data success");
        }

        retcode = DjiHighSpeedDataChannel_GetDataStreamState(&state);
        if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("get data stream state error");
            return NULL;
        } else {
            USER_LOG_INFO("data stream state: %d, %d, %d", state.realtimeBandwidthLimit, state.realtimeBandwidthBeforeFlowController, state.busyState);
        }
        */

        handler->TaskSleepMs(250);
    }
}
/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
