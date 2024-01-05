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
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <liveview/dji_camera_stream_decoder.hpp>
#include <liveview/test_liveview.hpp>
#include <dji_low_speed_data_channel.h>
#include <dji_high_speed_data_channel.h>
#include <dji_mop_channel.h>
#include <dji_logger.h>

#include "application.hpp"
#include "Fusion.h"

/* Private constants ---------------------------------------------------------*/
#define FPV_CAM          true
#define MAIN_CAM         false
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

    /*
    if ((fd_cpp2py = open("/home/dji/Desktop/cpp2py", O_WRONLY)) != 0) {
        USER_LOG_ERROR("cpp2py fifo: failed to open");
    } else {
        USER_LOG_INFO("cpp2py fifo: opened");
    }
    if ((fd_py2cpp = open("/home/dji/Desktop/py2cpp", O_RDONLY)) != 0) {
        USER_LOG_ERROR("py2cpp fifo: failed to open");
    } else {
        USER_LOG_INFO("py2cpp fifo: opened");
    }
    */

    std::string inf_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/cfgs/inf.yaml";
    std::string inf_eng = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/ckpts/yolov5s_hb2_inf_640.trt";
    std::string rgb_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/cfgs/rgb.yaml";
    std::string rgb_eng = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/ckpts/yolov5m_xd_rgb_1280.trt";
    std::string camera_cfg = "/home/dji/Documents/Payload-SDK/samples/sample_c++/platform/linux/target_tracking/configs/camera.yaml";
    // fusion = new Fusion(inf_cfg, inf_eng, rgb_cfg, rgb_eng, camera_cfg);
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

    /*
    // py2cpp
    retcode = handler->TaskCreate("recv_output", RecvOutputTask, TASK_STACK_SIZE, NULL, &s_RecvOutputThread);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("recv output task create error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("recv output task create success");
    }
    */

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
// cv::VideoCapture* cap_inf = new cv::VideoCapture("/home/dji/Desktop/inf.mp4");
// cv::VideoCapture* cap_rgb = new cv::VideoCapture("/home/dji/Desktop/rgb.mp4");
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
    width = mat.size().width;
    height = mat.size().height;
    USER_LOG_INFO("w=%d, h=%d", width, height);

    cv::Mat inf = mat(cv::Rect(0, 160, 960, 760));
    cv::Mat rgb = mat(cv::Rect(960, 160, 960, 760));
    /*
    cap_inf->read(inf);
    if (inf.size().width == 0) {
        cap_inf = new cv::VideoCapture("/home/dji/Desktop/inf.mp4");
        mtx.unlock();
	return;
    }
    cap_rgb->read(rgb);
    if (rgb.size().width == 0) {
        cap_rgb = new cv::VideoCapture("/home/dji/Desktop/rgb.mp4");
        mtx.unlock();
	return;
    }
    */

    /*
    vector<unsigned char> data;
    cv::imencode(".jpg", mat, data);
    USER_LOG_INFO("write to cpp2py %d bytes", data.size());
    write(fd_cpp2py, &data[0], data.size());
    */
    /*
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    cv::imwrite("/home/dji/Desktop/input.jpg", mat);
    */

    /*
    auto rects = fusion->inference_rgb(mat);
    USER_LOG_INFO("number of boxes: %d", rects.size());
    auto res = fusion->track(rects);
    USER_LOG_INFO("result: %s", &res[0]);
    memcpy(send_buffer, &res[0], res.size()+1);
    send_size = res.size();

    for (const auto& rect : rects) {
        cv::putText(mat, "people", cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(mat, rst, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
    }
    cv::imshow("input", mat);
    cv::waitKey(10);
    */

    auto det_rects_inf = fusion_inf->inference_inf(inf);
    auto trk_rects_inf = fusion_inf->track(det_rects_inf);
    USER_LOG_INFO("number of det boxes: %d", det_rects_inf.size());
    USER_LOG_INFO("number of trk boxes: %d", trk_rects_inf.size());

    auto det_rects_rgb = fusion_rgb->inference_rgb(rgb);
    auto trk_rects_rgb = fusion_rgb->track(det_rects_rgb);
    USER_LOG_INFO("number of det boxes: %d", det_rects_rgb.size());
    USER_LOG_INFO("number of trk boxes: %d", trk_rects_rgb.size());

    bboxs2img(trk_rects_inf, inf);
    cv::imshow("inf", inf);
    bboxs2img(trk_rects_rgb, rgb);
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
        retcode = DjiMopChannel_SendData(s_MopChannelNormalOutHandle, send_buffer, send_size, &send_size);
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
