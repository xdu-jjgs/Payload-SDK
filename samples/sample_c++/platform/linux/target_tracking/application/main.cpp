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

#include <liveview/dji_camera_stream_decoder.hpp>


#include <liveview/test_liveview_entry.hpp>
#include <perception/test_perception_entry.hpp>
#include <flight_control/test_flight_control.h>
#include <gimbal/test_gimbal_entry.hpp>
#include <hms/test_hms.h>
#include <waypoint_v2/test_waypoint_v2.h>
#include <waypoint_v3/test_waypoint_v3.h>
#include <gimbal_manager/test_gimbal_manager.h>
#include "application.hpp"
#include "fc_subscription/test_fc_subscription.h"
#include <gimbal_emu/test_payload_gimbal_emu.h>
#include <camera_emu/test_payload_cam_emu_media.h>
#include <camera_emu/test_payload_cam_emu_base.h>
#include <dji_logger.h>
#include "widget/test_widget.h"
#include "widget/test_widget_speaker.h"
#include <power_management/test_power_management.h>
#include "data_transmission/test_data_transmission.h"
#include <camera_manager/test_camera_manager.h>
#include "camera_manager/test_camera_manager_entry.h"

/* Private constants ---------------------------------------------------------*/
#define FPV_CAM          true
#define MAIN_CAM         false
#define VICE_CAM         false
#define TOP_CAM          false
#define TASK_STACK       2048
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

/* Private functions declaration ---------------------------------------------*/
static void ShowRgbImageCallback(CameraRGBImage img, void*);
static T_DjiReturnCode ReceiveDataFromPayload(const uint8_t *data, uint16_t len);

/* Exported functions definition ---------------------------------------------*/
int main(int argc, char **argv)
{
    Application application(argc, argv);

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

    retcode = DjiLowSpeedDataChannel_Init();
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("init data transmission module error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("init data transmission module success");
    }

    E_DjiChannelAddress address = DJI_CHANNEL_ADDRESS_PAYLOAD_PORT_NO1;
    retcode = DjiLowSpeedDataChannel_RegRecvDataCallback(address, ReceiveDataFromPayload);
    if (retcode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("register receive data from payload NO1 error");
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    } else {
        USER_LOG_INFO("register receive data from payload NO1 success");
    }

    osalHandler->TaskSleepMs(99999999);

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

/* Private functions definition-----------------------------------------------*/
static void ShowRgbImageCallback(CameraRGBImage img, void*)
{
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode returnCode;

    if (!mtx.try_lock()) return;

    Mat mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width * 3);
    cvtColor(mat, mat, COLOR_RGB2BGR);
    width = mat.size().width;
    height = mat.size().height;
    // USER_LOG_INFO("w=%d, h=%d", width, height);

    /*
    vector<unsigned char> data;
    imencode(".jpg", mat, data);
    USER_LOG_INFO("write to cpp2py %d bytes", data.size());
    write(fd_cpp2py, &data[0], data.size());
    */
    imwrite("/home/dji/Desktop/input.jpg", mat);

    // imshow("input", mat);
    // waitKey(1);

    if (connected) {
    }

    osalHandler->TaskSleepMs(1000);
    mtx.unlock();
}

static T_DjiReturnCode ReceiveDataFromPayload(const uint8_t *data, uint16_t len)
{
    char *printData = NULL;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    printData = osalHandler->Malloc(len + 1);
    if (printData == NULL) {
        USER_LOG_ERROR("malloc memory for printData fail.");
        return DJI_ERROR_SYSTEM_MODULE_CODE_MEMORY_ALLOC_FAILED;
    }

    strncpy(printData, (const char *) data, len);
    printData[len] = '\0';
    USER_LOG_INFO("receive data from payload port: %s, len:%d.", printData, len);
    DjiTest_WidgetLogAppend("receive data: %s, len:%d.", printData, len);

    osalHandler->Free(printData);

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}


/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/