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
//#include <liveview/test_liveview_entry.hpp>
#include <mop_channel/test_mop_channel.h>
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
#include <liveview/test_liveview.hpp>
#include <liveview/dji_camera_stream_decoder.hpp>
#include <dji_platform.h>
#include <dji_mop_channel.h>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../../../sample_c/module_sample/utils/util_misc.h"
#include <jsoncpp/json/json.h>

#include "v5lite.hpp"

using namespace cv;
using namespace std;

#define PIPELINE_IS_RELIABLE 0
#define TASK_STACK           2048
#define PIPELINE_ID          49155
#define BUFFER_SIZE          (64 * 1024)
#define RECV_BUF             (100 * 1024)

#define FILE_SERVICE_CLIENT_MAX_NUM 10

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/
static char camIndex = '1';  //camindex: [0] Fpv Camera  [1] Main Camera  [2] Vice Camera  [3] Top Camera
V5lite *v5lite;
std::queue<std::vector<V5lite::DetectRes>> que;
std::mutex mtx;   //need to update to dji_mutex
bool running = true;


static T_DjiMopChannelHandle s_MopChannelNormalHandle;
static T_DjiMopChannelHandle s_MopChannelNormalOutHandle;
static T_DjiTaskHandle s_MopChannelNormalSendTask;
static T_DjiTaskHandle s_MopChannelNormalRecvTask;
static T_DjiSemaHandle s_MopChannelReadySema;
static bool s_MopChannelConnected = false;

static T_DjiTaskHandle s_fileServiceAccept;
static T_DjiMopChannelHandle s_fileServiceHandle;
//static T_MopFileServiceClientContent s_fileSerContent[FILE_SERVICE_CLIENT_MAX_NUM];

int width, height;

/* Private functions declaration ---------------------------------------------*/
static T_DjiReturnCode DjiTest_HighPowerApplyPinInit();
static T_DjiReturnCode DjiTest_WriteHighPowerApplyPin(E_DjiPowerManagementPinState pinState);
static T_DjiReturnCode DjiUser_GetCurrentFileDirPath(const char *filePath, uint32_t pathBufferSize, char *dirPath);

static void *CameraStreamView(void *q);
static void ShowRgbImageCallback(CameraRGBImage img, void *userData);
static void *MopServerTask(void *p);
static void *SendTask_json(void *arg);
static void *RecvTask(void *arg);

/* Exported functions definition ---------------------------------------------*/
int main(int argc, char **argv)
{
    // function adding: select cameras in yaml

    Application application(argc, argv);
    char inputChar;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiReturnCode returnCode;
    T_DjiTestApplyHighPowerHandler applyHighPowerHandler;

    v5lite = new V5lite(argv[1]);
    v5lite->LoadEngine(argv[2]);

    pthread_t selectCam; 
    int camRst = pthread_create(&selectCam, NULL, CameraStreamView, NULL);
    //int srvRst = pthread_create(&serverTask, NULL, MopServerTask, NULL);
    returnCode = DjiMopChannel_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel init error, stat:0x%08llX.", returnCode);
        
    }

    returnCode = osalHandler->TaskCreate("mop_msdk_send_task", SendTask_json,
                                         TASK_STACK, NULL, &s_MopChannelNormalSendTask);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk send task create error, stat:0x%08llX.", returnCode);
        
    }
    returnCode = osalHandler->TaskCreate("mop_msdk_recv_task", RecvTask,
                                         TASK_STACK, NULL, &s_MopChannelNormalRecvTask);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk recv task create error, stat:0x%08llX.", returnCode);
        return DJI_ERROR_SYSTEM_MODULE_CODE_UNKNOWN;
    }
    /*if (srvRst != 0 ) {
        USER_LOG_ERROR("Server task create failed!\n");
    } else {
        USER_LOG_INFO("Server task create success!\n");
    }*/
    pthread_join(selectCam, NULL);
    //thread_join(serverTask, NULL);

    

    osalHandler->TaskSleepMs(2000);

    //goto start;
}

/* Private functions definition-----------------------------------------------*/
static T_DjiReturnCode DjiTest_HighPowerApplyPinInit()
{
    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static T_DjiReturnCode DjiTest_WriteHighPowerApplyPin(E_DjiPowerManagementPinState pinState)
{
    //attention: please pull up the HWPR pin state by hardware.
    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static void *CameraStreamView(void *q)
{
    char cameraIndexChar = 0;
    char demoIndexChar = 0;
    char isQuit = 0;
    CameraRGBImage camImg;
    char fpvName[] = "FPV_CAM";
    char mainName[] = "MAIN_CAM";
    char viceName[] = "VICE_CAM";
    char topName[] = "TOP_CAM";
    auto *Sample = new LiveviewSample();

    switch (camIndex) {
        case '0':
            Sample->StartFpvCameraStream(&ShowRgbImageCallback, &fpvName);
            break;
        case '1':
            Sample->StartMainCameraStream(&ShowRgbImageCallback, &mainName);
            break;
        case '2':
            Sample->StartViceCameraStream(&ShowRgbImageCallback, &viceName);
            break;
        case '3':
            Sample->StartTopCameraStream(&ShowRgbImageCallback, &topName);
            break;
        default:
            cout << "No camera selected";
            delete Sample;
            return NULL;
    }

    cout << "Please enter the 'q' or 'Q' to quit camera stream view\n"
         << endl;

    while (true) {
        cin >> isQuit;
        if (isQuit == 'q' || isQuit == 'Q') {
            break;
        }
    }

    switch (camIndex) {
        case '0':
            Sample->StopFpvCameraStream();
            break;
        case '1':
            Sample->StopMainCameraStream();
            break;
        case '2':
            Sample->StopViceCameraStream();
            break;
        case '3':
            Sample->StopTopCameraStream();
            break;
        default:
            cout << "No camera selected";
            delete Sample;
            return NULL;
    }

    delete Sample;
}

static void ShowRgbImageCallback(CameraRGBImage img, void *userData)
{
    string name = string(reinterpret_cast<char *>(userData));
    if (!mtx.try_lock()) return;

    Mat mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width * 3);
    USER_LOG_INFO("w=%d, h=%d", img.width, img.height);
    width = img.width;
    height = img.height;
    
    cvtColor(mat, mat, COLOR_RGB2BGR);
    //imshow(name, mat);
    auto rects = v5lite->InferenceImage(mat);
    
    if (running) {
    que.push(rects);
    }
    
    //osalHandler->TaskSleepMs(100);
    mtx.unlock();
    //imshow(name, mat);
    //cv::waitKey(1);

}

/*
static void *MopServerTask(void *p) {
    T_DjiReturnCode returnCode;
    T_DjiOsalHandler *threadmag = DjiPlatform_GetOsalHandler();

    returnCode = DjiMopChannel_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel init error, stat:0x%08llX.", returnCode);
        return NULL;
    }

    returnCode = threadmag->TaskCreate("mop_msdk_send_task", SendTask_json,
                                         BUFFER_SIZE, NULL, &s_MopChannelNormalSendTask);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel msdk send task create error, stat:0x%08llX.", returnCode);
        return NULL;
    }

  return NULL;
}
*/

static void* RecvTask(void *arg){
    uint8_t *recvBuf = NULL;
    uint32_t realLen;
    T_DjiReturnCode returnCode;
    uint32_t recvDataCount = 0;
    T_DjiOsalHandler *recvserver = DjiPlatform_GetOsalHandler();

    recvserver->TaskSleepMs(3000);

    recvBuf = static_cast<uint8_t *>(recvserver->Malloc(RECV_BUF));
    if (recvBuf == NULL) {
        USER_LOG_ERROR("[File-Service]  Malloc recv buffer error");
        running = false;
        return NULL;
    }
WAIT:
    if (s_MopChannelConnected == true){
        goto BEGINRECV;
    }
    else{
        recvserver->TaskSleepMs(3000);
        goto WAIT;
    }

#if PIPELINE_IS_RELIABLE
    returnCode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_RELIABLE);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
        return NULL;
    }
#else
    returnCode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_UNRELIABLE);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
        return NULL;
    }
#endif

BEGINRECV:
    while(running){
        memset(recvBuf, 0, RECV_BUF);

        returnCode = DjiMopChannel_RecvData(s_MopChannelNormalOutHandle, recvBuf,
                                            RECV_BUF, &realLen);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            if (returnCode == DJI_ERROR_MOP_CHANNEL_MODULE_CODE_CONNECTION_CLOSE) {
                USER_LOG_INFO("mop channel is disconnected");
                s_MopChannelConnected = false;
                recvserver->TaskSleepMs(3000);
                DjiMopChannel_Close(s_MopChannelNormalOutHandle);
                DjiMopChannel_Destroy(s_MopChannelNormalOutHandle);
                //goto REACCEPT;
            }
        } else {
            USER_LOG_INFO("mop channel recv data from channel length:%d count:%d", realLen, recvDataCount++);
        }
    }
}

static void* SendTask_json(void *arg) {
    uint8_t *sendBuf=NULL;
    uint32_t realLen = 0;
    T_DjiReturnCode returnCode;
    uint32_t sendDataCount = 0;
    T_DjiOsalHandler *sendserver = DjiPlatform_GetOsalHandler();

    sendserver->TaskSleepMs(3000);

    sendBuf = static_cast<uint8_t *>(sendserver->Malloc(BUFFER_SIZE));
    if (sendBuf == NULL) {
        USER_LOG_ERROR("[File-Service]  Malloc send buffer error");
        running = false;
        return NULL;
    }

#if PIPELINE_IS_RELIABLE
    returnCode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_RELIABLE);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
        return NULL;
    }
#else
    returnCode = DjiMopChannel_Create(&s_MopChannelNormalHandle, DJI_MOP_CHANNEL_TRANS_UNRELIABLE);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop channel create send handle error, stat:0x%08llX.", returnCode);
        return NULL;
    }
#endif

REBIND:
    returnCode = DjiMopChannel_Bind(s_MopChannelNormalHandle, PIPELINE_ID);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("mop bind channel error :0x%08llX", returnCode);
        sendserver->TaskSleepMs(3000);
        goto REBIND;
    }
REACCEPT:
    USER_LOG_INFO("--------------------------mop channel is connecting--------------------------");
    returnCode = DjiMopChannel_Accept(s_MopChannelNormalHandle, &s_MopChannelNormalOutHandle);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_WARN("mop accept channel error :0x%08llX", returnCode);
        sendserver->TaskSleepMs(3000);
        goto REACCEPT;
    }
    s_MopChannelConnected = true;
    USER_LOG_INFO("--------------------------mop channel is connected--------------------------");

    while (running) {

        if (que.empty()) {
        continue;
        }
        auto rects = que.front();
        que.pop();

        Json::Value pkg;
        Json::Value bbox;

        int offset = 0;
        //offset += sprintf_s(sendBuf + offset, BUFFER_SIZE, "%d,%d:", width, height);
        pkg["size"].append(width);
        pkg["size"].append(height);

        int i = 0;
        int max_det = 21;
        bbox[i].append(0);
        bbox[i].append(0);
        bbox[i].append(0);
        bbox[i].append(0);
        bbox[i].append(0);
        bbox[i].append(0);
        i++;
        for (const auto &rect: rects) {
        if (i >= max_det) {
            break;
        }
        //Json::Value box;
        bbox[i].append(int(rect.x));
        bbox[i].append(int(rect.y));
        bbox[i].append(int(rect.w));
        bbox[i].append(int(rect.h));
        bbox[i].append(int(rect.prob*100));
        //bbox[i].append(rect.prob);
        bbox[i].append(rect.classes);
        //pkg["bbox"].push_back({int(rect.x), int(rect.y), int(rect.w), int(rect.h), rect.prob, rect.classes});
        //bbox.append(box);
        i += 1;
        }

        pkg["bbox"] = bbox;
        Json::FastWriter sw;
        string s=sw.write(pkg);
        memcpy( sendBuf, &s[0], s.size());
        realLen = s.size();

        returnCode = DjiMopChannel_SendData(s_MopChannelNormalOutHandle, sendBuf,
                                            realLen, &realLen);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
            USER_LOG_ERROR("mop channel send data to channel error,stat:0x%08llX", returnCode);
            //Condition 1: returncode = 0xf0000000b -- Connection of channel is closed. The peer channel do not work or abnormally be closed. 
            //Suggestion : Please confirm state of the peer channel and reaccept the connection request of MSDK/OSDK?  Not solved yet.
            USER_LOG_INFO("mop channel is disconnected");
            //Condition 2: returncode = 0x000000E4b -- artifical disconneted the channel on app; it has been solved -- forced to jump to REACCEPCT. But it will over-solve Condition 1.
            s_MopChannelConnected = false;
            sendserver->TaskSleepMs(3000);
            DjiMopChannel_Close(s_MopChannelNormalOutHandle);
            DjiMopChannel_Destroy(s_MopChannelNormalOutHandle);
            goto REACCEPT;
            
        } else {
            sendDataCount++;
            USER_LOG_INFO("mop channel send data to channel length:%d count:%d", realLen, sendDataCount);
        }

        //sendserver->TaskSleepMs(1000 / 1);

        /*
        auto js_time = std::chrono::high_resolution_clock::to_time_t(std::chrono::high_resolution_clock::now());
        char jt[20];
        strftime(jt, sizeof(jt), "%Y%m%d-%H:%M:%S", localtime(&js_time));
        string wrtime = jt;
        pkg["time"] = wrtime;
        Json::FastWriter wt;
        ofstream os;
        
        os.open(jstxt, std::ios::out | std::ios::app);
        if (!os.is_open()){
        cout << "write stop" << endl;
        }
        os << wt.write(pkg);
        os.close();
        

        if (mopRet == MOP_PASSED) {
        DSTATUS("[File-Service] upload request ack %s", sendBuf);
        } else if (mopRet == MOP_TIMEOUT) {
        DERROR("[File-Service] send timeout");
        } else if (mopRet == MOP_CONNECTIONCLOSE) {
        DERROR("[File-Service] connection close");
        running = false;
        } else {
        DERROR("[File-Service] send error");
        }*/
        
    }
    sendserver->Free(sendBuf);
    return NULL;
}
/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
