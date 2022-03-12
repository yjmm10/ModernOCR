/*
 * @Author: Petrichor
 * @Date: 2022-03-07 13:22:18
 * @LastEditTime: 2022-03-12 11:59:18
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\AngleNet\AngleNet.h
 * 版权声明
 */
#ifndef __OCR_ANGLENET_H__
#define __OCR_ANGLENET_H__

// #include <utils/types.h>
// #include "utils/OcrStruct.h"
#include "core/modernocr.h"
#include <spdlog/spdlog.h>
using namespace ModernOCR;

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


class AngleNet {
public:
    AngleNet();

    ~AngleNet();

    void setNumThread(int numOfThread);

    // void initModel(const std::string &pathStr);
    bool LoadModel(const std::string &modelPath);

    std::vector<types::AngleInfo> getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                 const char *imgName, bool doAngle, bool mostAngle);

private:
    std::shared_ptr<spdlog::logger> log;
    bool isOutputAngleImg = false;

    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "AngleNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    char *inputName;
    char *outputName;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstWidth = 192;
    const int dstHeight = 32;

    types::AngleInfo getAngle(cv::Mat &src);
};


#endif //__OCR_ANGLENET_H__
