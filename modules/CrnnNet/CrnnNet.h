/*
 * @Author: Petrichor
 * @Date: 2022-03-07 17:27:29
 * @LastEditTime: 2022-03-12 12:08:45
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\CrnnNet\CrnnNet.h
 * 版权声明
 */
#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "core/modernocr.h"
using namespace ModernOCR;
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
// #include <opencv2/opencv.hpp>
class CrnnNet {
public:

    CrnnNet();

    ~CrnnNet();

    void setNumThread(int numOfThread);

    void initModel(const std::string &pathStr, const std::string &keysPath);
    bool LoadModel(const std::string &modelPath, const std::string &keysPath);

    std::vector<types::RecInfo> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName);

private:
    std::shared_ptr<spdlog::logger> log;
    bool isOutputDebugImg = false;
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    char *inputName;
    char *outputName;

    const float meanValues[3] = {127.5f, 127.5f, 127.5f};
    const float normValues[3] = {float(1.0 / 127.5), float(1.0 / 127.5), float(1.0 / 127.5)};
    const int dstHeight = 32;

    std::vector<std::string> keys;

    types::RecInfo scoreToTextLine(const std::vector<float> &outputData, int h, int w);

    types::RecInfo getTextLine(const cv::Mat &src);
};


#endif //__OCR_CRNNNET_H__
