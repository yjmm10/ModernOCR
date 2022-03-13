/*
 * @Author: Petrichor
 * @Date: 2022-03-07 17:27:29
 * @LastEditTime: 2022-03-13 17:53:59
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\DbNet\DbNet.h
 * 版权声明
 */
#ifndef __DBNET_H__
#define __DBNET_H__

#include "core/modernocr.h"
using namespace ModernOCR;
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

class DbNet {
public:
    DbNet();

    ~DbNet();

    void setNumThread(int numOfThread);
    bool LoadModel(const std::string &modelPath);
    std::vector<cv::Mat> Run(const std::string imgPath,types::DbNetParam newParam);
    std::vector<cv::Mat> Run(cv::Mat &src, types::DbNetParam newParam);
    std::vector<cv::Mat> Run(cv::Mat &src, int padding, float boxScoreThresh, float boxThresh, float unClipRatio, int maxSideLen);

private:
    const std::string _modelName = "DbNet";
    const std::string _version = "0.1.0";
    std::shared_ptr<spdlog::logger> log;
    bool update = false;
    types::DbNetParam params{};
    types::DetResult detResult;
    std::vector<double> allTimes;
    std::vector<std::string> timesInfo{"preprocess","inference","postprocess","crop"};

    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;
    char *inputName;
    char *outputName;

    const float meanValues[3] = {float(0.485 * 255), float(0.456 * 255), float(0.406 * 255)};
    const float normValues[3] = {float(1.0 / 0.229f / 255.0), float(1.0 / 0.224 / 255.0), float(1.0 / 0.225 / 255.0)};
};


#endif //__DBNET_H__
