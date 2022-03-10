/*
 * @Author: Petrichor
 * @Date: 2022-03-07 17:27:29
 * @LastEditTime: 2022-03-10 20:27:52
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\DbNet\DbNet.h
 * 版权声明
 */
#ifndef __DBNET_H__
#define __DBNET_H__

#include "utils/OcrStruct.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "utils/types.h"
class DbNet {
public:
    DbNet();

    ~DbNet();

    // struct TextBox {
    // std::vector<cv::Point> boxPoint;
    // float score;
    // };

    void setNumThread(int numOfThread);
    bool LoadModel(const std::string &modelPath);
    // bool Preprocess(cv::Mat &src,)

    void initModel(const std::string &pathStr);

    // std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
    //                                   float boxThresh, float unClipRatio);
    std::vector<cv::Mat> Run(cv::Mat &src, int padding, float boxScoreThresh, float boxThresh, float unClipRatio, int maxSideLen);
    std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<types::TextBox> &textBoxes);
    std::string getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag);
    // std::vector<float> substractMeanNormalize1(cv::Mat &src, const float *meanVals, const float *normVals);

private:
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;
    char *inputName;
    char *outputName;
    std::shared_ptr<spdlog::logger> log;
    // inline std::vector<TextBox> findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, ScaleParam &s,
    //                              const float boxScoreThresh, const float unClipRatio);

    const float meanValues[3] = {float(0.485 * 255), float(0.456 * 255), float(0.406 * 255)};
    const float normValues[3] = {float(1.0 / 0.229f / 255.0), float(1.0 / 0.224 / 255.0), float(1.0 / 0.225 / 255.0)};
    // const float mean[3] = {float(0.485), float(0.456 ), float(0.406)};
    // const float norm[3] = {float(1.0 / 0.229f ), float(1.0 / 0.224), float(1.0 / 0.225)};
    // std::vector<float> meanValues_vec{float(0.485), float(0.456), float(0.406)};
    // std::vector<float> normValues_vec{float(1.0 / 0.229f), float(1.0 / 0.224), float(1.0 / 0.225)};

    // std::vector<float> substractMeanNormalize(cv::Mat &src, const std::vector<float> meanVals, const std::vector<float> normVals);
};

namespace ModernOCR{
    namespace OP{
        namespace DbNet{
            // class pre
        }
        
    }
}


#endif //__DBNET_H__
