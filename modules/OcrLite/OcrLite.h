/*
 * @Author: Petrichor
 * @Date: 2022-03-07 14:21:06
 * @LastEditTime: 2022-03-09 23:29:19
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\OcrLite\OcrLite.h
 * 版权声明
 */
#ifndef __OCR_LITE_H__
#define __OCR_LITE_H__

#include "opencv2/core.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "utils/OcrStruct.h"
#include "DbNet/DbNet.h"
#include "AngleNet/AngleNet.h"
#include "CrnnNet/CrnnNet.h"

class OcrLite {
public:
    OcrLite();

    ~OcrLite();

    void setNumThread(int numOfThread);

    void initLogger(bool isConsole, bool isPartImg, bool isResultImg);

    void enableResultTxt(const char *path, const char *imgName);

    bool initModels(const std::string &detPath, const std::string &clsPath,
                    const std::string &recPath, const std::string &keysPath);

    void Logger(const char *format, ...);

    OcrResult detect(const char *path, const char *imgName,
                     int padding, int maxSideLen,
                     float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);

    OcrResult detect(const cv::Mat &mat,
                     int padding, int maxSideLen,
                     float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);
private:
    bool isOutputConsole = false;
    bool isOutputPartImg = false;
    bool isOutputResultTxt = false;
    bool isOutputResultImg = false;
    FILE *resultTxt;
    DbNet dbNet;
    AngleNet angleNet;
    CrnnNet crnnNet;

    std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes,
                                       const char *path, const char *imgName);

    OcrResult detect(const char *path, const char *imgName,
                     cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                     float boxScoreThresh = 0.6f, float boxThresh = 0.3f,
                     float unClipRatio = 2.0f, bool doAngle = true, bool mostAngle = true);
    OcrResult detect(const char* path, const char* imgName, 
                    cv::Mat &src, const std::vector<float> ratio_wh, const int padding,
                    float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);
   
};

#endif //__OCR_LITE_H__
