#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "utils/OcrStruct.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class DbNet {
public:
    DbNet();

    ~DbNet();

    void setNumThread(int numOfThread);

    void initModel(const std::string &pathStr);

    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
                                      float boxThresh, float unClipRatio);

private:
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;
    char *inputName;
    char *outputName;

    const float meanValues[3] = {float(0.485 * 255), float(0.456 * 255), float(0.406 * 255)};
    const float normValues[3] = {float(1.0 / 0.229f / 255.0), float(1.0 / 0.224 / 255.0), float(1.0 / 0.225 / 255.0)};
};


#endif //__OCR_DBNET_H__
