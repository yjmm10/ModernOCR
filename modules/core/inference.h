/*
 * @Author: Petrichor
 * @Date: 2022-03-11 12:49:03
 * @LastEditTime: 2022-03-11 16:26:10
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\core\inference.h
 * 版权声明
 */

#ifndef __ORT_H__
#define __ORT_H__
#include <vector>
#include "types.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
namespace ModernOCR{

    namespace ort{
        std::vector<char *> getInputNames(Ort::Session *session);

        std::vector<char *> getOutputNames(Ort::Session *session);

        void getInputName(Ort::Session *session, char *&inputName);

        void getOutputName(Ort::Session *session, char *&outputName);

         void ModelInference(Ort::Session *&session, char *&inputName, char *&outputName, 
                            cv::Mat& dst_resize, const float* meanValues, const float* normValues,
                            float *floatArray, std::vector<int64_t>& outputShape, int64_t& outputCount);

    };
};

#endif