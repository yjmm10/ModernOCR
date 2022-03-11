/*
 * @Author: Petrichor
 * @Date: 2022-03-11 11:28:13
 * @LastEditTime: 2022-03-11 16:26:04
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\core\inference.cpp
 * 版权声明
 */
#include"inference.h"

#include "operators.h"
#include <numeric>


namespace ModernOCR{
    namespace ort{
        std::vector<char *> getInputNames(Ort::Session *session) {
            Ort::AllocatorWithDefaultOptions allocator;
            size_t numInputNodes = session->GetInputCount();
            std::vector<char *> inputNodeNames(numInputNodes);
            std::vector<int64_t> inputNodeDims;

            //printf("Number of inputs = %zu\n", numInputNodes);

            for (int i = 0; i < numInputNodes; i++) {
                // print input node names
                char *inputName = session->GetInputName(i, allocator);
                printf("InputName[%d]=%s\n", i, inputName);
                inputNodeNames[i] = inputName;

                // print input node types
                //Ort::TypeInfo typeInfo = session->GetInputTypeInfo(i);
                //auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

                //ONNXTensorElementDataType type = tensorInfo.GetElementType();
                //printf("Input[%d] type=%d\n", i, type);

                // print input shapes/dims
                //inputNodeDims = tensorInfo.GetShape();
                //printf("Input[%d] num_dims=%zu\n", i, inputNodeDims.size());
                /*for (int j = 0; j < inputNodeDims.size(); j++)
                    printf("Input[%d] dim%d=%jd\n", i, j, inputNodeDims[j]);*/
            }
            return inputNodeNames;
        }

        std::vector<char *> getOutputNames(Ort::Session *session) {
            Ort::AllocatorWithDefaultOptions allocator;
            size_t numOutputNodes = session->GetOutputCount();
            std::vector<char *> outputNodeNames(numOutputNodes);
            //std::vector<int64_t> outputNodeDims;

            //printf("Number of outputs = %zu\n", numOutputNodes);

            for (int i = 0; i < numOutputNodes; i++) {
                // print input node names
                char *outputName = session->GetOutputName(i, allocator);
                printf("OutputName[%d]=%s\n", i, outputName);
                outputNodeNames[i] = outputName;

                // print input node types
                //Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
                //auto tensorInfo = type_info.GetTensorTypeAndShapeInfo();

                //ONNXTensorElementDataType type = tensorInfo.GetElementType();
                //printf("Output %d : type=%d\n", i, type);

                // print input shapes/dims
                //outputNodeDims = tensorInfo.GetShape();
                //printf("Output %d : num_dims=%zu\n", i, outputNodeDims.size());
                /*for (int j = 0; j < outputNodeDims.size(); j++)
                    printf("Output %d : dim %d=%jd\n", i, j, outputNodeDims[j]);*/
            }
            return outputNodeNames;
        }

        void getInputName(Ort::Session *session, char *&inputName) {
            size_t numInputNodes = session->GetInputCount();
            if (numInputNodes > 0) {
                Ort::AllocatorWithDefaultOptions allocator;
                {
                    char *t = session->GetInputName(0, allocator);
                    inputName = my_strdup(t);
                    allocator.Free(t);
                }
            }
        }

        void getOutputName(Ort::Session *session, char *&outputName) {
            size_t numOutputNodes = session->GetInputCount();
            if (numOutputNodes > 0) {
                Ort::AllocatorWithDefaultOptions allocator;
                {
                    char *t = session->GetOutputName(0, allocator);
                    outputName = my_strdup(t);
                    allocator.Free(t);
                }
            }
        }

         void ModelInference(Ort::Session *&session, char *&inputName, char *&outputName, 
                            cv::Mat& dst_resize, const float* meanValues, const float* normValues,
                            float *floatArray, std::vector<int64_t>& outputShape, int64_t& outputCount){
            std::vector<float> inputTensorValues;
            op::MeanNormalize(dst_resize,meanValues, normValues,inputTensorValues);

            std::array<int64_t, 4> inputShape{1, dst_resize.channels(), dst_resize.rows, dst_resize.cols};
            auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                                    inputTensorValues.size(), inputShape.data(),
                                                                    inputShape.size());
            assert(inputTensor.IsTensor());
            auto outputTensor = session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);
            assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());
            outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
            outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                                std::multiplies<int64_t>());
            floatArray = outputTensor.front().GetTensorMutableData<float>();
            // std::cout <<"这是1"<<*floatArray <<"\t"<<outputCount<<"\t"<<outputShape.size()<< std::endl;
            // fl
            
            // float* &floatArray,
        }

    };

};