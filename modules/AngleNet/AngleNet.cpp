#include <numeric>
#include "AngleNet.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
// // #include "AngleNet.h"
// // #include "utils/OcrUtils.h"
// #include <numeric>
// // #include "utils/operators.h"

AngleNet::AngleNet() {
    log = spdlog::get(_modelName);
    if(log==nullptr)
    {
        log = spdlog::basic_logger_mt(_modelName, "logs/ModernOCR.log");
    }
    log->info("{} Version: {} !", _modelName, _version);
}

AngleNet::~AngleNet() {
    delete session;
    free(inputName);
    free(outputName);
}

void AngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //===session options===
    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    //sessionOptions.SetIntraOpNumThreads(numThread);
    //set OMP_NUM_THREADS=16

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    sessionOptions.SetInterOpNumThreads(numThread);

    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

bool AngleNet::LoadModel(const std::string &modelPath) {
try
{
#ifdef _WIN32
    std::wstring dbPath = str::strToWstr(modelPath);
    session = new Ort::Session(env, dbPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif
    log->info("{} Model[{}] successfully loaded!", _modelName, modelPath);
}
catch(const std::exception& e)
{
    SPDLOG_LOGGER_ERROR(log,e.what());
}
    ort::getInputName(session, inputName);
    ort::getOutputName(session, outputName);
    return true;
}

// void AngleNet::initModel(const std::string &pathStr) {
// #ifdef _WIN32
//     std::wstring anglePath = str::strToWstr(pathStr);
//     session = new Ort::Session(env, anglePath.c_str(), sessionOptions);
// #else
//     session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
// #endif
//     ort::getInputName(session,inputName);
//     ort::getOutputName(session,outputName);
// }

types::AngleInfo AngleNet::getAngle(cv::Mat &src) {

    std::vector<float> inputTensorValues;
    op::MeanNormalize(src, meanValues, normValues,inputTensorValues);

    std::array<int64_t, 4> inputShape{1, src.channels(), src.rows, src.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());

    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);

    return {std::max_element(outputData.begin(),outputData.end()) - outputData.begin(),float(*std::max_element(outputData.begin(),outputData.end()))};
}

/*
types::AngleResult AngleNet::Run(std::vector<cv::Mat> &src, bool doAngle, bool mostAngle){
    double clsStartTime = utils::GetCurrentTime();
    auto angleImg = op::ResizeByValue(src, dstWidth, dstHeight);
    double preprocessTime = utils::GetCurrentTime() - clsStartTime;
    log->debug("Preprocess: [src_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
                            src.cols,src.rows,angleImg.cols,angleImg.rows,
                            preprocessTime);
    double inferenceTime = utils::GetCurrentTime();
    types::AngleInfo angle = getAngle(angleImg);
    double inferenceTime = utils::GetCurrentTime() - startAngle;
    log->debug("Inference: [cost {:.5} s].", inferenceTime);

// assert(!partImgs.isempty());
    int size = src.size();
    std::vector<types::AngleInfo> angles(size);
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            //preprocess
            double startAngle = utils::GetCurrentTime();
            auto angleImg = op::ResizeByValue(src[i], dstWidth, dstHeight);
            double preprocessTime = utils::GetCurrentTime() - startAngle;
            log->debug("Preprocess: [src_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
                            src[i].cols,src[i].rows,angleImg.cols,angleImg.rows,
                            preprocessTime);
            
            //inference
            // 模型推理
            startAngle = utils::GetCurrentTime();
            types::AngleInfo angle = getAngle(angleImg);
            double inferenceTime = utils::GetCurrentTime() - startAngle;
            log->debug("Inference: [cost {:.5} s].", inferenceTime);

            angles[i] = angle;

            // //OutPut AngleImg
            // if (isOutputAngleImg) {
            //     std::string angleImgFile = image::getDebugImgFilePath(path, imgName, i, "-angle-");
            //     image::saveImg(angleImg, angleImgFile.c_str());
            // }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = types::AngleInfo{-1, 0.f,0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = op::GetAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            types::AngleInfo angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;

}

*/
std::vector<types::AngleInfo> AngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                       const char *imgName, bool doAngle, bool mostAngle) {
    // assert(!partImgs.isempty());
    int size = partImgs.size();
    std::vector<types::AngleInfo> angles(size);
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            //preprocess
            double startAngle = utils::GetCurrentTime();
            auto angleImg = op::ResizeByValue(partImgs[i], dstWidth, dstHeight);
            double preprocessTime = utils::GetCurrentTime() - startAngle;
                log->debug("Preprocess: [src_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
                            partImgs[i].cols,partImgs[i].rows,angleImg.cols,angleImg.rows,
                            preprocessTime);
                
            //inference
            // 模型推理
            startAngle = utils::GetCurrentTime();
            types::AngleInfo angle = getAngle(angleImg);
            double inferenceTime = utils::GetCurrentTime() - startAngle;
            log->debug("Inference: [cost {:.5} s].", inferenceTime);

            angles[i] = angle;

            //OutPut AngleImg
            if (isOutputAngleImg) {
                std::string angleImgFile = image::getDebugImgFilePath(path, imgName, i, "-angle-");
                image::saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = types::AngleInfo{-1, 0.f,0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = op::GetAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            types::AngleInfo angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}