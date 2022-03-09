#include "DbNet.h"
#include "utils/OcrUtils.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
using namespace spdlog;
DbNet::DbNet() {
    log = spdlog::get("DbNet");
    if(log==nullptr)
    {
        log = spdlog::basic_logger_mt("DbNet", "logs/ModernOCR.txt");
        log->info("Create DbNet logs!");
    }else{
        log->info("Load DbNet logs!");
    }
}

DbNet::~DbNet() {
    delete session;
    free(inputName);
    free(outputName);
}

void DbNet::setNumThread(int numOfThread) {
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

void DbNet::initModel(const std::string &pathStr) {
try
{
#ifdef _WIN32
    std::wstring dbPath = strToWstr(pathStr);
    session = new Ort::Session(env, dbPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    log->info("DbNet successfully loaded!");
}
catch(const std::exception& e)
{
    SPDLOG_LOGGER_ERROR(log,e.what());
}
    getInputName(session, inputName);
    getOutputName(session, outputName);
}
//new 
inline std::vector<TextBox> DbNet::findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, std::vector<float> &ratio_wh,
                                 const float boxScoreThresh, const float unClipRatio) {
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    log->info("Find {} Contours!",123);
    for (unsigned int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;
        std::cout<<"here normal33"<<std::endl;  
        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / ratio_wh[0];
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), fMapMat.cols);

            clipMinBox[j].y = clipMinBox[j].y / ratio_wh[1];
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), fMapMat.rows);
        }

        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

inline std::vector<TextBox> DbNet::findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, ScaleParam &s,
                                 const float boxScoreThresh, const float unClipRatio) {
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    log->info("Find {} Contours!",123);
    for (unsigned int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / s.ratioWidth;
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), s.srcWidth);

            clipMinBox[j].y = clipMinBox[j].y / s.ratioHeight;
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), s.srcHeight);
        }

        rsBoxes.emplace_back(TextBox{clipMinBox, score});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, std::vector<float> &ratio_wh, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize = src.clone();
    // try
    // {
    //     resize(src, srcResize, cv::Size(src.width*ratio_wh[0], src.height*ratio_wh[1]));
    //     log->info("Resize images from ({}, {}) to ({}, {}).",src.cols,src.rows,s.dstWidth,s.dstHeight);
    // }
    // catch(const std::exception& e)
    // {
    //     SPDLOG_LOGGER_ERROR(log,e.what());
    // }
    std::cout<<srcResize.cols<<"           "<<srcResize.rows<<std::endl;
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};
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

    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));
    std::cout<<"here normal11"<<std::endl;
    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, ratio_wh, boxScoreThresh, unClipRatio);
}

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    try
    {
        resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
        log->info("Resize images from ({}, {}) to ({}, {}).",src.cols,src.rows,s.dstWidth,s.dstHeight);
    }
    catch(const std::exception& e)
    {
        SPDLOG_LOGGER_ERROR(log,e.what());
    }
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};
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

    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));

    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}
