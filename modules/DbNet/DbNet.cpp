#include <numeric>
#include "DbNet.h"

#include "core/modernocr.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using namespace spdlog;
using namespace ModernOCR;
DbNet::DbNet() {
    const std::string modelName("DbNet");
    log = spdlog::get(modelName);
    if(log==nullptr)
    {
        log = spdlog::basic_logger_mt(modelName, "logs/ModernOCR.log");
        log->info("Create {} logs!",modelName);
    }else{
        log->info("Load {} logs!",modelName);
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

bool DbNet::LoadModel(const std::string &modelPath) {
try
{
#ifdef _WIN32
    std::wstring dbPath = str::strToWstr(modelPath);
    session = new Ort::Session(env, dbPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif
    log->info("DbNet Model[{}] successfully loaded!",modelPath);
}
catch(const std::exception& e)
{
    SPDLOG_LOGGER_ERROR(log,e.what());
}
    ort::getInputName(session, inputName);
    ort::getOutputName(session, outputName);
    return true;
}

// void DbNet::initModel(const std::string &pathStr) {
// try
// {
// #ifdef _WIN32
//     std::wstring dbPath = str::strToWstr(pathStr);
//     session = new Ort::Session(env, dbPath.c_str(), sessionOptions);
// #else
//     session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
// #endif
//     log->info("DbNet successfully loaded!");
// }
// catch(const std::exception& e)
// {
//     SPDLOG_LOGGER_ERROR(log,e.what());
// }
//     ort::getInputName(session, inputName);
//     ort::getOutputName(session, outputName);
// }

std::vector<cv::Mat>
DbNet::Run(cv::Mat &src, int padding, float boxScoreThresh, float boxThresh, float unClipRatio,int maxSideLen) {
    double startTime = utils::GetCurrentTime();
    // preprocess
    // 边界填充之后增加获取比例
    //src 640 padding 740 resize 736
    cv::Mat dst_padding,dst_resize;
    op::SetPadding(src,dst_padding,padding);
    std::vector<float> ratio_wh;
    op::ResizeByMaxSide(dst_padding,dst_resize,maxSideLen,ratio_wh);
    double preprocessTime = utils::GetCurrentTime() - startTime;
    log->info("Preprocess: [src_wh: ({},{}), padding_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
                src.cols,src.rows,dst_padding.cols,dst_padding.rows,dst_resize.cols,dst_resize.rows, preprocessTime);
    
    //inference
    // std::cout<<outputName<<std::endl;
    // std::vector<int64_t> outputShape;
    // int64_t outputCount;
    // float* floatArray;
    // ort::ModelInference(session, inputName, outputName,
    //                 dst_resize,meanValues, normValues,
    //                 floatArray,outputShape, outputCount);
    
    startTime = utils::GetCurrentTime();
    std::vector<float> inputTensorValues;

    op::MeanNormalize(dst_resize,meanValues, normValues,inputTensorValues);

    std::array<int64_t, 4> inputShape{1, dst_resize.channels(), dst_resize.rows, dst_resize.cols};
    auto memoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    Ort::Value inputTensor(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size()));
    assert(inputTensor.IsTensor());
    auto outputTensor(session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1));
    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    
    double inferenceTime = utils::GetCurrentTime() - startTime;
    log->info("Inference: [cost {:.5} s].", inferenceTime);
    startTime = utils::GetCurrentTime();

    //-----Data preparation-----
    cv::Mat fMapMat(dst_resize.rows, dst_resize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));
    
    assert(dst_resize.data());
    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    //postprocess    
    //  形态学闭运算，先膨胀再腐蚀,增大范围
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat dilation_map;
    cv::morphologyEx(norfMapMat, dilation_map, cv::MORPH_CLOSE, dila_ele);
    dilation_map.copyTo(norfMapMat);

    
    float minArea = 3;
    // resize图像的坐标点信息
    std::vector<types::BoxInfo> rsBoxes;
    rsBoxes.clear();

    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    log->debug("Postprocess: [find {} Contours, cost {:.5}s.]",contours.size(),utils::GetCurrentTime() - startTime);
    for (auto contour : contours) {
        if(contour.size()<4)    continue;
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox;
        op::GetMinBoxes(contour, minSideLen, perimeter,minBox);
        if (minSideLen < minArea)
            continue;
        float score;
        op::BoxScoreFast(fMapMat, contour,score);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox;
        op::UnClip(minBox, perimeter, unClipRatio, clipBox);
        
        std::vector<cv::Point> clipMinBox;
        op::GetMinBoxes(clipBox, minSideLen, perimeter,clipMinBox);

        //---use clipper end---
        if(clipMinBox.size()<4 || minSideLen < minArea + 2)
            continue;

        // box坐标对应到padding图像中
        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / ratio_wh[0];
            // 避免越界
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), int(dst_resize.cols/ratio_wh[0]));

            clipMinBox[j].y = clipMinBox[j].y / ratio_wh[1];
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), int(dst_resize.rows/ratio_wh[1]));
        }
        rsBoxes.insert(rsBoxes.begin(),types::BoxInfo{clipMinBox, score});
    }
    
    //---------- getPartImages ----------
    // 获取resize后图像的box
    std::vector<cv::Mat> partImages = image::getPartImages(dst_resize, rsBoxes);
    double postprocessTime = utils::GetCurrentTime() - startTime;
    log->debug("Postprocess: [Get {} boxes(images), cost {:.5}s.]\n{}",rsBoxes.size(),postprocessTime,utils::GetBoxInfo(rsBoxes));

    //Save result.jpg
    // /isOutputResultImg
    // 输出图像文件
    // if (true) {
    //     printf("---------- step: drawTextBoxes ----------\n");
    //     // 在padding的图像上画框
    //     drawTextBoxes(dst_padding, rsBoxes, 5);
    //     //cropped to original size
    //     cv::Mat rgbBoxImg, textBoxImg;

    //     if (padding > 0) {
    //         dst_padding(cv::Rect(padding,padding,src.cols,src.rows)).copyTo(rgbBoxImg);
    //     } else {
    //         rgbBoxImg = dst_padding;
    //     }
    //     cvtColor(rgbBoxImg, textBoxImg, cv::COLOR_RGB2BGR);//convert to BGR for Output Result Img

    //     std::string resultImgFile = getResultImgFilePath(".", "image");
    //     imwrite(resultImgFile, textBoxImg);
    // }

    return partImages;
}