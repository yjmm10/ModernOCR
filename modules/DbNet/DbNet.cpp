#include <numeric>
#include "DbNet.h"

#include "core/modernocr.h"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using namespace spdlog;
using namespace ModernOCR;
DbNet::DbNet() {
    // const std::string modelName("DbNet");
    log = spdlog::get(_modelName);
    if(log==nullptr)
    {
        log = spdlog::basic_logger_mt(_modelName, "logs/ModernOCR.log");
    }
    log->info("{} Version: {} !", _modelName, _version);
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

std::vector<cv::Mat> DbNet::Run(const std::string imgPath,types::DbNetParam newParam){
    cv::Mat bgrSrc = cv::imread(imgPath, cv::IMREAD_COLOR);//default : BGR
    cv::Mat originSrc;
    cvtColor(bgrSrc, originSrc, cv::COLOR_BGR2RGB);// convert to RGB

    types::OcrResult result;

    cv::Mat src;
    originSrc.copyTo(src);
    // std::vector<cv::Mat> Run(cv::Mat &src, types::DbNetParam newParams)
}

std::vector<cv::Mat>
DbNet::Run(cv::Mat &src, types::DbNetParam newParams){
    // ????????????
    if((params==types::DbNetParam{0,0.0f,0.0f,0.0f,0}) || (update && !(params==newParams))){
        params=newParams;
        log->info("Update {} params: [padding: {}, boxScoreThresh: {}, boxThresh: {}, unClipRatio: {}, maxSideLen: {}.]",_modelName, params.padding, params.boxScoreThresh, params.boxThresh, params.unClipRatio, params.maxSideLen);
    }
    // ??????dbnet??????
    return Run(src, params.padding, params.boxScoreThresh, params.boxThresh, params.unClipRatio, params.maxSideLen);
}

std::vector<cv::Mat> DbNet::Run(cv::Mat &src, 
        int padding, float boxScoreThresh, float boxThresh, float unClipRatio,int maxSideLen) {
    double dbStartTime = utils::GetCurrentTime();
    // preprocess
    // ????????????????????????????????????
    //src 640 padding 740 resize 736
    cv::Mat dst_padding,dst_resize;
    op::SetPadding(src,dst_padding,padding);
    std::vector<float> ratio_wh;
    op::ResizeByMaxSide(dst_padding,dst_resize,maxSideLen,ratio_wh);
    double preprocessTime = utils::GetCurrentTime() - dbStartTime;
    log->debug("Preprocess: [src_wh: ({},{}), padding_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
                src.cols,src.rows,dst_padding.cols,dst_padding.rows,dst_resize.cols,dst_resize.rows, preprocessTime);
        
    double inferenceTime = utils::GetCurrentTime();
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
    
    inferenceTime = utils::GetCurrentTime() - inferenceTime;
    log->debug("Inference: [cost {:.5} s].", inferenceTime);
    double postprocessTime = utils::GetCurrentTime();

    //-----Data preparation-----
    cv::Mat fMapMat(dst_resize.rows, dst_resize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));
    
    assert(dst_resize.data());
    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    //postprocess
    //  ???????????????????????????????????????,????????????
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat dilation_map;
    cv::morphologyEx(norfMapMat, dilation_map, cv::MORPH_CLOSE, dila_ele);
    dilation_map.copyTo(norfMapMat);

    
    float minArea = 3;
    // resize????????????????????????
    detResult.allBoxInfo.clear();

    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    log->debug("Postprocess: [find {} Contours, cost {:.5}s.]",contours.size(),utils::GetCurrentTime() - postprocessTime);
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

        // box???????????????padding?????????
        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / ratio_wh[0];
            // ????????????
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), int(dst_resize.cols/ratio_wh[0]));

            clipMinBox[j].y = clipMinBox[j].y / ratio_wh[1];
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), int(dst_resize.rows/ratio_wh[1]));
        }
        detResult.allBoxInfo.insert(detResult.allBoxInfo.begin(),types::BoxInfo{clipMinBox, score});
        
    }
    postprocessTime = utils::GetCurrentTime() - dbStartTime;
    log->debug("Postprocess: [Get {} boxes, cost {:.5}s.]\n{}",detResult.allBoxInfo.size(),postprocessTime,utils::GetBoxInfo(detResult.allBoxInfo));
    
    double cropTime = utils::GetCurrentTime();
    //---------- getPartImages ----------
    // ??????resize????????????box
    std::vector<cv::Mat> partImages = image::getPartImages(dst_resize, detResult.allBoxInfo);
    cropTime = utils::GetCurrentTime()- cropTime;
    log->debug("Crop Image: [Get {} images, cost {:.5}s.]\n{}",partImages.size(),cropTime,utils::GetBoxInfo(detResult.allBoxInfo));
    
    detResult.time = utils::GetCurrentTime() - dbStartTime;
    allTimes = std::vector<double>{preprocessTime,inferenceTime,postprocessTime,cropTime};
    //Save result.jpg
    // /isOutputResultImg
    // ??????????????????
    // if (true) {
    //     printf("---------- step: drawTextBoxes ----------\n");
    //     // ???padding??????????????????
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


/*
std::vector<cv::Mat>
DbNet::Run(cv::Mat &src, int padding, float boxScoreThresh, float boxThresh, float unClipRatio,int maxSideLen) {
    double startTime = utils::GetCurrentTime();
    // preprocess
    // ????????????????????????????????????
    //src 640 padding 740 resize 736
    cv::Mat dst_padding,dst_resize;
    op::SetPadding(src,dst_padding,padding);
    std::vector<float> ratio_wh;
    op::ResizeByMaxSide(dst_padding,dst_resize,maxSideLen,ratio_wh);
    double preprocessTime = utils::GetCurrentTime() - startTime;
    log->debug("Preprocess: [src_wh: ({},{}), padding_wh: ({},{}), resize_wh: ({},{}), cost {:.5} s].",
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
    log->debug("Inference: [cost {:.5} s].", inferenceTime);
    startTime = utils::GetCurrentTime();

    //-----Data preparation-----
    cv::Mat fMapMat(dst_resize.rows, dst_resize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));
    
    assert(dst_resize.data());
    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    //postprocess
    //  ???????????????????????????????????????,????????????
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat dilation_map;
    cv::morphologyEx(norfMapMat, dilation_map, cv::MORPH_CLOSE, dila_ele);
    dilation_map.copyTo(norfMapMat);

    
    float minArea = 3;
    // resize????????????????????????
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

        // box???????????????padding?????????
        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / ratio_wh[0];
            // ????????????
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), int(dst_resize.cols/ratio_wh[0]));

            clipMinBox[j].y = clipMinBox[j].y / ratio_wh[1];
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), int(dst_resize.rows/ratio_wh[1]));
        }
        rsBoxes.insert(rsBoxes.begin(),types::BoxInfo{clipMinBox, score});
    }
    
    //---------- getPartImages ----------
    // ??????resize????????????box
    std::vector<cv::Mat> partImages = image::getPartImages(dst_resize, rsBoxes);
    double postprocessTime = utils::GetCurrentTime() - startTime;
    log->debug("Postprocess: [Get {} boxes(images), cost {:.5}s.]\n{}",rsBoxes.size(),postprocessTime,utils::GetBoxInfo(rsBoxes));

    //Save result.jpg
    // /isOutputResultImg
    // ??????????????????
    // if (true) {
    //     printf("---------- step: drawTextBoxes ----------\n");
    //     // ???padding??????????????????
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

*/