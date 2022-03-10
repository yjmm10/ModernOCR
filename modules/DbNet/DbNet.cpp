
#include "DbNet.h"
#include "utils/OcrUtils.h"
#include "utils/operators.h"
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
/*
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
*/
/*
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
*/

std::string DbNet::getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag) {
    std::string filePath;
    filePath.append(path).append(imgName).append(tag).append(std::to_string(i)).append(".jpg");
    return filePath;
}

std::vector<cv::Mat> DbNet::getPartImages(cv::Mat &src, std::vector<types::TextBox> &textBoxes) {
    std::vector<cv::Mat> partImages;
    for (int i = 0; i < textBoxes.size(); ++i) {
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
        // //OutPut DebugImg
        if (true) {
            std::string debugImgFile = getDebugImgFilePath("", "hh", i, "-part-");
            saveImg(partImg, debugImgFile.c_str());
        }
    }
    return partImages;
}

std::vector<cv::Mat>
DbNet::Run(cv::Mat &src, int padding, float boxScoreThresh, float boxThresh, float unClipRatio,int maxSideLen) {
    // preprocess
    // 边界填充之后增加获取比例
    //src 640 padding 740 resize 736
    cv::Mat dst_padding,dst_resize;
    op::SetPadding(src,dst_padding,padding);
    std::vector<float> ratio_wh;
    op::ResizeByMaxSide(dst_padding,dst_resize,maxSideLen,ratio_wh);
    
    //inference
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
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    //-----Data preparation-----
    cv::Mat fMapMat(dst_resize.rows, dst_resize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, size_t(outputCount * sizeof(float)));
    
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
    std::vector<types::TextBox> rsBoxes;
    rsBoxes.clear();

    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    log->info("Find {} Contours!",123);
    for (unsigned int i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox;
        op::GetMinBoxes(contours[i], minSideLen, perimeter,minBox);
        if (minSideLen < minArea)
            continue;
        float score;
        op::BoxScoreFast(fMapMat, contours[i],score);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox;
        op::UnClip(minBox, perimeter, unClipRatio, clipBox);
        std::vector<cv::Point> clipMinBox;
        op::GetMinBoxes(clipBox, minSideLen, perimeter,clipMinBox);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        // box坐标对应到padding图像中
        for (unsigned int j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = clipMinBox[j].x / ratio_wh[0];
            // 避免越界
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), int(dst_resize.cols/ratio_wh[0]));

            clipMinBox[j].y = clipMinBox[j].y / ratio_wh[1];
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), int(dst_resize.rows/ratio_wh[1]));
        }
        rsBoxes.insert(rsBoxes.begin(),types::TextBox{clipMinBox, score});
    }
   
    // 输出所有的数据
    // for (int i = 0; i < rsBoxes.size(); ++i) {
    //     printf("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
    //            rsBoxes[i].score,
    //            rsBoxes[i].boxPoint[0].x, rsBoxes[i].boxPoint[0].y,
    //            rsBoxes[i].boxPoint[1].x, rsBoxes[i].boxPoint[1].y,
    //            rsBoxes[i].boxPoint[2].x, rsBoxes[i].boxPoint[2].y,
    //            rsBoxes[i].boxPoint[3].x, rsBoxes[i].boxPoint[3].y);
    // }



    //---------- getPartImages ----------
    // 获取resize后图像的box
    std::vector<cv::Mat> partImages = getPartImages(dst_resize, rsBoxes);


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