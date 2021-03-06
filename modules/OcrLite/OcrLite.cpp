#include "OcrLite.h"
#include <stdarg.h> //windows&linux
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <string>

OcrLite::OcrLite() {}

OcrLite::~OcrLite() {
    if (isOutputResultTxt) {
        fclose(resultTxt);
    }
}

void OcrLite::setNumThread(int numOfThread) {
    dbNet.setNumThread(numOfThread);
    angleNet.setNumThread(numOfThread);
    crnnNet.setNumThread(numOfThread);
}

void OcrLite::initLogger(bool isConsole, bool isPartImg, bool isResultImg) {
    isOutputConsole = isConsole;
    isOutputPartImg = isPartImg;
    isOutputResultImg = isResultImg;
}

void OcrLite::enableResultTxt(const char *path, const char *imgName) {
    isOutputResultTxt = true;
    std::string resultTxtPath = image::getResultTxtFilePath(path, imgName);
    printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(resultTxtPath.c_str(), "w");
}

bool OcrLite::initModels(const std::string &detPath, const std::string &clsPath,
                         const std::string &recPath, const std::string &keysPath) {
    Logger("=====Init Models=====\n");
    Logger("--- Init DbNet ---\n");
    dbNet.LoadModel(detPath);

    Logger("--- Init AngleNet ---\n");
    angleNet.LoadModel(clsPath);

    Logger("--- Init CrnnNet ---\n");
    crnnNet.LoadModel(recPath, keysPath);

    Logger("Init Models Success!\n");
    return true;
}

void OcrLite::Logger(const char *format, ...) {
    if (!(isOutputConsole || isOutputResultTxt)) return;
    char *buffer = (char *) malloc(8192);
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    if (isOutputConsole) printf("%s", buffer);
    if (isOutputResultTxt) fprintf(resultTxt, "%s", buffer);
    free(buffer);
}

cv::Mat makePadding(cv::Mat &src, const int padding) {
    if (padding <= 0) return src;
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

/*
OcrResult OcrLite::detect(const char *path, const char *imgName,
                          const int padding, const int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    std::string imgFile = getSrcImgFilePath(path, imgName);

    cv::Mat bgrSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR
    cv::Mat originSrc;
    cvtColor(bgrSrc, originSrc, cv::COLOR_BGR2RGB);// convert to RGB

    // ????????????????????????????????????
    cv::Mat dst_padding,dst_resize;
    op::SetPadding(originSrc,dst_padding,padding);
    std::vector<float> ratio_wh;
    op::ResizeByMaxSide(dst_padding,dst_resize,maxSideLen,ratio_wh);
    

    // int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    // int resize;
    // if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
    //     resize = originMaxSide;
    // } else {
    //     resize = maxSideLen;
    // }
    // resize += 2*padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    // Logger("rect x %d y %d",paddingRect.x,paddingRect.y);
    // double startTime = getCurrentTime();


    // double dbNetTime = getCurrentTime() - startTime;
    // Logger("dbNetTime(%fms)\n", dbNetTime);
    
    
    // startTime = getCurrentTime();
    
    // cv::Mat paddingSrc = makePadding(originSrc, padding);
    
    // dbNetTime = getCurrentTime() - startTime;
    // Logger("dbNetTime(%fms)\n", dbNetTime);

    // // if(std::memcmp(dst.data, paddingSrc.data, dst.total()*dst.elemSize())==0)
    // //     Logger("????????????????????????\n");
    // ScaleParam scale = getScaleParam(paddingSrc, resize);
    cv::Mat paddingSrc  = dst_padding;
    ScaleParam scale;
    scale.srcWidth = paddingSrc.cols;
    scale.srcHeight=paddingSrc.rows;
    scale.dstWidth=dst_resize.cols;
    scale.dstHeight=dst_resize.rows;
    scale.ratioWidth = ratio_wh[0];
    scale.ratioHeight = ratio_wh[0];
    OcrResult result;
    result = detect(path, imgName, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}
*/

/*
OcrResult OcrLite::detect(const cv::Mat& mat, int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle)
{
    cv::Mat originSrc;
    cvtColor(mat, originSrc, cv::COLOR_BGR2RGB);// convert to RGB
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    }
    else {
        resize = maxSideLen;
    }
    resize += 2 * padding;
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    OcrResult result;
    result = detect(NULL, NULL, paddingSrc, paddingRect, scale,
        boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

*/

// std::vector<cv::Mat> OcrLite::getPartImages(cv::Mat &src, std::vector<types::BoxInfo> &textBoxes,
//                                             const char *path, const char *imgName) {
//     std::vector<cv::Mat> partImages;
//     for (int i = 0; i < textBoxes.size(); ++i) {
//         cv::Mat partImg;
//         image::getRotateCropImage(src, textBoxes[i].boxPoint,partImg);
//         partImages.emplace_back(partImg);
//         //OutPut DebugImg
//         if (isOutputPartImg) {
//             std::string debugImgFile = image::getDebugImgFilePath(path, imgName, i, "-part-");
//             image::saveImg(partImg, debugImgFile.c_str());
//         }
//     }
//     return partImages;
// }

/*
OcrResult OcrLite::detect(const char *path, const char *imgName,
                          cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {

    cv::Mat textBoxPaddingImg = src.clone();
    int thickness = getThickness(src);

    Logger("=====Start detect=====\n");
    Logger("??????ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n", scale.srcWidth, scale.srcHeight,
           scale.dstWidth, scale.dstHeight,
           scale.ratioWidth, scale.ratioHeight);

    Logger("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)\n", dbNetTime);

    for (int i = 0; i < textBoxes.size(); ++i) {
        Logger("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
               textBoxes[i].score,
               textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
               textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
               textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
               textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    }

    Logger("---------- step: drawTextBoxes ----------\n");
    drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    //---------- getPartImages ----------
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

    Logger("---------- step: angleNet getAngles ----------\n");
    std::vector<Angle> angles;
    angles = angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

    //Log Angles
    for (int i = 0; i < angles.size(); ++i) {
        Logger("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
    }

    //Rotate partImgs
    for (int i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 0) {
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    Logger("---------- step: crnnNet getTextLine ----------\n");
    std::vector<TextLine> textLines = crnnNet.getTextLines(partImages, path, imgName);
    //Log TextLines
    for (int i = 0; i < textLines.size(); ++i) {
        Logger("textLine[%d](%s)\n", i, textLines[i].text.c_str());
        std::ostringstream txtScores;
        for (int s = 0; s < textLines[i].charScores.size(); ++s) {
            if (s == 0) {
                txtScores << textLines[i].charScores[s];
            } else {
                txtScores << " ," << textLines[i].charScores[s];
            }
        }
        Logger("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
        Logger("crnnTime[%d](%fms)\n", i, textLines[i].time);
    }

    std::vector<TextBlock> textBlocks;
    for (int i = 0; i < textLines.size(); ++i) {
        std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        int padding = originRect.x;//padding conversion
        boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
        boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
        boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
        boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
        TextBlock textBlock{boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
                            angles[i].time, textLines[i].text, textLines[i].charScores, textLines[i].time,
                            angles[i].time + textLines[i].time};
        textBlocks.emplace_back(textBlock);
    }

    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End detect=====\n");
    Logger("FullDetectTime(%fms)\n", fullTime);

    //cropped to original size
    cv::Mat rgbBoxImg, textBoxImg;

    if (originRect.x > 0 && originRect.y > 0) {
        textBoxPaddingImg(originRect).copyTo(rgbBoxImg);
    } else {
        rgbBoxImg = textBoxPaddingImg;
    }
    cvtColor(rgbBoxImg, textBoxImg, cv::COLOR_RGB2BGR);//convert to BGR for Output Result Img

    //Save result.jpg
    if (isOutputResultImg) {
        std::string resultImgFile = getResultImgFilePath(path, imgName);
        imwrite(resultImgFile, textBoxImg);
    }

    std::string strRes;
    for (int i = 0; i < textBlocks.size(); ++i) {
        strRes.append(textBlocks[i].text);
        strRes.append("\n");
    }

    return OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
}
*/

types::OcrResult OcrLite::detect_new(const char *path, const char *imgName,
                          const int padding, const int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    std::string imgFile = image::getSrcImgFilePath(path, imgName);

    cv::Mat bgrSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR
    cv::Mat originSrc;
    cvtColor(bgrSrc, originSrc, cv::COLOR_BGR2RGB);// convert to RGB

    types::OcrResult result;

    cv::Mat dst_resize;
    originSrc.copyTo(dst_resize);
    Logger("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = utils::GetCurrentTime();
    // std::vector<TextBox> textBoxes 
    types::DbNetParam dbnetParam;
    dbnetParam.padding=padding; 
    dbnetParam.boxScoreThresh=boxScoreThresh;
    dbnetParam.boxThresh=boxThresh;
    dbnetParam.unClipRatio=unClipRatio;
    dbnetParam.maxSideLen=maxSideLen;
    // dbNet.Run(dst_resize, padding, boxScoreThresh, boxThresh, unClipRatio,maxSideLen);
    std::vector<cv::Mat> partImages = dbNet.Run(dst_resize, dbnetParam);
    double endDbNetTime = utils::GetCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    Logger("dbNetTime(%fms)\n", dbNetTime);


    Logger("---------- step: angleNet getAngles ----------\n");
    std::vector<types::AngleInfo> angles;
    angles = angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

    //Log Angles
    for (int i = 0; i < angles.size(); ++i) {
        Logger("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
    }

    //Rotate partImgs
    for (int i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 0) {
            partImages.at(i) = image::matRotateClockWise180(partImages[i]);
        }
    }

    Logger("---------- step: crnnNet getTextLine ----------\n");
    std::vector<types::RecInfo> textLines = crnnNet.getTextLines(partImages, path, imgName);
    //Log TextLines
    for (int i = 0; i < textLines.size(); ++i) {
        Logger("textLine[%d](%s)\n", i, textLines[i].text.c_str());
        std::ostringstream txtScores;
        for (int s = 0; s < textLines[i].charScores.size(); ++s) {
            if (s == 0) {
                txtScores << textLines[i].charScores[s];
            } else {
                txtScores << " ," << textLines[i].charScores[s];
            }
        }
        Logger("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
        Logger("crnnTime[%d](%fms)\n", i, textLines[i].time);
    }

    std::vector<types::TextBlock> textBlocks;
    // for (int i = 0; i < textLines.size(); ++i) {
    //     std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
        
    //     boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
    //     boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
    //     boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
    //     boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
    //     TextBlock textBlock{boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
    //                         angles[i].time, textLines[i].text, textLines[i].charScores, textLines[i].time,
    //                         angles[i].time + textLines[i].time};
    //     textBlocks.emplace_back(textBlock);
    // }

    double endTime = utils::GetCurrentTime();
    double fullTime = endTime - startTime;
    Logger("=====End detect=====\n");
    Logger("FullDetectTime(%fms)\n", fullTime);

    // //cropped to original size
    cv::Mat rgbBoxImg, textBoxImg;

    // if (padding > 0) {
    //     // dst_resize(paddingRect).copyTo(rgbBoxImg);
    //     // std::cout<<paddingRect.x<<" "<<paddingRect.x<<" "<<paddingRect.width<<" "<<paddingRect.height<<std::endl;
    //     // std::cout<<paddingRect.x<<" "<<paddingRect.x<<" "<<int(dst_resize.cols/ratio_wh[0]-2*padding)<<" "<<int(dst_resize.rows/ratio_wh[1]-2*padding)<<std::endl;
    //     dst_resize(cv::Rect(padding,padding,int(dst_resize.cols/ratio_wh[0]-2*padding),int(dst_resize.rows/ratio_wh[1]-2*padding))).copyTo(rgbBoxImg);
    //     // rgbBoxImg(cv::Rect(padding,padding,int(dst_resize.cols/ratio_wh[0]-2*padding),int(dst_resize.rows/ratio_wh[1]-2*padding)));
    // } else {
    //     rgbBoxImg = dst_resize;
    // }
    // cvtColor(dst_resize, textBoxImg, cv::COLOR_RGB2BGR);//convert to BGR for Output Result Img

    // //Save result.jpg
    // if (isOutputResultImg) {
    //     std::string resultImgFile = getResultImgFilePath(path, imgName);
    //     imwrite(resultImgFile, textBoxImg);
    // }

    std::string strRes;
    // for (int i = 0; i < textBlocks.size(); ++i) {
    //     strRes.append(textBlocks[i].text);
    //     strRes.append("\n");
    // }

    return types::OcrResult{dbNetTime, textBlocks, textBoxImg, fullTime, strRes};
    // return result;
}