/*
 * @Author: Petrichor
 * @Date: 2022-03-09 15:36:06
 * @LastEditTime: 2022-03-11 10:58:55
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\utils\operators.cpp
 * 版权声明
 */

// #include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "operators.h"
#include "clipper.hpp"
namespace op
{
    void SetPadding(cv::Mat &src, cv::Mat &dst, const int padding) {
        /**
         * @Description:  图像四边添加空白边

         * @Args: 
                src:    源图像
                padding:空白大小

         * @Returns:    目标图像
         */        
        if (padding <= 0)
            src.copyTo(dst); 
        else
            cv::copyMakeBorder(src, dst, padding, padding, padding, padding, cv::BORDER_ISOLATED, cv::Scalar(255, 255, 255));
    }

    void ResizeByMaxSide(const cv::Mat &src,cv::Mat &dst,const int max_side_threshold, std::vector<float> &ratio_wh){
        auto src_w = src.cols;
        auto src_h = src.rows;

        auto ratio = 1.f;
        auto max_side = src_w >= src_h ? src_w : src_h;
        if(max_side > max_side_threshold)
            ratio = src_h > src_w ? float(max_side_threshold/src_h) : float(max_side_threshold/src_w);
        auto dst_h = std::max(int(round(float(src_h*ratio) / 32) * 32), 32);
        auto dst_w = std::max(int(round(float(src_w*ratio) / 32) * 32), 32);
        cv::resize(src,dst,cv::Size(dst_w,dst_h));
        ratio_wh.push_back(float(dst_w)/float(src_w));
        ratio_wh.push_back(float(dst_h)/float(src_h));
    }

    void Normalize(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &norm) {
        double e = 1.0 / 255.0;
        (*im).convertTo(*im, CV_32FC3, e);
        std::vector<cv::Mat> bgr_channels(3);
        cv::split(*im, bgr_channels);
        for (auto i = 0; i < bgr_channels.size(); i++) {
            bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * norm[i],
                                    (0.0 - mean[i]) * norm[i]);
        }
        cv::merge(bgr_channels, *im);
    }
    
    void Permute(const cv::Mat *im, float *data) {
        int rh = im->rows;
        int rw = im->cols;
        int rc = im->channels();
        for (int i = 0; i < rc; ++i) {
            cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
        }
    }

    void MeanNormalize(cv::Mat &src, const float *mean, const float *norm,std::vector<float>& inputTensorValues) {
        /**
         * @Description: 正则化数据

         * @Args: 
                src:                图像
                mean:               通道均值
                norm:               通道方差
                inputTensorValues:  输出数据

         * @Returns:  NULL
         */   
        auto inputTensorSize = src.cols * src.rows * src.channels();
        inputTensorValues.resize(inputTensorSize);
        size_t numChannels = src.channels();
        size_t imageSize = src.cols * src.rows;

        for (size_t pid = 0; pid < imageSize; pid++) {
            for (size_t ch = 0; ch < numChannels; ++ch) {
                float data = (float) (src.data[pid * numChannels + ch] * norm[ch] - mean[ch] * norm[ch]);
                inputTensorValues[ch * imageSize + pid] = data;
            }
        }
    }

    void GetMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize,std::vector<cv::Point>& minBoxVec) {
        cv::RotatedRect textRect = cv::minAreaRect(inVec);
        cv::Mat boxPoints2f;
        cv::boxPoints(textRect, boxPoints2f);

        float *p1 = (float *) boxPoints2f.data;
        std::vector<cv::Point> tmpVec;
        for (int i = 0; i < 4; ++i, p1 += 2) {
            tmpVec.emplace_back(int(p1[0]), int(p1[1]));
        }

        std::sort(tmpVec.begin(), tmpVec.end(), [](const cv::Point &a, const cv::Point &b) -> bool { return a.x < b.x; });

        minBoxVec.clear();

        int index1, index2, index3, index4;
        if (tmpVec[1].y > tmpVec[0].y) {
            index1 = 0;
            index4 = 1;
        } else {
            index1 = 1;
            index4 = 0;
        }

        if (tmpVec[3].y > tmpVec[2].y) {
            index2 = 2;
            index3 = 3;
        } else {
            index2 = 3;
            index3 = 2;
        }

        minBoxVec.clear();

        minBoxVec.push_back(tmpVec[index1]);
        minBoxVec.push_back(tmpVec[index2]);
        minBoxVec.push_back(tmpVec[index3]);
        minBoxVec.push_back(tmpVec[index4]);

        minSideLen = (std::min)(textRect.size.width, textRect.size.height);
        allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

        
    }
    void BoxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox,float& score) {
        std::vector<cv::Point> box = inBox;
        int width = inMat.cols;
        int height = inMat.rows;
        int maxX = -1, minX = 1000000, maxY = -1, minY = 1000000;
        for (int i = 0; i < box.size(); ++i) {
            if (maxX < box[i].x)
                maxX = box[i].x;
            if (minX > box[i].x)
                minX = box[i].x;
            if (maxY < box[i].y)
                maxY = box[i].y;
            if (minY > box[i].y)
                minY = box[i].y;
        }
        maxX = (std::min)((std::max)(maxX, 0), width - 1);
        minX = (std::max)((std::min)(minX, width - 1), 0);
        maxY = (std::min)((std::max)(maxY, 0), height - 1);
        minY = (std::max)((std::min)(minY, height - 1), 0);

        for (int i = 0; i < box.size(); ++i) {
            box[i].x = box[i].x - minX;
            box[i].y = box[i].y - minY;
        }

        std::vector<std::vector<cv::Point>> maskBox;
        maskBox.push_back(box);
        cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
        cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);

        score = cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                        maskMat).val[0];
    }

    void UnClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio,std::vector<cv::Point>& outBox) {
        ClipperLib::Path poly;

        for (int i = 0; i < inBox.size(); ++i) {
            poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
        }

        double distance = unClipRatio * ClipperLib::Area(poly) / (double) perimeter;

        ClipperLib::ClipperOffset clipperOffset;
        clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
        ClipperLib::Paths polys;
        polys.push_back(poly);
        clipperOffset.Execute(polys, distance);

        outBox.clear();
        std::vector<cv::Point> rsVec;
        for (int i = 0; i < polys.size(); ++i) {
            ClipperLib::Path tmpPoly = polys[i];
            for (int j = 0; j < tmpPoly.size(); ++j) {
                outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
            }
        }
    }


    cv::Mat ResizeByValue(cv::Mat &src, int dstWidth, int dstHeight) {
        cv::Mat srcResize;
        float scale = (float) dstHeight / (float) src.rows;
        int angleWidth = int((float) src.cols * scale);
        cv::resize(src, srcResize, cv::Size(angleWidth, dstHeight));
        cv::Mat srcFit = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(255, 255, 255));
        if (angleWidth < dstWidth) {
            cv::Rect rect(0, 0, srcResize.cols, srcResize.rows);
            srcResize.copyTo(srcFit(rect));
        } else {
            cv::Rect rect(0, 0, dstWidth, dstHeight);
            srcResize(rect).copyTo(srcFit);
        }
        return srcFit;
    }

    void ResizeBySize(const cv::Mat &src, cv::Mat &dst, int dstWidth, int dstHeight){
        cv::resize(src, dst, cv::Size(dstWidth, dstHeight));
    }

    std::vector<int> GetAngleIndexes(std::vector<types::AngleInfo> &angles) {
        std::vector<int> angleIndexes;
        angleIndexes.reserve(angles.size());
        for (int i = 0; i < angles.size(); ++i) {
            angleIndexes.push_back(angles[i].index);
        }
        return angleIndexes;
    }
} // namespace op
