/*
 * @Author: Petrichor
 * @Date: 2022-03-09 15:44:24
 * @LastEditTime: 2022-03-11 10:59:03
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\utils\operators.h
 * 版权声明
 */
#ifndef __OPERATORS_H__
#define __OPERATORS_H__

#include <opencv2/core.hpp>
#include <vector>
#include "types.h"

namespace op
{
    void SetPadding(cv::Mat &src, cv::Mat &dst, const int padding);
    void ResizeByMaxSide(const cv::Mat &src,cv::Mat &dst,const int max_side_threshold, std::vector<float> &ratio_wh);

    void Normalize(cv::Mat *im, std::vector<float> &mean, std::vector<float> &norm);
    void Permute(const cv::Mat *im, float *data);
    void MeanNormalize(cv::Mat &src, const float *mean, const float *norm,std::vector<float>& data);

    void GetMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize,std::vector<cv::Point>& minBoxVec);
    void BoxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox,float& score);
    void UnClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio,std::vector<cv::Point>& outBox);

    cv::Mat ResizeByValue(cv::Mat &src, int dstWidth, int dstHeight);
    std::vector<int> GetAngleIndexes(std::vector<types::AngleInfo> &angles);

    void ResizeBySize(const cv::Mat &src, cv::Mat &dst, int dstWidth, int dstHeight);

} // namespace op


#endif
