/*
 * @Author: Petrichor
 * @Date: 2022-03-11 11:40:37
 * @LastEditTime: 2022-03-11 17:21:28
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\core\utils.cpp
 * 版权声明
 */

#include "utils.h"

namespace ModernOCR {
    namespace utils{


    };
    
    namespace image{
        cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<cv::Point> box) {
            cv::Mat image;
            src.copyTo(image);
            std::vector<cv::Point> points = box;

            int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
            int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
            int left = int(*std::min_element(collectX, collectX + 4));
            int right = int(*std::max_element(collectX, collectX + 4));
            int top = int(*std::min_element(collectY, collectY + 4));
            int bottom = int(*std::max_element(collectY, collectY + 4));

            cv::Mat imgCrop;
            image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

            for (int i = 0; i < points.size(); i++) {
                points[i].x -= left;
                points[i].y -= top;
            }

            int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                        pow(points[0].y - points[1].y, 2)));
            int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                        pow(points[0].y - points[3].y, 2)));

            cv::Point2f ptsDst[4];
            ptsDst[0] = cv::Point2f(0., 0.);
            ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
            ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
            ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

            cv::Point2f ptsSrc[4];
            ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
            ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
            ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
            ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

            cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

            cv::Mat partImg;
            cv::warpPerspective(imgCrop, partImg, M,
                                cv::Size(imgCropWidth, imgCropHeight),
                                cv::BORDER_REPLICATE);

            if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
                cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
                cv::transpose(partImg, srcCopy);
                cv::flip(srcCopy, srcCopy, 0);
                return srcCopy;
            } else {
                return partImg;
            }
        }

        std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<types::BoxInfo> &textBoxes) {
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

    };


    namespace str{

    };
};

