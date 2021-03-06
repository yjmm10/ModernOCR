/*
 * @Author: Petrichor
 * @Date: 2022-03-11 11:42:06
 * @LastEditTime: 2022-03-12 11:36:51
 * @LastEditors: Petrichor
 * @Description:  
 * @FilePath: \ModernOCR\modules\core\utils.h
 * 版权声明
 */

#ifndef __UTILS_H__
#define __UTILS_H__

// #include <opencv2/core.hpp>
#include "types.h"
namespace ModernOCR {
    namespace utils
    {                 
        inline double GetCurrentTime() {
            return (static_cast<double>(cv::getTickCount())) / cv::getTickFrequency() * 1000;//单位毫秒
        };
        inline bool isFileExists(const std::string &name) {
            struct stat buffer;
            return (stat(name.c_str(), &buffer) == 0);
        };
        inline std::string GetBoxInfo(std::vector<types::BoxInfo> rsBoxes) {
            std::string result="";
            for (int i = 0; i < rsBoxes.size(); ++i) {
                char info[1024];
                std::sprintf(info,"BoxInfo(after padding)[%d][score:%.3f, pos:(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
                   rsBoxes[i].score,
                   rsBoxes[i].boxPoint[0].x, rsBoxes[i].boxPoint[0].y,
                   rsBoxes[i].boxPoint[1].x, rsBoxes[i].boxPoint[1].y,
                   rsBoxes[i].boxPoint[2].x, rsBoxes[i].boxPoint[2].y,
                   rsBoxes[i].boxPoint[3].x, rsBoxes[i].boxPoint[3].y);
                result +=std::string(info);
            }
            return result;
        }

    } // namespace common
    
    
    namespace image{
        inline void saveImg(cv::Mat &img, const char *imgPath) {
            cv::imwrite(imgPath, img);
        }

        inline std::string getSrcImgFilePath(const char *path, const char *imgName) {
            std::string filePath;
            filePath.append(path).append(imgName);
            return filePath;
        }

        inline std::string getResultTxtFilePath(const char *path, const char *imgName) {
            std::string filePath;
            filePath.append(path).append(imgName).append("-result.txt");
            return filePath;
        }

        inline std::string getResultImgFilePath(const char *path, const char *imgName) {
            std::string filePath;
            filePath.append(path).append(imgName).append("-result.jpg");
            return filePath;
        }

        inline std::string getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag) {
            std::string filePath;
            filePath.append(path).append(imgName).append(tag).append(std::to_string(i)).append(".jpg");
            return filePath;
        }

        inline cv::Mat matRotateClockWise180(cv::Mat src) {
            flip(src, src, 0);
            flip(src, src, 1);
            return src;
        }

        inline cv::Mat matRotateClockWise90(cv::Mat src) {
            transpose(src, src);
            flip(src, src, 1);
            return src;
        }

        cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<cv::Point> box);

        std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<types::BoxInfo> &textBoxes);


    };  
    namespace str{
        inline std::wstring strToWstr(std::string str) {
            if (str.length() == 0)
                return L"";
            std::wstring wstr;
            wstr.assign(str.begin(), str.end());
            return wstr;
        };
    };
};


#endif

