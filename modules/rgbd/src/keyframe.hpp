// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_RGBD_KEYFRAME_HPP__
#define __OPENCV_RGBD_KEYFRAME_HPP__

#include "opencv2/core.hpp"

namespace cv
{
namespace large_kinfu
{

struct KeyFrame
{

    Mat DNNFeature;
    int submapID;

    KeyFrame();

    KeyFrame(Mat _DNNfeature, int _submapID);
};

class KeyFrameDatabase
{
public:

    KeyFrameDatabase();

    ~KeyFrameDatabase() = default;

    void addKeyFrame( const Mat& DNNFeature, int frameID, int submapID);

    Ptr<KeyFrame> getKeyFrameByID(int keyFrameID);

    bool deleteKeyFrame(int keyFrameID);

    Ptr<KeyFrame> getKeyFrameByIndex(int index);

    int getSize();

    bool empty();

    void reset();

private:
    // < keyFrameID, KeyFrame>
    std::map<int, Ptr<KeyFrame>> DataBase;
};

}// namespace large_kinfu
}// namespace cv

#endif
