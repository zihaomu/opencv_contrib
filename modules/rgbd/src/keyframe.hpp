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

// It works like doubly linked list.
struct KeyFrame
{

    Mat DNNFeature;
    int submapID;

    int preKeyFrameID;
    int nextKeyFrameID;

    KeyFrame();
    KeyFrame(Mat _DNNfeature, int _submapID);
    KeyFrame(Mat _DNNfeature, int _submapID, int preKeyFrameID);

};

class KeyFrameDatabase
{
public:

    KeyFrameDatabase();

    KeyFrameDatabase(int maxSizeDB);

    ~KeyFrameDatabase() = default;

    void addKeyFrame( const Mat& DNNFeature, int frameID, int submapID);

    Ptr<KeyFrame> getKeyFrameByID(int keyFrameID);

    bool deleteKeyFrameByID(int keyFrameID);

    int getSize();

    bool empty();

    void reset();

    void shrinkDB();

    int getLastKeyFrameID();

    std::vector<int> getCandidateKF(const Mat& currentFeature, const double& similarityLow, double& bestSimilarity, int& bestId);

    double score(InputArray feature1, InputArray feature2);
    
    // Debug only
    void printDB();

private:
    // < keyFrameID, KeyFrame>
    std::map<int, Ptr<KeyFrame> > DataBase;

    int maxSizeDB;

    int lastKeyFrameID;


};

}// namespace large_kinfu
}// namespace cv

#endif
