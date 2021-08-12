// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "keyframe.hpp"

namespace cv
{
namespace large_kinfu
{

KeyFrame::KeyFrame()
{
    submapID = -1 ;
}

KeyFrame::KeyFrame(Mat _DNNfeature, int _submapID) : DNNFeature(_DNNfeature), submapID(_submapID)
{};

void KeyFrameDatabase::addKeyFrame(const Mat& DNNFeature, int frameID, int submapID)
{
    DataBase[frameID] = makePtr<KeyFrame>(DNNFeature, submapID);
}

bool KeyFrameDatabase::deleteKeyFrame(int keyFrameID)
{
    auto keyFrame = DataBase.find(keyFrameID);
    if(keyFrame == DataBase.end())
    {
        return false;
    } else{
        DataBase.erase(keyFrame);
        return true;
    }
}

Ptr<KeyFrame> KeyFrameDatabase::getKeyFrameByID(int keyFrameID)
{
    auto keyFrame = DataBase.find(keyFrameID);
    if(keyFrame == DataBase.end())
    {
        return {};
    } else
    {
        return keyFrame->second;
    }
}

Ptr<KeyFrame> KeyFrameDatabase::getKeyFrameByIndex(int index)
{
    if(index > this->getSize())
    {
        return {};
    } else{
        return DataBase[index];
    }
}

int KeyFrameDatabase::getSize()
{
    return DataBase.size();
}

bool KeyFrameDatabase::empty()
{
    return DataBase.empty();
}

void KeyFrameDatabase::reset()
{
    DataBase.clear();
}

}// namespace large_kinfu
}// namespace cv
