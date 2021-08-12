// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_LOOP_CLOSURE_DETECTION_H__
#define __OPENCV_LOOP_CLOSURE_DETECTION_H__

#include "opencv2/dnn.hpp"
#include "keyframe.hpp"

namespace cv{
namespace large_kinfu{

class LoopClosureDetectionImpl : public LoopClosureDetection
{
public:
    LoopClosureDetectionImpl(const String& modelBin, const String& modelTxt, const Size& input_size, int backendId = 0, int targetId = 0);

    void addFrame(InputArray img, const int frameID, const int submapID, int& tarSubmapID, bool& ifLoop) CV_OVERRIDE;

    bool loopCheck(int& tarSubmapID);

    void reset() CV_OVERRIDE;

    bool processFrame(InputArray img, OutputArray DNNfeature);

    bool newFrameCheck();

    float score(InputArray feature1, InputArray feature2);

private:
    Ptr<KeyFrameDatabase> KFDataBase;
    Ptr<dnn::Net> net;
    Size inputSize;
    int currentFrameID;
    Mat currentFeature;
    Ptr<KeyFrame> bestLoopFrame;

    // Param: Only for DeepLCD
    int minDatabaseSize = 50;
    float similarityHigh = 0.94;
    float similarityLow = 0.92;

};

}
}
#endif
