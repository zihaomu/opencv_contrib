// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "loop_closure_detection.hpp"
//#include "keyframe.cpp"

namespace cv{
namespace large_kinfu{

LoopClosureDetectionImpl::LoopClosureDetectionImpl(const String& _modelBin, const String& _modelTxt, const Size& _inputSize, int _backendId, int _targetId)
{
    inputSize = _inputSize;
    CV_Assert(!_modelBin.empty());
    if(_modelTxt.empty())
    {
        net = makePtr<dnn::Net>(dnn::readNet(_modelBin));
    } else{
        net = makePtr<dnn::Net>(dnn::readNet(_modelBin, _modelTxt));
    }

    net->setPreferableBackend(_backendId);
    net->setPreferableTarget(_targetId);
}

bool LoopClosureDetectionImpl::loopCheck(int& tarSubmapID)
{
    //Calculate the similarity with all pictures in the database.

    // If the KFDataBase is too small, then skip.
    if(KFDataBase->getSize() < minDatabaseSize )
        return false;

    float maxScore = 0;
    int count = 0;
    int bestId = 0;
    int DBsize = KFDataBase->getSize();

    // Traverse the database
    for(int i = 0; i < DBsize; i++)
    {
        Ptr<KeyFrame> DBkeyFrame = KFDataBase->getKeyFrameByIndex(i);
        float similarity = score(currentFeature, DBkeyFrame->DNNFeature);
        if(similarity > maxScore)
        {
            maxScore = similarity;
            bestId = i;
        }
        if(similarity > similarityLow)
        {
            count++;
        }
    }

    if(maxScore < similarityHigh || count > 3)
        return false;

    bestLoopFrame = KFDataBase->getKeyFrameByIndex(bestId);

    // find target submap ID
    if(bestLoopFrame->submapID == -1)
        return false;
    else
    {
        tarSubmapID = bestLoopFrame->submapID;
        return true;
    }
}

void LoopClosureDetectionImpl::addFrame(InputArray _img, const int frameID, const int submapID, int& tarSubmapID, bool& ifLoop)
{
    CV_Assert(!_img.empty());
    currentFrameID = frameID;

    Mat img;
    if (_img.isUMat())
    {
        _img.copyTo(img);
    }
    else
    {
        img = _img.getMat();
    }

    // feature Extract.
    processFrame(_img, currentFeature);

    // Key frame filtering.
    ifLoop = loopCheck(tarSubmapID);

    // add Frame to KeyFrameDataset.
    if(ifLoop)
        return;
    else
    {
        KFDataBase->addKeyFrame(currentFeature, frameID, submapID);
    }
}

float LoopClosureDetectionImpl::score(InputArray feature1, InputArray feature2)
{
    Mat mat1, mat2;
    mat1 = feature1.getMat();
    mat2 = feature2.getMat();
    Mat out = mat2 * mat1.t();
    return out.at<float>(0,0);
}

void LoopClosureDetectionImpl::reset()
{
    KFDataBase->reset();
}

bool LoopClosureDetectionImpl::processFrame(InputArray img, OutputArrayOfArrays output)
{
    Mat outputFeature = output.getMat();
    Mat blob = dnn::blobFromImage(img, 1.0/255.0, inputSize);
    net->setInput(blob);
    net->forward(outputFeature);
    outputFeature /= norm(outputFeature);
    return true;
}

Ptr<LoopClosureDetection> LoopClosureDetection::create(const String& modelBin, const String& modelTxt, const Size& input_size, int backendId, int targetId)
{
    CV_Assert(!modelBin.empty());
    return makePtr<LoopClosureDetectionImpl>(modelBin, modelTxt, input_size, backendId, targetId);
}

} // namespace large_kinfu
}// namespace cv
