// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "loop_closure_detection.hpp"
#include "keyframe.cpp"
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

    KFDataBase = makePtr<KeyFrameDatabase>(maxDatabaseSize);
}

bool LoopClosureDetectionImpl::loopCheck(int& tarSubmapID)
{
    //Calculate the similarity with all pictures in the database.

    // If the KFDataBase is too small, then skip.
    if(KFDataBase->getSize() < minDatabaseSize )
        return false;

    double maxScore = 0;
    int bestId = -1;

    std::vector<int> candidateKFs;

    // Find candidate key frames which similarity are greater than the similarityLow.
    candidateKFs = KFDataBase->getCandidateKF(currentFeature, similarityLow, maxScore, bestId);

    if( candidateKFs.empty() || maxScore < similarityHigh)
        return false;

    // Remove consecutive keyframes and keyframes from the currentSubmapID.
    std::vector<int> duplicateKFs;
    std::vector<int>::iterator iter = candidateKFs.begin();
    std::vector<int>::iterator iterTemp;
    while (iter != candidateKFs.end() )
    {
        Ptr<KeyFrame> keyFrameDB = KFDataBase->getKeyFrameByID(*iter);

        if(keyFrameDB && keyFrameDB->nextKeyFrameID != -1)
        {
            iterTemp = find(candidateKFs.begin(), candidateKFs.end(), keyFrameDB->nextKeyFrameID);
            if( iterTemp != candidateKFs.end() || keyFrameDB->submapID == currentSubmapID )
            {
                duplicateKFs.push_back(*iterTemp);
            }
        }
        iter++;
    }

    // Delete duplicated KFs.
    for(int deleteID : duplicateKFs)
    {
        iterTemp = find(candidateKFs.begin(), candidateKFs.end(), deleteID);
        if(iterTemp != candidateKFs.end())
        {
            candidateKFs.erase(iterTemp);
        }
    }

    // Remove the keyframe belonging to the currentSubmap.
    iter = candidateKFs.begin();
    while (iter != candidateKFs.end() )
    {
        Ptr<KeyFrame> keyFrameDB = KFDataBase->getKeyFrameByID(*iter);
        if(keyFrameDB->submapID == currentFrameID)
        {
            candidateKFs.erase(iter);
        }
        iter++;
    }

    // If all candidate KF from the same submap, then return true.
    int tempSubmapID = -1;
    iter = candidateKFs.begin();

    // If the candidate frame does not belong to the same submapID,
    // it means that it is impossible to specify the target SubmapID.
    while (iter != candidateKFs.end() ) {
        Ptr<KeyFrame> keyFrameDB = KFDataBase->getKeyFrameByID(*iter);
        if(tempSubmapID == -1)
        {
            tempSubmapID = keyFrameDB->submapID;
        }else
        {
            if(tempSubmapID != keyFrameDB->submapID)
                return false;
        }
        iter++;
    }

    // Check whether currentFrame is closed to previous looped Keyframe.
    if(currentFrameID - preLoopedKFID < 20)
        return false;

    if(!candidateKFs.empty())
        bestLoopFrame = KFDataBase->getKeyFrameByID(candidateKFs[0]);
    else
        return false;

    // find target submap ID
    if(bestLoopFrame->submapID == -1 || bestLoopFrame->submapID == currentSubmapID)
        return false;
    else
    {
        tarSubmapID = bestLoopFrame->submapID;
        preLoopedKFID = currentFrameID;
        currentFrameID = -1;
        
        return true;
    }
}

void LoopClosureDetectionImpl::addFrame(InputArray _img, const int frameID, const int submapID, int& tarSubmapID, bool& ifLoop)
{

    CV_Assert(!_img.empty());
    currentFrameID = frameID;
    currentSubmapID = submapID;

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
    processFrame(img, currentFeature);

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

void LoopClosureDetectionImpl::reset()
{
    KFDataBase->reset();
}

void LoopClosureDetectionImpl::processFrame(InputArray img, Mat& output)
{
    Mat imgBlur, outMat;
    cv::GaussianBlur(img, imgBlur, cv::Size(7, 7), 0);
    Mat blob = dnn::blobFromImage(imgBlur, 1.0/255.0, inputSize);
    net->setInput(blob);
    net->forward(outMat);

    outMat /= norm(outMat);
    output = outMat.clone();
    
    //! Add ORB feature.
}

Ptr<LoopClosureDetection> LoopClosureDetection::create(const String& modelBin, const String& modelTxt, const Size& input_size, int backendId, int targetId)
{
    CV_Assert(!modelBin.empty());
    return makePtr<LoopClosureDetectionImpl>(modelBin, modelTxt, input_size, backendId, targetId);
}

} // namespace large_kinfu
}// namespace cv
