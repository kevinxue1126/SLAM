/**
* This file is part of ORB-SLAM2.
* LocalMapping
* The function of LocalMapping is to put the key frame sent in Tracking in the mlNewKeyFrame list
* Process new keyframes, map point check and culling, generate new map points, Local BA, keyframe culling.
* The main work is to maintain the local map, that is, the Mapping in SLAM.
* 
* The Tracking thread only judges whether the current frame needs to be added to a key frame, 
* and does not really add a map, because the main function of the Tracking thread is local positioning.
* 
* The work of processing key frames and map points in the map, 
* including how to join and delete, is done in the LocalMapping thread
* 
*/

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:
    
    bool CheckNewKeyFrames();
    
    // Process new keyframes：ProcessNewKeyFrame()   
    void ProcessNewKeyFrame();
    
    // In CreateNewMapPoint(), new map points are added to the local map through the current keyframe
    void CreateNewMapPoints();
    
    // Check culling for recently added MapPoints in ProcessNewKeyFrame and CreateNewMapPoints
    void MapPointCulling();
    
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
