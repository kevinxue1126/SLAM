/**
* This file is part of ORB-SLAM2.
* map point  map point on normal frame
*               map points on keyframes
* Create map points, set of observation frames, optimal descriptors
* Ring point detection, the number of observations is less than 2, delete map points
* The map point distance, the relative coordinates of the camera center of the reference frame
* Map point, relative reference frame, the distance of the camera center in each scale space of each layer on the image pyramid
* 
* Map points can be constructed through key frames or ordinary frames, 
* but in the end, they must correspond to key frames. Map points constructed through ordinary frames are only temporarily used by Tracking for tracking.
* 
* 
* 
* Add map point observation: there is a common view relationship between keyframes that can observe the same map point
* 
*/

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

	long unsigned int MapPoint::nNextId=0;
	mutex MapPoint::mGlobalMutex;// global thread lock

	// Create keyframe map points, world coordinate point, owning keyframe, owning map
	// The reference frame is a key frame, and the map point will correspond to many frame key frames to establish a common viewing relationship between key frames
	MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
	    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
	    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
	    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
	    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
	{
	    Pos.copyTo(mWorldPos);// world coordinate point, copied to the class
	    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

	    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	    mnId=nNextId++;
	}

	// Create common frame, map point, world coordinate point, owning map, owning common frame and frame id
	// The reference frame is an ordinary frame, and the map point only corresponds to the feature point of the current ordinary frame
	MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
	    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
	    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
	    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
	    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
	{
	    Pos.copyTo(mWorldPos);// World coordinate point, copied into the class
	    cv::Mat Ow = pFrame->GetCameraCenter();// Ordinary frame, camera, light line coordinates (monocular left camera)
	    mNormalVector = mWorldPos - Ow;// The coordinates of the point relative to the camera optical center 
	    mNormalVector = mNormalVector/cv::norm(mNormalVector);

	    cv::Mat PC = Pos - Ow;// Relative to camera optical center coordinates
	    const float dist = cv::norm(PC);// The distance of the point coordinates from the camera's optical center 
	    const int level = pFrame->mvKeysUn[idxF].octave;// The number of levels in the pyramid where the key points are located
	    const float levelScaleFactor =  pFrame->mvScaleFactors[level];// // The scale factor of the pyramid where the key points are located
	    const int nLevels = pFrame->mnScaleLevels;

	    mfMaxDistance = dist*levelScaleFactor;// maximum distance
	    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];// shortest distance

	    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);// descriptor

	    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
	    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
	    mnId=nNextId++;
	}

	void MapPoint::SetWorldPos(const cv::Mat &Pos)
	{
	    unique_lock<mutex> lock2(mGlobalMutex);
	    unique_lock<mutex> lock(mMutexPos);
	    Pos.copyTo(mWorldPos);
	}

	cv::Mat MapPoint::GetWorldPos()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return mWorldPos.clone();// 复制
	}

	cv::Mat MapPoint::GetNormal()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return mNormalVector.clone();// The coordinates of the point relative to the camera optical center 
	}

	KeyFrame* MapPoint::GetReferenceKeyFrame()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mpRefKF;// frame of reference for map points
	}

	// Add the observed keyframes, the corresponding keypoint id, 
	// and update the number of frames that can be observed to change the point
	// Add map point observation frame: there is a common view relationship between key frames that can observe the same map point
	void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    if(mObservations.count(pKF))// pKF has emerged
		return;
	    mObservations[pKF]=idx;// If not, add observation keyframes
	    
	  // Update the number of frames where the change point can be observed
	  if(pKF->mvuRight[idx] >= 0)// there is a match
		nObs += 2;// There is also a matching point, which is also observed in the frame of the matching point
	    else
		nObs++; 
	}

	// Delete the observation frame of the point
	// Update the number of times the point was observed -2 (with matching point pairs) / -1 (without matching point pairs)
	// Update reference frame
	// Dead point All observation frame pointers The matching point pairs related to the map point in the observation frame
	
	// Delete map point observation: 
	// Delete the corresponding key frame observation relationship from the mObservation and nObs members of the current map point. 
	// If the key frame happens to be the reference frame, it needs to be re-specified. 
	// When the number of observation cameras is less than or equal to 2, the map point needs to be eliminated. 
	// Deleting observation relationships and deleting map points (and replacing map points) need to be distinguished!
	void MapPoint::EraseObservation(KeyFrame* pKF)
	{
	    bool bBad=false;
	    {
		unique_lock<mutex> lock(mMutexFeatures);
		if(mObservations.count(pKF))
		{
	 	    // Delete the corresponding key frame observation relationship from the mObservation and nObs, members of the current map point
		    int idx = mObservations[pKF];// observation frame id
		    if(pKF->mvuRight[idx] >= 0)// There are matching frames
			nObs-=2;// its matching frame when the keyframe is deleted -= 2
		    else
			nObs--;// no matching frame - 1

		    mObservations.erase(pKF);// delete observation frame
		    
              	    // The deleted observation frame is the reference frame, and the reference frame of the current frame is set as the first frame of its observation
		    if(mpRefKF==pKF)
			mpRefKF = mObservations.begin()->first;

		    // If only 2 observations or less, discard point
		    if(nObs<=2)// The number of times the point was observed
			bBad=true;// bad point
		}
	    }
	    if(bBad)//delete bad point 
		SetBadFlag();//Delete the observation frame pointer of the map point, the matching point pair related to this point in the observation frame, the point is also deleted
	}
	
	// Get the observation frame of the map point
	map<KeyFrame*, size_t> MapPoint::GetObservations()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mObservations;
	}
	
	// Returns the number of times the point was observed
	int MapPoint::Observations()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return nObs;
	}

	// The number of times the map point is observed is too small, the point is not for reference
	// Delete the observation frame of the map point, and point to other points. Objects may also be observed by these observation points and cannot be deleted
	// Delete matching point pairs associated with remap points within the observation frame
	void MapPoint::SetBadFlag()
	{
	    map<KeyFrame*,size_t> obs;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		mbBad=true;
		obs = mObservations;// Save the observation frame corresponding to the changed point
		mObservations.clear();// Clear observation frame
	    }
	    // map<KeyFrame*,size_t>::iterator mit 
	    for(auto mit = obs.begin(), mend=obs.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;// keyframe pointer
		pKF->EraseMapPointMatch(mit->second);// delete the matching point corresponding to the changed point in the corresponding key frame
	    }

	    mpMap->EraseMapPoint(this);// delete this point
	}

	// substitute point
	MapPoint* MapPoint::GetReplaced()
	{
	    unique_lock<mutex> lock1(mMutexFeatures);
	    unique_lock<mutex> lock2(mMutexPos);
	    return mpReplaced;
	}
	
	
	// Replacement point for map point, enter another map point that can replace the local map point
	// Replace the current map point (this) with pMp. Mainly used in closed loop
	// Adjust map points and keyframes to create new relationships
	void MapPoint::Replace(MapPoint* pMP)
	{
	    if(pMP->mnId == this->mnId)//Return directly to the same point
		return;

	    int nvisible, nfound;
	    map<KeyFrame*,size_t> obs;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		obs=mObservations;// All observation frames at this point
		mObservations.clear();// Clear the observation frame pointer
		mbBad=true;//bad point
		nvisible = mnVisible;// visible number
		nfound = mnFound;// track number
		mpReplaced = pMP;
	    }
	    // map<KeyFrame*,size_t>::iterator mit
	    for( auto mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)//For all observation frames of the original point, detect the relationship with the replacement point
	    {
		// Replace measurement in keyframe
		KeyFrame* pKF = mit->first;// Observation frame first frame

		if(!pMP->IsInKeyFrame(pKF))// Observations not at the original point are in keyframes
		{
		  // The map point corresponding to the mit->second index in the KeyFrame, replace the original this with pMP
		    pKF->ReplaceMapPointMatch(mit->second, pMP);
		    pMP->AddObservation(pKF,mit->second);// Add observation keyframes for pMp map points
		}
		else
		{
		    pKF->EraseMapPointMatch(mit->second);// Delete the map point in the map
		}
	    }
	    pMP->IncreaseFound(nfound);
	    pMP->IncreaseVisible(nvisible);
	    pMP->ComputeDistinctiveDescriptors();

	    mpMap->EraseMapPoint(this);// delete the original point
	}

	// Whether it is a dead pixel, the number of observations is less than 2
	// has been observed
	bool MapPoint::isBad()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    unique_lock<mutex> lock2(mMutexPos);
	    return mbBad;
	}

	// Number of observables +n
	void MapPoint::IncreaseVisible(int n)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    mnVisible+=n;
	}

	// Trackable times +n
	void MapPoint::IncreaseFound(int n)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    mnFound+=n;
	}

	// Trackable / Viewable
	// A low GetFoundRatio means that the map point is within the field of view of many keyframes, but not many feature points are matched.
	// The difference between visible and found: the visible map point is within the field of view, and the found map point has the number of frames corresponding to the feature point.
	// Generally speaking, found map points must be visible, but visible map points are likely not found
	float MapPoint::GetFoundRatio()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return static_cast<float>(mnFound)/mnVisible;
	}

	// get the most representative descriptor
	// Each map point (a point in the world) has a descriptor on each observation frame
	// Orb binary string Hamming distance between descriptors
	// Calculate the median of the distances between each descriptor and other descriptors
	// The smallest median distance descriptor for
	// Among the multiple feature points where the map point is observed (corresponding to multiple key frames), the descriptor with the highest degree of discrimination is selected as the descriptor of the map point;
	void MapPoint::ComputeDistinctiveDescriptors()
	{
	    // All observation frames
	    map<KeyFrame*,size_t> observations;
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		if(mbBad)
		    return;
		observations=mObservations;// All observation frames
	    }
	    if(observations.empty())
		return;
	    // all descriptors
	    // Retrieve all observed descriptors
	    vector<cv::Mat> vDescriptors;//The descriptor set of the map point on all observation frames   
	    vDescriptors.reserve(observations.size());   
	    // map<KeyFrame*,size_t>::iterator mit
	    for( auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;

		if(!pKF->isBad())
		    vDescriptors.push_back(pKF->mDescriptors.row(mit->second));// Descriptors of this point under all observation frames
	    }
	    if(vDescriptors.empty())
		return;
	     // the distance between all descriptors at this point
	    // Compute distances between them
	    const size_t N = vDescriptors.size();// The number of descriptors in the point
	    float Distances[N][N];// The distance between N descriptors at this point
	    for(size_t i=0;i<N;i++)
	    {
		Distances[i][i]=0;
		for(size_t j=i+1;j<N;j++)
		{
		    int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);// Hamming matching distance
		    Distances[i][j]=distij;
		    Distances[j][i]=distij;
		}
	    }
	     // Minimum median distance  
	    // Calculate the distance between each descriptor and other descriptors, sort, find the median and distance
	    // The corresponding descriptor on the observation frame corresponding to the smallest median distance of all descriptors
	    // Take the descriptor with least median distance to the rest   
	    int BestMedian = INT_MAX;
	    int BestIdx = 0;
	    for(size_t i=0;i<N;i++)
	    {
		vector<int> vDists(Distances[i],Distances[i]+N);//distance between each descriptor and other descriptors
		sort(vDists.begin(),vDists.end());// sort
		int median = vDists[0.5*(N-1)];// median distance
		if(median<BestMedian)
		{
		    BestMedian = median;//keep the smallest median distance
		    BestIdx = i;
		}
	    }
	    // Minimum median distance for the corresponding descriptor
	    {
		unique_lock<mutex> lock(mMutexFeatures);
		mDescriptor = vDescriptors[BestIdx].clone();// and other descriptors most wanted descriptors
	    }
	}

	// Get the most representative descriptor of the map point in all observation frames
	cv::Mat MapPoint::GetDescriptor()
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return mDescriptor.clone();
	}

	// Returns the position of the given frame in the set of observation frames for this map point
	int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    if(mObservations.count(pKF))// Find the position of the keyframe in the set of observation keyframes
		return mObservations[pKF];
	    else
		return -1;
	}

	// whether the given keyframe is within the set of observation frames at this point
	bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    return (mObservations.count(pKF));
	}
	
	// 2. Calculate the average observation direction and depth of map points
	// Update map point, relative observation frame, camera center, normalized coordinates
	// Update map points, the minimum and maximum distances under each pyramid level under the reference frame
	// The average observation direction and the range of the observation distance of the map point are all preparations for the subsequent descriptor fusion.	
	void MapPoint::UpdateNormalAndDepth()
	{
	    map<KeyFrame*,size_t> observations;// Observation frame multi-frame
	    KeyFrame* pRefKF;//reference keyframe
	    cv::Mat Pos;//world 3D coordinates
	    {
		unique_lock<mutex> lock1(mMutexFeatures);
		unique_lock<mutex> lock2(mMutexPos);
		if(mbBad)
		    return;
		observations=mObservations;// map point observation frame
		pRefKF=mpRefKF;//map point reference keyframe
		Pos = mWorldPos.clone();//world 3D coordinates
	    }
	    if(observations.empty())
		return;

 	    // [1] Update the observation direction, calculate the 3D world, the unitized relative coordinates of the map point in each observation frame, and the relative camera under the camera
	    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
	    int n=0;
	    // map<KeyFrame*,size_t>::iterator mit
	    // Map points to all observed keyframe camera center vectors, normalized and added
	    for(auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKF = mit->first;// observation key frame
		cv::Mat Owi = pKF->GetCameraCenter();// Observe the keyframe camera coordinate center
		cv::Mat normali = mWorldPos - Owi;//The relative coordinates (direction vector) of the 3D point to the camera center of the observation frame
		normal = normal + normali/cv::norm(normali);//归一化后相加 
		n++;
	    }
// 【2】更新观测深度  在参考帧下 各个图像金字塔下 的距离----------------------
    // 通常来说，距离较近的地图点，将在金字塔层数较高的地方提取出，距离较远的地图点，
    // 在金字塔层数较低的地方提取出（金字塔层数越低，分辨率越高，才能识别出远点）
	    cv::Mat PC = Pos - pRefKF->GetCameraCenter();// 相对于参考帧 相机中心 的 相对坐标
	    const float dist = cv::norm(PC);// 3D点相对于 参考帧相机中心的 距离
	    // 因此通过地图点的信息（主要是对应描述子），我们可以获得该地图点对应的金字塔层级：
	    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;// 在 参考帧下 图像金字塔 中的层级位置
	    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];// 对应层级下 的尺度因子
	    const int nLevels = pRefKF->mnScaleLevels;
	    {
		unique_lock<mutex> lock3(mMutexPos);
		
		// 乘上参考帧中描述子获取时金字塔放大尺度，得到最大距离mfMaxDistance
		mfMaxDistance = dist*levelScaleFactor;// 原来的距离 在 对于层级尺度下的 距离
		
		// 最大距离除以整个金字塔最高层的放大尺度得到最小距离mfMinDistance。
		mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
		mNormalVector = normal/n;
	    }
	}

	float MapPoint::GetMinDistanceInvariance()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return 0.8f*mfMinDistance;// 各个图像金字塔下 的距离 最小距离
	}

	float MapPoint::GetMaxDistanceInvariance()
	{
	    unique_lock<mutex> lock(mMutexPos);
	    return 1.2f*mfMaxDistance;// 各个图像金字塔下 的距离 最大距离
	}

/*
 注意金字塔ScaleFactor和距离的关系：
      当前特征点对应ScaleFactor为1.2的意思是：
          图片分辨率下降1.2倍后，可以提取出该特征点
              （分辨率更高时，肯定也可以提取出，
                  这里取金字塔中能够提取出该特征点最高层级作为该特征点的层级）

同时，由当前特征点的距离，可以推测所在层级。
 */	
	int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
	{
	    float ratio;
	    {
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;//当前特征点的距离
	    }

	    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
	    if(nScale<0)
		nScale = 0;
	    else if(nScale>=pKF->mnScaleLevels)
		nScale = pKF->mnScaleLevels-1;

	    return nScale;// 预测尺度
	}

	int MapPoint::PredictScale(const float &currentDist, Frame* pF)
	{
	    float ratio;
	    {
		unique_lock<mutex> lock(mMutexPos);
		ratio = mfMaxDistance/currentDist;
	    }

	    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
	    if(nScale<0)
		nScale = 0;
	    else if(nScale>=pF->mnScaleLevels)
		nScale = pF->mnScaleLevels-1;

	    return nScale;
	}



} //namespace ORB_SLAM
