/**
* This file is part of ORB-SLAM2.
* Keyframe
* The KeyFrame class is constructed using the Frame class. 
* What kind of Frame can be considered as a key frame and when a key frame needs to be added is implemented in the tracking module. 
* Selected representative frames from ordinary frames
* 
* 
* Since part of the data in the KeyFrame will be accessed and modified by multiple threads, 
* it is necessary to add thread locks to these members to ensure that only one thread has access rights at the same time. Thread safety is involved:
* [1] Setting of key frame pose (lock(mMutexPose));
* [2] Setting the connection relationship between key frames (lock(mMutexConnections));
* [3] The operation of keyframes corresponding to map points (lock(mMutexFeatures)), including calculating the weight between connected keyframes through map points.
* 
* A map is maintained in KeyFrame, which saves the KeyFrame* and weight (the number of MapPonits that are viewed together) with the current frame. 
* The relationship between key frames is done with a weighted directed graph, 
* so it is necessary to understand the principle of its spanning tree spanning tree.
* 
*
* What is more difficult to understand in KeyFrame is the SetBagFlag() function. 
* Before actually deleting the current key frame, it is necessary to deal with the relationship between the father and son key frames. 
* Otherwise, the graph maintained by the entire key frame will be broken or chaotic, 
* and it will not be able to provide the backend with more information. Good initial value.
*
*
* It is understood that the father has died, the son needs to find a new father, 
* and find it in the candidate father. The father of the current frame (mpParent) must be in the candidate father;
*
*/

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

    long unsigned int KeyFrame::nNextId=0;
    //Copy Constructor  Default initialization
    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
	mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
	mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
	mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
	mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
	fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
	mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
	mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
	mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
	mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
	mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
	mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
	mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
	mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
    {
	mnId=nNextId++;// keyframe id

	mGrid.resize(mnGridCols);// The grid of the key frame is the vector of the vertor's vector
	for(int i=0; i<mnGridCols;i++)// 64 columns per column
	{
	    mGrid[i].resize(mnGridRows);//becomes 48 lines in size
	    for(int j=0; j<mnGridRows; j++)//each line
		mGrid[i][j] = F.mGrid[i][j];// normal frame grid
	}

	SetPose(F.mTcw);    
    }
    

    // Convert the descriptor matrix of the current frame (which can be converted into a vector) into a bag of words model vector
   // （DBoW2::BowVector mBowVec； DBoW2::FeatureVector mFeatVec；）：
    void KeyFrame::ComputeBoW()
    {
	if(mBowVec.empty() || mFeatVec.empty())
	{
	   // The mat type descriptor is converted into the container mat type
	    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
	    // Feature vector associate features with nodes in the 4th level (from leaves up)
	    // We assume the vocabulary tree has 6 levels, change the 4 otherwise
	    mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
	}
    }

    /*
     * Set member variables in KeyFrame
     * mTcw，Transform the world coordinate system to the camera coordinate system
     * mTwc，Convert the camera coordinate system to the world coordinate system
     * Ow（Left eye camera center coordinates）= Rcw.t() * (-tcw )  
     * 
     */
    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
	unique_lock<mutex> lock(mMutexPose);
	Tcw_.copyTo(Tcw);// copy to class variable  
	cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);// world to camera rotation matrix
	cv::Mat tcw = Tcw.rowRange(0,3).col(3);   // world to camera translation vector
	cv::Mat Rwc = Rcw.t(); // camera to world rotation matrix
	Ow = -Rwc*tcw;// The camera center point is in the world coordinate system coordinates

	// The direct inversion of cw requires a large amount of calculation. Generally, 
        // the matrix inversion will be represented by an equivalent matrix expression during implementation. 
        // Here Ow corresponds to the translation vector -RTt in Tcw-1.
	Twc = cv::Mat::eye(4,4,Tcw.type());
	Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
	Ow.copyTo(Twc.rowRange(0,3).col(3));
	cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
	Cw = Twc*center;
    }

    // world to camera pose
    cv::Mat KeyFrame::GetPose()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Tcw.clone();
    }
    // camera to the world
    cv::Mat KeyFrame::GetPoseInverse()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Twc.clone();
    }

    // Single camera optical center point coordinates
    cv::Mat KeyFrame::GetCameraCenter()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Ow.clone();
    }

    // Binocular camera baseline center point Coordinates
    cv::Mat KeyFrame::GetStereoCenter()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Cw.clone();
    }

    // rotation vector
    cv::Mat KeyFrame::GetRotation()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Tcw.rowRange(0,3).colRange(0,3).clone();
    }
    // translation vector
    cv::Mat KeyFrame::GetTranslation()
    {
	unique_lock<mutex> lock(mMutexPose);
	return Tcw.rowRange(0,3).col(3).clone();
    }

    // Add connections between keyframes, through the weight connection between keyframes, weight refers to the map points observed together between the two keyframes
    void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
    {
	{
	    unique_lock<mutex> lock(mMutexConnections);
	    if(!mConnectedKeyFrameWeights.count(pKF))
		mConnectedKeyFrameWeights[pKF]=weight;
	    else if(mConnectedKeyFrameWeights[pKF]!=weight)
		mConnectedKeyFrameWeights[pKF]=weight;
	    else
		return;
	}
	// Each keyframe maintains its own map, which records the weight between it and other keyframes. 
	// Each time a new connection keyframe is added to the current keyframe, the map structure needs to be reordered according to weight
	UpdateBestCovisibles();
    }

    // Sort by weights corresponding to keyframes
    void KeyFrame::UpdateBestCovisibles()
    {
	unique_lock<mutex> lock(mMutexConnections);
	vector<pair<int,KeyFrame*> > vPairs;
	// Pair is to combine 2 data into one data. When such a requirement is required, pair can be used. For example, map in stl is to store key and value together.
	vPairs.reserve(mConnectedKeyFrameWeights.size());
	
	// mit is of type map<KeyFrame*,int>::iterator
	for(auto mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
	  vPairs.push_back(make_pair(mit->second,mit->first));

	sort(vPairs.begin(),vPairs.end());
	list<KeyFrame*> lKFs;
	list<int> lWs;
	for(size_t i=0, iend=vPairs.size(); i<iend;i++)
	{
	    lKFs.push_front(vPairs[i].second);
	    lWs.push_front(vPairs[i].first);
	}
	
	mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
	mvOrderedWeights = vector<int>(lWs.begin(), lWs.end()); 
    }

    //  Returns all concatenated keyframes to get a sequence of keyframes
    set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
    {
	unique_lock<mutex> lock(mMutexConnections);
	set<KeyFrame*> s;
      	// mit is of type map<KeyFrame*,int>::iterator
	for(auto mit=mConnectedKeyFrameWeights.begin(); mit!=mConnectedKeyFrameWeights.end(); mit++)
	    s.insert(mit->first);// Insert keyframes
	return s;
    }

    // Returns all connected keyframes Returns all sequence keyframe containers
    vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
    {
	unique_lock<mutex> lock(mMutexConnections);
	return mvpOrderedConnectedKeyFrames;
    }

    // Return the top N optimal keyframes
    vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
	unique_lock<mutex> lock(mMutexConnections);
	if((int)mvpOrderedConnectedKeyFrames.size()<N)
	    return mvpOrderedConnectedKeyFrames;
	else
	    return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
    }

    // binary search
    vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
	unique_lock<mutex> lock(mMutexConnections);

	if(mvpOrderedConnectedKeyFrames.empty())
	    return vector<KeyFrame*>();
        // Binary search returns the element with weight w for
	vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
	if(it==mvOrderedWeights.end())
	    return vector<KeyFrame*>();
	else
	{
	    int n = it-mvOrderedWeights.begin();
	    return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
	}
    }

    // Returns the weight of the keyframe for
    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
	unique_lock<mutex> lock(mMutexConnections);
	if(mConnectedKeyFrameWeights.count(pKF))
	    return mConnectedKeyFrameWeights[pKF];
	else
	    return 0;
    }

    // Add map point
    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
    {
	unique_lock<mutex> lock(mMutexFeatures);
	mvpMapPoints[idx]=pMP;
    }

    // Delete map point by id
    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
	unique_lock<mutex> lock(mMutexFeatures);
	mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
    }
    // delete map point
    void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
    {
	int idx = pMP->GetIndexInKeyFrame(this);//得到id
	if(idx>=0)
	    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
    }

    // Replace map point
    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
    {
	mvpMapPoints[idx]=pMP;
    }

    // get all map points
    set<MapPoint*> KeyFrame::GetMapPoints()
    {
	unique_lock<mutex> lock(mMutexFeatures);
	set<MapPoint*> s;
	for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
	{
	    if(!mvpMapPoints[i])
		continue;
	    MapPoint* pMP = mvpMapPoints[i];
	    if(!pMP->isBad())
		s.insert(pMP);
	}
	return s;
    }

    // Track map points, the number of key points that can be observed
    int KeyFrame::TrackedMapPoints(const int &minObs)
    {
	unique_lock<mutex> lock(mMutexFeatures);

	int nPoints=0;
	const bool bCheckObs = minObs>0;
	for(int i=0; i<N; i++)
	{
	    MapPoint* pMP = mvpMapPoints[i];
	    if(pMP)
	    {
		if(!pMP->isBad())
		{
		    if(bCheckObs)
		    {
			if(mvpMapPoints[i]->Observations() >= minObs)
			    nPoints++;
		    }
		    else
			nPoints++;
		}
	    }
	}
	return nPoints;
    }
    
    // get all map points  sort(vector)
    vector<MapPoint*> KeyFrame::GetMapPointMatches()
    {
	unique_lock<mutex> lock(mMutexFeatures);
	return mvpMapPoints;
    }

    MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
    {
	unique_lock<mutex> lock(mMutexFeatures);
	return mvpMapPoints[idx];
    }

    // Establish connections between keyframes
    void KeyFrame::UpdateConnections()
    {
	map<KeyFrame*,int> KFcounter;

	vector<MapPoint*> vpMP;

	{
	    unique_lock<mutex> lockMPs(mMutexFeatures);
	    vpMP = mvpMapPoints;
	}

	//For all map points in keyframe check in which other keyframes are they seen
	//Increase counter for those keyframes
	// vector<MapPoint*>::iterator vit 
	for(auto vit = vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
	{
	    MapPoint* pMP = *vit;

	    if(!pMP)
		continue;

	    if(pMP->isBad())
		continue;
	    
	  //  map<KeyFrame*,size_t> observations observations 
	  auto  observations = pMP->GetObservations();
	  // map<KeyFrame*,size_t>::iterator mit 
	    for(auto  mit = observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		if(mit->first->mnId == mnId)
		    continue;
		KFcounter[mit->first]++;
	    }
	}

	// This should not happen
	if(KFcounter.empty())
	    return;

	//If the counter is greater than threshold add connection
	//In case no keyframe counter is over threshold add the one with maximum counter
	int nmax=0;
	KeyFrame* pKFmax=NULL;
	int th = 15;

	vector<pair<int,KeyFrame*> > vPairs;
	vPairs.reserve(KFcounter.size());
	// map<KeyFrame*,int>::iterator mit
	for(auto  mit = KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
	{
	    if(mit->second > nmax)
	    {
		nmax = mit->second;
		pKFmax=mit->first;
	    }
	    if(mit->second >= th)
	    {
		vPairs.push_back(make_pair(mit->second , mit->first));
		(mit->first)->AddConnection(this, mit->second);
	    }
	}

	if(vPairs.empty())
	{
	    vPairs.push_back(make_pair(nmax,pKFmax));
	    pKFmax->AddConnection(this,nmax);
	}

        sort(vPairs.begin(),vPairs.end());// Frame reordering of map points over 15 observations
	list<KeyFrame*> lKFs;// Keyframe
	list<int> lWs;// number of observations
	for(size_t i=0; i<vPairs.size();i++)
	{
	    lKFs.push_front(vPairs[i].second);// orderly
	    lWs.push_front(vPairs[i].first);
	}

	{
	    unique_lock<mutex> lockCon(mMutexConnections);

	    // mspConnectedKeyFrames = spConnectedKeyFrames;
	    mConnectedKeyFrameWeights = KFcounter;
	    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
	    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

	    if(mbFirstConnection && mnId!=0)
	    {
		mpParent = mvpOrderedConnectedKeyFrames.front();
		mpParent->AddChild(this);
		mbFirstConnection = false;
	    }

	}
    }

    // keyframe tree
    // add child
    // set red-black binary tree
    void KeyFrame::AddChild(KeyFrame *pKF)
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	mspChildrens.insert(pKF); // insert
    }

    // delete child
    void KeyFrame::EraseChild(KeyFrame *pKF)
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	mspChildrens.erase(pKF);
    }

    void KeyFrame::ChangeParent(KeyFrame *pKF)
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	mpParent = pKF;
	pKF->AddChild(this);
    }

    set<KeyFrame*> KeyFrame::GetChilds()
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	return mspChildrens;
    }

    KeyFrame* KeyFrame::GetParent()
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	return mpParent;
    }

    bool KeyFrame::hasChild(KeyFrame *pKF)
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	return mspChildrens.count(pKF);
    }

    void KeyFrame::AddLoopEdge(KeyFrame *pKF)
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	mbNotErase = true;
	mspLoopEdges.insert(pKF);
    }

    set<KeyFrame*> KeyFrame::GetLoopEdges()
    {
	unique_lock<mutex> lockCon(mMutexConnections);
	return mspLoopEdges;
    }

    void KeyFrame::SetNotErase()
    {
	unique_lock<mutex> lock(mMutexConnections);
	mbNotErase = true;
    }

    // First set it as a bad frame. If the frame is not a loopback frame, it can be deleted; if the frame is a loopback frame, it cannot be deleted.
    void KeyFrame::SetErase()
    {
	{
	    unique_lock<mutex> lock(mMutexConnections);
	    if(mspLoopEdges.empty())
	    {
		mbNotErase = false;
	    }
	}

	if(mbToBeErased)
	{
	    SetBadFlag();
	}
    }

    /*
     * What is more difficult to understand in KeyFrame is the SetBagFlag() function. 
     * Before actually deleting the current key frame, it is necessary to deal with the relationship between the father and son key frames. Otherwise, 
     * the graph maintained by the entire key frame will be broken or chaotic, and it will not be able to provide the backend with more information. 
     * Good initial value.
     * 
     * It is understood that the father has died, the son needs to find a new father, and find it in the candidate father. 
     * The father of the current frame (mpParent) must be in the candidate father;
     */
    void KeyFrame::SetBadFlag()
    {   
	{
	    unique_lock<mutex> lock(mMutexConnections);
	    if(mnId==0)
		return;
	    else if(mbNotErase)
	    {
		mbToBeErased = true;
		return;
	    }
	}
    	//[1] Delete the original connection relationship
       // map<KeyFrame*,int>::iterator
	for(auto mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit != mend; mit++)
	    mit->first->EraseConnection(this);

	for(size_t i=0; i<mvpMapPoints.size(); i++)
	    if(mvpMapPoints[i])
		mvpMapPoints[i]->EraseObservation(this);
	   {
	    unique_lock<mutex> lock(mMutexConnections);
	    unique_lock<mutex> lock1(mMutexFeatures);

	    mConnectedKeyFrameWeights.clear();
	    mvpOrderedConnectedKeyFrames.clear();

	    // Update Spanning Tree
   	    // [2] First, put the father of the current frame into the candidate father
	    set<KeyFrame*> sParentCandidates;
	    sParentCandidates.insert(mpParent);

	    // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
	    // Include that children as new parent candidate for the rest
	    while(!mspChildrens.empty())
	    {
		bool bContinue = false;

		int max = -1;
		KeyFrame* pC;
		KeyFrame* pP;
		
		// [3] Traverse all the sons of the current frame
                  // set<KeyFrame*>::iterator
		for(auto sit=mspChildrens.begin(), send=mspChildrens.end(); sit != send; sit++)
		{
		    KeyFrame* pKF = *sit;// child frame of the current frame
		    if(pKF->isBad())
			continue;
		    
           	    // [4] Then traverse each common view frame of son A
		    // Check if a parent candidate is connected to the keyframe
		    vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
		    for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
		    {   // set<KeyFrame*>::iterator
	        	// [5] Check whether each co-view frame of the son frame is one of the candidate father frames
			for( auto spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
			{
		 	    // [6] If there is a candidate father frame in the co-viewing frame of the son's frame A, 
			    // update the father of the son's frame A to the candidate father. 
			    // It should be that the current frame needs to be deleted, and the son needs to find a new father.
			    if(vpConnected[i]->mnId == (*spcit)->mnId)
			    {
				int w = pKF->GetWeight(vpConnected[i]);
				if(w>max)
				{
				    pC = pKF;
				    pP = vpConnected[i];
				    max = w;
				    bContinue = true;
				}
			    }
			}
		    }
		}

		if(bContinue)
		{
		    pC->ChangeParent(pP);
		    sParentCandidates.insert(pC);// And put A into the candidate father (because A has connected the whole graph at this time)
		    mspChildrens.erase(pC);
		}
		else
		    break;
	    }

	    // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
	    if(!mspChildrens.empty())
	      // set<KeyFrame*>::iterator
		for(auto  sit=mspChildrens.begin(); sit != mspChildrens.end(); sit++)
		{
		    (*sit)->ChangeParent(mpParent);// Directly set the father of the son's frame B as the father of the current frame, and leave it to the grandfather to manage
		}

	    mpParent->EraseChild(this);// Father deletes current frame
	    mTcp = Tcw*mpParent->GetPoseInverse();
	    mbBad = true;
	}
	mpMap->EraseKeyFrame(this);//map delete current frame
	mpKeyFrameDB->erase(this);// keyframe database, delete the current frame
    }

    bool KeyFrame::isBad()
    {
	unique_lock<mutex> lock(mMutexConnections);
	return mbBad;
    }

    void KeyFrame::EraseConnection(KeyFrame* pKF)
    {
	bool bUpdate = false;
	{
	    unique_lock<mutex> lock(mMutexConnections);
	    if(mConnectedKeyFrameWeights.count(pKF))
	    {
		mConnectedKeyFrameWeights.erase(pKF);
		bUpdate=true;
	    }
	}

	if(bUpdate)
	    UpdateBestCovisibles();
    }

    vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
    {
	vector<size_t> vIndices;
	vIndices.reserve(N);

	const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
	if(nMinCellX>=mnGridCols)
	    return vIndices;

	const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
	if(nMaxCellX<0)
	    return vIndices;

	const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
	if(nMinCellY>=mnGridRows)
	    return vIndices;

	const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
	if(nMaxCellY<0)
	    return vIndices;

	for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
	{
	    for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
	    {
		const vector<size_t> vCell = mGrid[ix][iy];
		for(size_t j=0, jend=vCell.size(); j<jend; j++)
		{
		    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
		    const float distx = kpUn.pt.x-x;
		    const float disty = kpUn.pt.y-y;

		    if(fabs(distx)<r && fabs(disty)<r)
			vIndices.push_back(vCell[j]);
		}
	    }
	}

	return vIndices;
    }

    bool KeyFrame::IsInImage(const float &x, const float &y) const
    {
	return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
    }

    cv::Mat KeyFrame::UnprojectStereo(int i)
    {
	const float z = mvDepth[i];
	if(z>0)
	{
	    const float u = mvKeys[i].pt.x;
	    const float v = mvKeys[i].pt.y;
	    const float x = (u-cx)*z*invfx;
	    const float y = (v-cy)*z*invfy;
	    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

	    unique_lock<mutex> lock(mMutexPose);
	    return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
	}
	else
	    return cv::Mat();
    }

    //  Monocular, ambient and depth median
    float KeyFrame::ComputeSceneMedianDepth(const int q)
    {
	vector<MapPoint*> vpMapPoints;
	cv::Mat Tcw_;
	{
	    unique_lock<mutex> lock(mMutexFeatures);
	    unique_lock<mutex> lock2(mMutexPose);
	    vpMapPoints = mvpMapPoints;
	    Tcw_ = Tcw.clone();
	}

	vector<float> vDepths;
	vDepths.reserve(N);
	cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);// Coordinate value
	Rcw2 = Rcw2.t();
	float zcw = Tcw_.at<float>(2,3);// z-axis
	for(int i=0; i<N; i++)
	{
	    if(mvpMapPoints[i])
	    {
		MapPoint* pMP = mvpMapPoints[i];
		cv::Mat x3Dw = pMP->GetWorldPos();
		float z = Rcw2.dot(x3Dw) + zcw;// amount of translation
		vDepths.push_back(z);//depth
	    }
	}

	sort(vDepths.begin(),vDepths.end());// sort

	return vDepths[(vDepths.size()-1)/q];//median depth
    }

} //namespace ORB_SLAM
