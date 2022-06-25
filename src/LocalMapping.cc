/**
* This file is part of ORB-SLAM2.
* 
* The function of LocalMapping is to put the keyframes sent from Tracking in the mlNewKeyFrame list. 
* Process new keyframes, check and remove map points, generate new map points, Local BA, and remove keyframes. 
* The main work is to maintain the local map, that is, the Mapping in SLAM.
* 
* 
* The Tracking thread only determines whether the current frame needs to be added to a key frame, 
* and does not really add a map, because the main function of the Tracking thread is local positioning.
* 
* The work of processing key frames and map points in the map, including how to join and delete, is done in the LocalMapping thread
* 
* 
* Map
* Process the new keyframe KeyFrame to complete the local map construction
* Insert keyframes ------>  Process map points (filter generated map points, then generate map points)  -------->  Local BA minimizes reprojection error ---Adjustment--------> Filter newly inserted keyframes
*
* mlNewKeyFrames     list queue to store keyframes
* 1】check queue
*       CheckNewKeyFrames();
* 
* 2】Proces New Key Frames 
* 	ProcessNewKeyFrame(); Update the association between map points MapPoints and key frame KeyFrame, UpdateConnections() updates the association
* 
* 3】MapPoints
*       Remove newly added but poor quality map points from the map
*       a)  IncreaseFound common point of view  / IncreaseVisible  projected on the image < 25%
*       b) Too few keyframes to observe this point
* 
* 4】MapPoints
* 	Some map points recovered by triangulation during movement and key frames with a high degree of common vision
* 
* 5】MapPoints fusion
*       Detect the current keyframe and adjacent, duplicate map points in the keyframe (two levels of adjacent) leave the map point with the height of the observation frame
* 
* 6】The local BA is minimized, the reprojection error is matched with the map points in the adjacent keyframes of the current keyframe, 
*    and the reprojection error is minimized for the local BA to optimize the point coordinates and pose
*      
* 
* 7】Keyframe culling
*      More than 90% of its map points can be observed by other co-view keyframes (at least 3), and this keyframe is considered redundant and can be deleted
* 
*/
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

	LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
	    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
	    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
	{
	}

	void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
	{
	    mpLoopCloser = pLoopCloser;
	}

	void LocalMapping::SetTracker(Tracking *pTracker)
	{
	    mpTracker=pTracker;
	}

	void LocalMapping::Run()
	{

	    mbFinished = false;

	    while(1)
	    {
		// Tracking will see that Local Mapping is busy
		// Step 1: Set the inter-process access flag to tell the Tracking thread that the LocalMapping thread is processing new keyframes and is busy
		SetAcceptKeyFrames(false);

		// Check if there are keyframes in the queue
		// The list of pending keyframes is not empty
		if(CheckNewKeyFrames())
		{
		    // BoW conversion and insertion in Map
		    // Step 2: Calculate the dictionary word vector BoW map of the keyframe feature points, insert the keyframe into the map
		    ProcessNewKeyFrame();

		    // Check recent MapPoints
		    // Eliminate unqualified MapPoints introduced in ProcessNewKeyFrame function
		    // Step 3: Fusion of newly added map points Check and eliminate the most recently added MapPoints in ProcessNewKeyFrame and CreateNewMapPoints    
		    //   MapPointCulling();
		    // Triangulate new MapPoints
		    
		    // Step 4: Create new map points Some new map points MapPoints are recovered by triangulation with adjacent keyframes during camera motion	    
		    CreateNewMapPoints();
		    
		    MapPointCulling();// move from above to below

	      	    // The last keyframe in the queue has been processed
		    if(!CheckNewKeyFrames())
		    {
			// Find more matches in neighbor keyframes and fuse point duplications
			// Step 5: Fusion of adjacent frame map points Check and fuse the duplicate MapPoints of the current key frame and adjacent frames (two-level adjacent)
			SearchInNeighbors();
		    }

		    mbAbortBA = false;

		    // The last keyframe in the queue has been processed, and the closed loop detection did not request to stop LocalMapping
		    if(!CheckNewKeyFrames() && !stopRequested())
		    {
			// Step 6: Local map optimization Local BA
			if(mpMap->KeyFramesInMap() > 2)
			    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

			// Step 7: Keyframe fusion, detecting and eliminating redundant keyframes in adjacent keyframes of the current frame
			KeyFrameCulling();
		    }
		    // Step 8: Add the current frame to the closed loop detection queue
		    mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
		}
		// Step 9: Wait for the thread to be idle to complete the insertion and fusion of one frame of keyframes
		else if(Stop())
		{
		    // Safe area to stop
		    while(isStopped() && !CheckFinish())
		    {
			usleep(3000);
		    }
		    if(CheckFinish())//检查 是否完成
			break;
		}
		
               // Check reset
		ResetIfRequested();

		// Step 10: Tell 	Tracking will see that Local Mapping is not busy
		SetAcceptKeyFrames(true);

		if(CheckFinish())
		    break;

		usleep(3000);
	    }

	    SetFinish();
	}

	/**
	 * @brief Insert keyframes
	 *
	 * Insert keyframes into the map for future local map optimization
	 * Here is just inserting keyframes into the list and waiting
	 * @param pKF KeyFrame
	 */
	void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
	{
	    unique_lock<mutex> lock(mMutexNewKFs);
	     // Insert a keyframe into the list of pending keyframes
	    mlNewKeyFrames.push_back(pKF);
	    mbAbortBA=true;// BA optimization stop
	}

	/**
	 * @brief    Check if there are any keyframes waiting to be processed in the list
	 * @return Returns true if it exists
	 */
	bool LocalMapping::CheckNewKeyFrames()
	{
	    unique_lock<mutex> lock(mMutexNewKFs);
	    return(!mlNewKeyFrames.empty());// Is the list of pending keyframes empty
	}


  
  
	/**
	* @brief Process keyframes in a list
	* 
	* - Calculate Bow, speed up triangulation of new MapPoints
	* - Associate the current keyframe to MapPoints, and update the average observation direction and observation distance range of MapPoints
	* - Insert keyframes, update Covisibility graph and Essential graph
	* @see VI-A keyframe insertion
	*/
	void LocalMapping::ProcessNewKeyFrame()
	{	  
	     // Step 1: Take a frame of pending keyframes from the buffer queue
             // The Tracking thread inserts keyframes into the LocalMapping and stores them in the queue
	    {
		unique_lock<mutex> lock(mMutexNewKFs);
		// Get a keyframe from the list waiting to be inserted
		mpCurrentKeyFrame = mlNewKeyFrames.front();
		mlNewKeyFrames.pop_front();// out of the team
	    }

	    // Compute Bags of Words structures
	    // Step 2: Calculate the Bow mapping relationship of the keyframe feature points    
	    //  Calculate the current key frame Bow according to the dictionary, which is convenient for later triangulation to restore new map points
	    mpCurrentKeyFrame->ComputeBoW();// frame descriptor, a vector of linear representations of dictionary words

	    // Associate MapPoints to the new keyframe and update normal and descriptor
	    //Step 3: Track the MapPoints on the new match and the current keyframe binding in the process of tracking the local map
	    // Match the MapPoints in the local map with the current frame in the TrackLocalMap function, 
	    // but do not associate these matching MapPoints with the current frame 
	    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	    for(size_t i=0; i<vpMapPointMatches.size(); i++)
	    {
		MapPoint* pMP = vpMapPointMatches[i];// Every map point that matches the current keyframe
		if(pMP)//map point exists
		{
		    if(!pMP->isBad())
		    {
		       // Update properties for MapPoints that are overtracked in the current frame
			if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))//in the lower field of view
			{
			    pMP->AddObservation(mpCurrentKeyFrame, i);// Add keyframes to map points
			    pMP->UpdateNormalAndDepth();// Map point update, average viewing direction and viewing distance depth
			    pMP->ComputeDistinctiveDescriptors();// After adding keyframes, update the best descriptor of the map point
			}
			else // this can only happen for new stereo points inserted by the Tracking
			{
			   // Put the newly inserted MapPoints during binocular or RGBD tracking into mlpRecentAddedMapPoints and wait to check
                           // MapPoints are also generated by triangulation in the CreateNewMapPoints function
                           // These MapPoints will be checked by the MapPointCulling function
			    mlpRecentAddedMapPoints.push_back(pMP);
			    // Candidate map points to be checked are stored in mlpRecentAddedMapPoints
			}
		    }
		}
	    }    

	    // Update links in the Covisibility Graph
	    // Step 4: Update the connection relationship between keyframes, Covisibility diagram and Essential diagram (tree)
	    mpCurrentKeyFrame->UpdateConnections();

	    // Insert Keyframe in Map
	    // Step 5: Insert that keyframe into the map
	    mpMap->AddKeyFrame(mpCurrentKeyFrame);
	}
	
	
	/**
	 * @brief Eliminate ProcessNewKeyFrame (map points not on the frame enter the list to be checked) and 
	 *             CreateNewMapPoints (the new map points generated by two frames of triangular transformation enter the list to be checked)
	 *             Poor quality MapPoints introduced in the function
	 * @see VI-B recent map points culling
	 */
	void LocalMapping::MapPointCulling()
	{
	    // Check Recent Added MapPoints
	    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();//map point to be detected
	    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;//current keyframe id

	    // The initial three keyframes from the keyframe of adding the map point, the first frame is not counted, 
	    // the next two frames see the number of frames of the map point, for monocular <=2, for binocular and RGBD<=3 ;
	    int nThObs;
	    if(mbMonocular)
		nThObs = 2;
	    else
		nThObs = 3;
	    const int cnThObs = nThObs;
 	    // Iterate over the MapPoints waiting to be checked
	    while(lit !=mlpRecentAddedMapPoints.end())
	    {
		MapPoint* pMP = *lit;
		if(pMP->isBad())
		{
 		    // Step 1: MapPoints that are already dead points are deleted directly from the check list	  
		    lit = mlpRecentAddedMapPoints.erase(lit);
		}
		//  Tracking (matching) to the normal frame number of the map point (IncreaseFound) < The normal number of frames that the map point should be observed (25%*IncreaseVisible):
		// Although the map point is within the field of view, it is rarely detected by ordinary frames
		else if(pMP->GetFoundRatio()<0.25f )
		{
 		// Step 2: Eliminate MapPoints that do not meet VI-B conditions
		// VI-B Condition 1:
		// Compared with the number of Frames tracked to the MapPoint, it is estimated that the proportion of the Frames that can observe the MapPoint should be greater than 25%
		// IncreaseFound() / IncreaseVisible(the map point is in view) < 25%，note not necessarily keyframes。		  
		    pMP->SetBadFlag();
		    lit = mlpRecentAddedMapPoints.erase(lit);//Remove from pending list
		}
		
		// The initial three key frames and the number of map point observations should not be too few, 
		// and the single-purpose requirements are more stringent, and all three frames need to be seen
		else if(( (int)nCurrentKFid - (int)pMP->mnFirstKFid) >=2 && pMP->Observations() <= cnThObs)
		{
  		     // Step 3: Eliminate MapPoints that do not meet VI-B conditions
            	    // VI-B Condition 2: No less than 2 frames have elapsed since the establishment of this point, 
		    // but the number of key frames observed at this point does not exceed cnThObs frames, then the point inspection fails
		    pMP->SetBadFlag();
		    lit = mlpRecentAddedMapPoints.erase(lit);//Remove from pending list
		}
		else if(((int)nCurrentKFid - (int)pMP->mnFirstKFid ) >= 3)
		    // Step 4: Since the establishment of this point, 3 frames have passed (the map points in the first three frames are more valuable and need special inspection), and the detection of the MapPoint is abandoned	  
		    lit = mlpRecentAddedMapPoints.erase(lit);
		else
		    lit++;
	    }
	}


	/**
	 * @brief Some MapPoints are recovered by triangulation during camera movement and key frames with a high degree of common vision
	 *  Recover some new map points according to the current keyframe, excluding local map points that match the current keyframe (already processed in ProcessNewKeyFrame),
	 *  First process the relationship between the new keyframe and the local map point, then check the local map point, 
	 *  and finally restore the new local map point through the new keyframe: CreateNewMapPoints()
	 *  
	 * 
	 *  Step 1: Find the first nn adjacent frames vpNeighKFs with the highest common view degree in the common view key frame of the current key frame
	 *  Step 2: Traverse each keyframe adjacent to the current keyframe vpNeighKFs
	 *  Step 3: Determine whether the baseline of the camera movement (the relative coordinates of the camera between the two needles) is long enough
	 *  Step 4: Calculate the fundamental matrix between the two keyframes according to their poses F = inv(K1 transpose) * t12 cross product R12 * inv(K2)
	 *  Step 5: Accelerate the matching through the inter-frame dictionary vector, limit the search range during matching by epipolar constraints, and perform feature point matching	
	 *  Step 6: Generate 3D points by triangulation for each pair of matching points 2d-2d, similar to the Triangulate function	
	 *  Step 6.1: Take out matching feature points
	 *  Step 6.2: Use the matching point back projection to get the parallax angle to decide whether to use triangulation restoration (large parallax angle) or direct 2-d point back projection (small parallax angle)
	 *  Step 6.3: For binoculars, use the binocular baseline depth to get the parallax angle
	 *  Step 6.4: When the parallax angle is large, use triangulation to restore 3D points
	 *  Step 6.4: When the binocular parallax angle is small, the two-dimensional point is back-projected into a three-dimensional point by using the depth value, and it is skipped directly for a single purpose.
	 *  Step 6.5: Detect if the generated 3D point is in front of the camera
	 *  Step 6.6: Calculate the reprojection error of the 3D point under the current key frame, skip if the error is large
	 *  Step 6.7: Calculate the reprojection error of 3D points under adjacent keyframes, skip if the error is large
	 *  Step 6.9: The 3D point is successfully generated by triangulation, and it is constructed as a map point MapPoint
	 *  Step 6.9: Add properties to this MapPoint
	 *  Step 6.10: Put the newly generated points into the detection queue mlpRecentAddedMapPoints and hand them over to MapPointCulling() to check whether the generated points are suitable
	 * @see  
	 */	
	void LocalMapping::CreateNewMapPoints()
	{
	    // Retrieve neighbor keyframes in covisibility graph
	    int nn = 10;
	    if(mbMonocular)
		nn=20;//Monocular
		
	    // Step 1: Find the common view degree in the common view key frame of the current key frame, the highest nn frame adjacent frame vpNeighKFs
	    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

	    ORBmatcher matcher(0.6,false);// descriptor matcher
            // current keyframe
	    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();// world ---> current keyframe
	    cv::Mat Rwc1 = Rcw1.t();// current keyframe ---> world
	    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
	    cv::Mat Tcw1(3,4,CV_32F);// world ---> current keyframe, transformation matrix
	    Rcw1.copyTo(Tcw1.colRange(0,3));
	    tcw1.copyTo(Tcw1.col(3));
	    // Get the coordinates of the current keyframe in the world coordinate system
	    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();
            // In-camera parameters
	    const float &fx1 = mpCurrentKeyFrame->fx;
	    const float &fy1 = mpCurrentKeyFrame->fy;
	    const float &cx1 = mpCurrentKeyFrame->cx;
	    const float &cy1 = mpCurrentKeyFrame->cy;
	    const float &invfx1 = mpCurrentKeyFrame->invfx;
	    const float &invfy1 = mpCurrentKeyFrame->invfy;
	    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;//match point

	    int nnew=0;

	    // Search matches with epipolar restriction and triangulate
	    // Step 2: Traverse each keyframe vpNeighKFs adjacent to the current keyframe	    
	    for(size_t i=0; i<vpNeighKFs.size(); i++)
	    {
		if(i>0 && CheckNewKeyFrames())
		    return;

		KeyFrame* pKF2 = vpNeighKFs[i];//Keyframe
		// Check first that baseline is not too short
	   	// The coordinates of the adjacent keyframes in the world coordinate system
		cv::Mat Ow2 = pKF2->GetCameraCenter();
	   	// Baseline vector, camera-relative coordinate between two keyframes
		cv::Mat vBaseline = Ow2-Ow1;
	   	// Baseline length	
		const float baseline = cv::norm(vBaseline);
		
		// Step 3: Determine whether the baseline of camera motion is long enough
		if(!mbMonocular)
		{
		    if(baseline < pKF2->mb)
		    continue;// If it is a stereo camera, the keyframe spacing is too small to not generate 3D points
		}
		else// Monocular camera
		{
		   // Median scene depth of adjacent keyframes
		    const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);//median depth
		    // The ratio of baseline and depth of field
		    const float ratioBaselineDepth = baseline/medianDepthKF2; 
                   // If it is very far away (the scale is very small), then the current adjacent keyframes are not considered, and 3D points are not generated
		    if(ratioBaselineDepth < 0.01)
			continue;
		}

		// Compute Fundamental Matrix
		// Step 4: Calculate the fundamental matrix between two keyframes based on their poses	
      		 // Calculate the fundamental matrix between two keyframes based on the poses of the two keyframes
       		// F =  inv(K1 Transpose)*E*inv(K2) = inv(K1 Transpose) * t12 cross product R12 * inv(K2)
		cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

		// Search matches that fullfil epipolar constraint
		// Step 5: Accelerate the matching through the inter-frame dictionary vector, limit the search range during matching by epipolar constraints, and perform feature point matching		
		vector<pair<size_t,size_t> > vMatchedIndices;// Feature matching candidate points
		matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

         	 // adjacent keyframes
		cv::Mat Rcw2 = pKF2->GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pKF2->GetTranslation();
		cv::Mat Tcw2(3,4,CV_32F);// transformation matrix
		Rcw2.copyTo(Tcw2.colRange(0,3));
		tcw2.copyTo(Tcw2.col(3));
		// Camera internal parameters
		const float &fx2 = pKF2->fx;
		const float &fy2 = pKF2->fy;
		const float &cx2 = pKF2->cx;
		const float &cy2 = pKF2->cy;
		const float &invfx2 = pKF2->invfx;
		const float &invfy2 = pKF2->invfy;

		// Triangulate each match
		// Triangulate each matching point pair
		// Step 6: For each pair of matching points 2d-2d, generate 3D points by triangulation, similar to the Triangulate function		
		const int nmatches = vMatchedIndices.size();
		for(int ikp=0; ikp<nmatches; ikp++)
		{
	 	    // Step 6.1: Take out matching feature points	  
		    const int &idx1 = vMatchedIndices[ikp].first; // The index of the current matching pair in the current keyframe
		    const int &idx2 = vMatchedIndices[ikp].second;// The index of the current matching pair in the adjacent keyframe
		    //current keyframe 
		    const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
		    const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
		    bool bStereo1 = kp1_ur >= 0;//The right image matching point abscissa >= 0 is the binocular/depth camera
		     // Adjacent keyframes
		    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
		    const float kp2_ur = pKF2->mvuRight[idx2];
		    bool bStereo2 = kp2_ur >= 0;

	 	    // Step 6.2: Use the matching point back projection to get the parallax angle    
		    // Check parallax between rays 
		    cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x - cx1)*invfx1, (kp1.pt.y - cy1)*invfy1, 1.0);
		    cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x - cx2)*invfx2, (kp2.pt.y - cy2)*invfy2, 1.0);
		    
		    // From the camera coordinate system to the world coordinate system, get the cosine value of the parallax angle
		    cv::Mat ray1 = Rwc1*xn1;// Camera coordinate system ------> World coordinate system
		    cv::Mat ray2 = Rwc2*xn2;
		    // vector a × vector b / (vector a modulo × vector bar modulo) = cosine of the included angle
		    const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));
		    
		    // Add 1 to make cosParallaxStereo randomly initialized to a large value
		    float cosParallaxStereo = cosParallaxRays+1;
		    float cosParallaxStereo1 = cosParallaxStereo;
		    float cosParallaxStereo2 = cosParallaxStereo;
	 	    // Step 6.3: For binoculars, use the binocular baseline depth to get the parallax angle
		    if(bStereo1)
			cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
		    else if(bStereo2)
			cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));
		    // Get the parallax angle of binocular observation
		    cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);
		    
	 	    // Step 6.4: Triangulation to restore 3D points
		    cv::Mat x3D;
		    // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998) Indicates that the parallax angle is normal
                    // cosParallaxRays < cosParallaxStereo Indicates that the parallax angle is small
                    // When the parallax angle is small, use trigonometry to restore 3D points, and when the parallax angle is large, use binocular to restore 3D points (binocular and depth are valid)
		    if(cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
		    {
		       // Linear Triangulation Method
			cv::Mat A(4,4,CV_32F);
			A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
			A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
			A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
			A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

			cv::Mat w,u,vt;
			cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

			x3D = vt.row(3).t();

			if(x3D.at<float>(3)==0)
			    continue;

			// Euclidean coordinates
			x3D = x3D.rowRange(0,3) / x3D.at<float>(3);

		    }
	   	    //Step 6.4: When the binocular parallax angle is small, the two-dimensional point is back-projected into a three-dimensional point using the depth value
		    else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
		    {
			x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);             
		    }
		    else if(bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
		    {
			x3D = pKF2->UnprojectStereo(idx2);
		    }
	       	    //3D points cannot be generated when the monocular parallax is small
		    else
			continue; // No stereo and very low parallax

		    cv::Mat x3Dt = x3D.t();
		    
	 	     // Step 6.5: Detect if the generated 3D point is in front of the camera
		    //Check triangulation in front of cameras
		    float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);// Only the z coordinate value is counted
		    if(z1<= 0)
			continue;
		    float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
		    if(z2 <= 0)
			continue;
		    
         	    // Step 6.6: Calculate the reprojection error of the 3D point under the current keyframe
		    //Check reprojection error in first keyframe
		    const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
		    const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);//Camera normalized coordinates
		    const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
		    const float invz1 = 1.0/z1;
		    if(!bStereo1)
		    {// Monocular
			float u1 = fx1*x1*invz1 + cx1;//pixel coordinates
			float v1 = fy1*y1*invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			if((errX1*errX1+errY1*errY1) > 5.991*sigmaSquare1)
			    continue;//Projection error is too large to skip
		    }
		    else
		    {
			float u1 = fx1*x1*invz1+cx1;
			float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1; //Right image matching point abscissa
			float v1 = fy1*y1*invz1+cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			float errX1_r = u1_r - kp1_ur;
			// Threshold calculated based on the chi-square test (assuming that the measurement has a one-pixel deviation)
			if((errX1*errX1 + errY1*errY1 + errX1_r*errX1_r) > 7.8*sigmaSquare1)
			    continue;//Projection error is too large to skip
		    }
         	    // Step 6.7: Calculate the reprojection error of 3D points under adjacent keyframes
		    //Check reprojection error in second keyframe
		    const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
		    const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
		    const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
		    const float invz2 = 1.0/z2;
		    if(!bStereo2)
		    {// Monocular
			float u2 = fx2*x2*invz2+cx2;
			float v2 = fy2*y2*invz2+cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
			    continue;//Projection error is too large to skip
		    }
		    else
		    {	// Right image matching point abscissa difference
			float u2 = fx2*x2*invz2+cx2;
			float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;//Right image matching point abscissa
			float v2 = fy2*y2*invz2+cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			float errX2_r = u2_r - kp2_ur;
			if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
			    continue;//Projection error is too large to skip
		    }
		    
           	    // Step 6.8: Check for Scale Continuity
		    //Check scale consistency
		    cv::Mat normal1 = x3D-Ow1;//  In the world coordinate system, the vector between the 3D point and the camera, the direction is from the camera to the 3D point
		    float dist1 = cv::norm(normal1);// modulo length
		    cv::Mat normal2 = x3D-Ow2;
		    float dist2 = cv::norm(normal2);
		    if(dist1==0 || dist2==0)
			continue;// modulo length 0 skip
                   // ratioDist is the distance ratio without considering the pyramid scale
		    const float ratioDist = dist2/dist1;
		   // The scale of the pyramid scale factor
		    const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];
		    /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
			continue;*/
		    // The depth ratio and the pyramid level ratio under the two images should be similar
		    // |ratioDist/ratioOctave |<ratioFactor
		    if(ratioDist * ratioFactor<ratioOctave || ratioDist > ratioOctave*ratioFactor)
			continue;
		    
	  	     // Step 6.9: The 3D point is successfully generated by triangulation, and it is constructed as a map point MapPoint
		    // Triangulation is succesfull
		    MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
		    

            	    // Step 6.9: Add properties to this MapPoint:
	       	    // a. Observe the key frame of the MapPoint
		    pMP->AddObservation(mpCurrentKeyFrame,idx1); // Map point
		    pMP->AddObservation(pKF2,idx2);// 
		    mpCurrentKeyFrame->AddMapPoint(pMP,idx1);// Keyframe
		    pKF2->AddMapPoint(pMP,idx2);
              	    // b. Descriptor of the MapPoint
		    pMP->ComputeDistinctiveDescriptors();
               	    // c. The average observation direction and depth range of the MapPoint
		    pMP->UpdateNormalAndDepth();
               	    // d. Add map points to the map
		    mpMap->AddMapPoint(pMP);
	
	
            	    // Step 6.10: Put the newly generated points into the detection queue mlpRecentAddedMapPoints
                    // These MapPoints will be checked by the MapPointCulling function
		    mlpRecentAddedMapPoints.push_back(pMP);

		    nnew++;
		}
	    }
	}
	

	/**
	 * @brief   Check and fuse the map points MapPoints where the current keyframe is duplicated with adjacent frames (first-level and second-level adjacent frames)
	 * Step 1: Obtain the first-level adjacent keyframes with the top nn weights of the current keyframe in the covisibility frame connection graph (selected by the number of times the map points of the current frame are observed)
	 * Step 2: Obtain the top 5 second-level adjacent keyframes of the current keyframe in its first-level neighboring frames in the covisibility graph
	 * Step 3: Integrate the map points MapPoints of the current frame with the map points MapPoints of the first-level and second-level adjacent frames (retain the one with the highest number of observations)
	 * Step 4: Find all the map points MapPoints of the first-level and second-level adjacent frames and fuse them with the map points MapPoints of the current frame
	 * Step 5: Update the descriptor, depth, main direction of observation and other attributes of the map point MapPoints of the current frame
	 * Step 5: Update the current connection relationship with other frames (the number of times the map points of each other are observed, etc.)
	 * @return  None
	 */	
	void LocalMapping::SearchInNeighbors()
	{
	    // Retrieve neighbor keyframes
 	    // Step 1: Obtain the first-level adjacent keyframes with the top nn weights of the current keyframe in the covisibility graph
	    int nn = 10;
	    if(mbMonocular)
		nn=20;//Monocular	
	    // first-level adjacency	
	    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
	    vector<KeyFrame*> vpTargetKFs;// The last qualified primary and secondary adjacent keyframes
	    // Iterate over each first-level adjacent frame
	    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
	    {
		KeyFrame* pKFi = *vit;// first-level adjacent keyframes
		if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)//bad frame or already added
		    continue;
		vpTargetKFs.push_back(pKFi);// Join the last qualified adjacent keyframes
		pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;// Adjacent matching has been done
		
 		// Step 2: Obtain the top 5 second-level adjacent key frames of the current key frame in the covisibility graph of its first-level neighboring frames
		// Extend to some second neighbors
		const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
		// Iterate over every second level adjacent frame
		for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
		{
		    KeyFrame* pKFi2 = *vit2;// Secondary adjacent keyframes
		    if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
			continue;
		    vpTargetKFs.push_back(pKFi2);
		}
	    }

	    // Step 3: Integrate the map points MapPoints of the current frame with the map points MapPoints of the first-level and second-level adjacent frames respectively
	    // Search matches by projection from current KF in target KFs
	    ORBmatcher matcher;
	    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//map points that match the current frame
	    // vector<KeyFrame*>::iterator
	    for(auto  vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
	    {
		//its first-level and second-level adjacent frames
		KeyFrame* pKFi = *vit;
		// Project the MapPoints of the current frame to the adjacent key frame pKFi, search for matching key points in the additional area, and determine whether there are duplicate MapPoints
		// 1. If the MapPoint can match the feature point of the key frame, and the point has a corresponding MapPoint, then merge the two MapPoints (select the one with more observations)
		// 2. If the MapPoint can match the feature point of the key frame, and the point does not have a corresponding MapPoint, then add a MapPoint to the point		
		matcher.Fuse(pKFi,vpMapPointMatches);
	    }
	    
	    
	    // Step 4: Fusion of all map points MapPoints of the first-level and second-level adjacent frames with the current frame (MapPoints)
            // Traverse each first-level adjacency and second-level adjacency keyframes to find all map points
	    // Search matches by projection from target KFs in current KF
	    // A collection of all MapPoints used to store first-level adjacency and second-level adjacency keyframes
	    vector<MapPoint*> vpFuseCandidates;// All map points of the first and second adjacent frames
	    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());// The number of frames × the number of map points per frame
            // vector<KeyFrame*>::iterator
	    for(auto vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
	    {
		KeyFrame* pKFi = *vitKF;//its first-level and second-level adjacent frames
		vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();//map point
		
		// vector<MapPoint*>::iterator
		for(auto vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
		{
		    MapPoint* pMP = *vitMP;// Each map point of the first-level and second-level adjacent frames
		    if(!pMP)
			continue;
		    if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
			continue;
		    // Join the collection and mark as joined
		    pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId; //tag added
		    vpFuseCandidates.push_back(pMP); // Join first-level and second-level adjacent frames
		}
	    }
	    
            //All the map points of the first-level and second-level adjacent frames are merged with the current frame
            // Project MapPoints to the current frame, search for matching key points in the additional area, and determine whether there are duplicate map points
	    // 1. If the MapPoint can match the feature point of the current frame, and the point has a corresponding MapPoint, then merge the two MapPoints (select the one with more observations)
	    // 2. If the MapPoint can match the feature point of the current frame, and the point does not have a corresponding MapPoint, then add a MapPoint to the point
	    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

	    // Step 5: Update the descriptor, depth, observation main direction and other attributes of the current frame MapPoints
	    // Update points
	    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//All matching map points in the current frame
	    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
	    {
		MapPoint* pMP=vpMapPointMatches[i];//Current frame, map points matched by each keypoint
		if(pMP)//存在
		{
		    if(!pMP->isBad())
		    {
			pMP->ComputeDistinctiveDescriptors(); // update, Descriptor of the map point (select the best descriptor among all observed descriptors)
			pMP->UpdateNormalAndDepth();          // Update the mean viewing direction and viewing distance
		    }
		}
	    }
	    //Step 5: After updating the MapPoints of the current frame, update the connection relationship with other frames, and observe the number of mutual map points and other information
            // update covisibility picture
	    // Update connections in covisibility graph
	    mpCurrentKeyFrame->UpdateConnections();
	}
	

	/**
	 * @brief    keyframe culling
	 *  In the keyframe in the Covisibility Graph keyframe connection graph, 
	 *  if more than 90% of the map points MapPoints can be observed by other keyframes (at least 3), 
	 *  the keyframe is considered to be a redundant keyframe.
	 *  
	 * @param  pKF1 keyframe 1
	 * @param  pKF2 keyframe 2
	 * @return Fundamental matrix F between two keyframes
	 */	
	void LocalMapping::KeyFrameCulling()
	{
	    // Check redundant keyframes (only local keyframes)
	    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
	    // in at least other 3 keyframes (in the same or finer scale)
	    // We only consider close stereo points
	  
	    // Step 1: Extract all co-view keyframes (association frames) of the current frame according to the Covisibility Graph keyframe connection graph 
	    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
	    
            // vector<KeyFrame*>::iterator
            // Iterate over all local keyframes	    
	    for(auto  vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
	    {
		KeyFrame* pKF = *vit;// each locally associated frame of the current frame
		if(pKF->mnId == 0)//The first frame keyframe is skipped when initializing the world keyframe
		    continue;
		
		// Step 2: Extract the map points MapPoints of each common view keyframe
		const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();// Map points for locally associated frame matching

		int nObs = 3;
		const int thObs=nObs; //3
		int nRedundantObservations=0;
		int nMPs=0;
		
		// Step 3: Traverse the MapPoints of the local keyframe to determine whether more than 90% of the MapPoints can be observed by other keyframes (at least 3)	
		for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
		{
		    MapPoint* pMP = vpMapPoints[i];// MapPoints of the local keyframe
		    if(pMP)
		    {
			if(!pMP->isBad())
			{
			    if(!mbMonocular)
			    {  // For binocular, only close MapPoints are considered, no more than mbf*35/fx
				if(pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
				    continue;
			    }

			    nMPs++;
			    // Map Points MapPoints are observed by at least three keyframes
			    if(pMP->Observations() > thObs)// Number of observation frames > 3
			    {
				const int &scaleLevel = pKF->mvKeysUn[i].octave;// Pyramid levels
				const map<KeyFrame*, size_t> observations = pMP->GetObservations();// Local observation keyframe map
				int nObs=0;
				for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
				{
				    KeyFrame* pKFi = mit->first;
				    if(pKFi==pKF)
					continue;
				    const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;// Pyramid levels
				    
                                   // The scale constraint requires that the feature scale of MapPoint in this local key frame is greater than (or similar to) the feature scale of other key frames
				    if(scaleLeveli <= scaleLevel+1)
				    {
					nObs++;
					if(nObs >= thObs)
					    break;
				    }
				}
				if(nObs >= thObs)
				{// The MapPoint is observed by at least three keyframes
				    nRedundantObservations++;
				}
			    }
			}
		    }
		}  
		
 		// Step 4: If more than 90% of the MapPoints of the local key frame can be observed by other key frames (at least 3), it is considered as redundant key frame
		if(nRedundantObservations > 0.9*nMPs)
		    pKF->SetBadFlag();
	    }
	}
	
	/**
	 * @brief    Calculate the fundamental matrix between two keyframes based on the poses of the two keyframes
	 *                 F = inv(K1 transpose)*E*inv(K2) = inv(K1 transpose)*t cross product R*inv(K2)
	 * @param  pKF1 keyframe 1
	 * @param  pKF2 keyframe 2
	 * @return Fundamental matrix F between two keyframes
	 */
	cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
	{
    	    // Essential Matrix: t12 fork multiplied by R12
    	    // Fundamental Matrix: inv(K1 transpose)*E*inv(K2)
	    cv::Mat R1w = pKF1->GetRotation();   // Rc1w
	    cv::Mat t1w = pKF1->GetTranslation();
	    cv::Mat R2w = pKF2->GetRotation();   // Rc2w
	    cv::Mat t2w = pKF2->GetTranslation();// t c2 w

	    cv::Mat R12 = R1w*R2w.t();// R12 =Rc1w *  Rwc2 // c2 -->w --->c1
	    cv::Mat t12 = -R12*t2w + t1w; // tw2  + t1w    // c2 -->w --->c1

	    // The cross product matrix of t12
	    cv::Mat t12x = SkewSymmetricMatrix(t12);

	    const cv::Mat &K1 = pKF1->mK;
	    const cv::Mat &K2 = pKF2->mK;

	    return K1.t().inv()*t12x*R12*K2.inv();
	}

	/**
	 * @brief  Request to stop local mapping thread
	 * @return None
	 */	
	void LocalMapping::RequestStop()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    mbStopRequested = true;//local mapping
	    unique_lock<mutex> lock2(mMutexNewKFs);
	    mbAbortBA = true;//Stop BA optimization
	}
	
	/**
	 * @brief   Stop local mapping thread
	 * @return  None
	 */	
	bool LocalMapping::Stop()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    if(mbStopRequested && !mbNotStop)
	    {
		mbStopped = true;
		cout << "局部建图停止 Local Mapping STOP" << endl;
		return true;
	    }
	    return false;
	}
	
	/**
	 * @brief    Check if the local mapping thread is stopped
	 * @return   whether to stop sign
	 */	
	bool LocalMapping::isStopped()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    return mbStopped;
	}
	
	/**
	 * @brief   return request to stop local mapping thread
	 * @return  request to stop local mapping thread
	 */	
	bool LocalMapping::stopRequested()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    return mbStopRequested;
	}
	
	/**
	 * @brief   Release the local mapping thread
	 * @return  None
	 */	
	void LocalMapping::Release()
	{
	    unique_lock<mutex> lock(mMutexStop);
	    unique_lock<mutex> lock2(mMutexFinish);
	    if(mbFinished)
		return;
	    mbStopped = false;
	    mbStopRequested = false;
	    // list<KeyFrame*>::iterator
	    for(auto lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
		delete *lit;//delete keyframes
	    mlNewKeyFrames.clear();

	    cout << "局部建图释放 Local Mapping RELEASE" << endl;
	}

	/**
	 * @brief    Returns the flag that a new keyframe can be received
	 * @return   Is it possible to receive a new keyframe
	 */		
	bool LocalMapping::AcceptKeyFrames()
	{
	    unique_lock<mutex> lock(mMutexAccept);
	    return mbAcceptKeyFrames;
	}
	
	/**
	 * @brief    Set the flag that can receive a new keyframe
	 * @return   None
	 */
	void LocalMapping::SetAcceptKeyFrames(bool flag)
	{
	    unique_lock<mutex> lock(mMutexAccept);
	    mbAcceptKeyFrames=flag;
	}
	
	/**
	 * @brief   Set do not stop sign
	 * @return  success or failure
	 */
	bool LocalMapping::SetNotStop(bool flag)
	{
	    unique_lock<mutex> lock(mMutexStop);

	    if(flag && mbStopped)//  In the case of already stopped, set do not stop
		return false;

	    mbNotStop = flag;

	    return true;
	}
	/**
	 * @brief    Stop global optimization BA
	 * @return   Is it possible to receive a new keyframe
	 */
	void LocalMapping::InterruptBA()
	{
	    mbAbortBA = true;
	}

	
	
	/**
	 * @brief   Compute the cross product matrix of vectors
	 * @return  the cross product of this vector
	 */	
	cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
	{
	    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
		    v.at<float>(2),               0,-v.at<float>(0),
		    -v.at<float>(1),  v.at<float>(0),              0);
	}
	
	/**
	 * @brief   request reset
	 * @return  none
	 */
	void LocalMapping::RequestReset()
	{
	    {
		unique_lock<mutex> lock(mMutexReset);
		mbResetRequested = true;
	    }

	    while(1)
	    {
		{
		    unique_lock<mutex> lock2(mMutexReset);
		    if(!mbResetRequested)
			break;
		}
		usleep(3000);
	    }
	}
	
	/**
	 * @brief   reset thread
	 * @return  none
	 */
	void LocalMapping::ResetIfRequested()
	{
	    unique_lock<mutex> lock(mMutexReset);
	    if(mbResetRequested)
	    {
		mlNewKeyFrames.clear();
		mlpRecentAddedMapPoints.clear();
		mbResetRequested=false;
	    }
	}
	
	void LocalMapping::RequestFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    mbFinishRequested = true;
	}

	bool LocalMapping::CheckFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    return mbFinishRequested;
	}

	void LocalMapping::SetFinish()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    mbFinished = true;    
	    unique_lock<mutex> lock2(mMutexStop);
	    mbStopped = true;
	}

	bool LocalMapping::isFinished()
	{
	    unique_lock<mutex> lock(mMutexFinish);
	    return mbFinished;
	}

} //namespace ORB_SLAM
