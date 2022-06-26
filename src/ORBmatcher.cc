/**
* This file is part of ORB-SLAM2.
* ORB_ match point
* The class is responsible for
* 1 Between feature points and feature points,
* 2 Projection relationship between map points and feature points
* 3 bag of words model DBow2 for matching
* 4 Sim3 pose matching.
* 
* It is used to assist in the completion of monocular initialization, triangulation to restore new map points, tracking, 
* relocalization and loop closing, so it is more important.
* 
* 
* match between classes,local match, global match etc.
*
* The matching in Relocalization and LoopClosing is matching in a set of key frames of many frames, 
* which belongs to Place Recognition, so DBow is required, and projection matching is applicable between two frames, 
* or within the projection range (local map, the previous key frame corresponds to map points) between the MapPoints and the current frame.
* 
*/

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{
	// Threshold and other parameters
	const int ORBmatcher::TH_HIGH = 100;      // Similarity Transformation, Descriptor Matching Threshold
	const int ORBmatcher::TH_LOW = 50;        // Euclidean Transform, Descriptor Matching Threshold    
	const int ORBmatcher::HISTO_LENGTH = 30;  // Matching point pairs, the number of histogram grids of the observation direction difference

	ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
	{
	}

  
	  /**
	  * @brief Track Local MapPoint through projection
	  *
	  * Project the Local MapPoint into the current frame, thereby increasing the MapPoints of the current frame \n
	  * Local MapPoints have been reprojected (isInFrustum()) to the current frame in SearchLocalPoints() \n
	  * and marked whether the points are in the current frame's field of view, i.e. mbTrackInView \n
	  * For these MapPoints, matches are selected according to the descriptor distance near their projection points, 
	  * and the final direction voting mechanism is eliminated.
	  * 
	  * @param  F           current frame
	  * @param  vpMapPoints Local MapPoints The set of map points corresponding to the frame associated with the local map point and the current frame
	  * @param  th          threshold
	  * @return             Number of successful matches
	  * @see SearchLocalPoints() isInFrustum()
	  */
	int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
	{
	    int nmatches=0;

	    const bool bFactor = th != 1.0;

	    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)//local map point
	    {
		MapPoint* pMP = vpMapPoints[iMP];//local map point
	  	// Step 1: Determine whether the point is to be projected
		if(!pMP->mbTrackInView)//out of sight
		    continue;

		if(pMP->isBad())
		    continue;
		// Step 2: Pyramid level predicted by distance, the level is relative to the current frame
		const int &nPredictedLevel = pMP->mnTrackScaleLevel;

		// The size of the window will depend on the viewing direction
		// Step 3: The size of the search window depends on the viewing angle. If the angle between the current viewing angle and the average viewing angle is close to 0 degrees, r takes a smaller value
		float r = RadiusByViewingCos(pMP->mTrackViewCos);
	      	// Increase the range if a coarser search is required
		if(bFactor)
		    r *= th;
		
	        // Get candidate matching points in the current frame
		// Step 4: Search by projected points (projected to the current frame, see isInFrustum()) and search window and predicted scale to find nearby interest points
		const vector<size_t> vIndices =
			F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

		if(vIndices.empty())
		    continue;

		const cv::Mat MPdescriptor = pMP->GetDescriptor();// Descriptors for local map points

		int bestDist=256;
		int bestLevel= -1;
		int bestDist2=256;
		int bestLevel2 = -1;
		int bestIdx =-1 ;

		// Step 5: Map point descriptor and current frame candidate, key point descriptor matching
		// Get best and second matches with near keypoints
		// vector<size_t>::const_iterator vit
		for(auto  vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)// each candidate matching point
		{
		    const size_t idx = *vit;// each candidate matching point

	  	    // Step 6: The key point of the current frame, there is already a corresponding map point or a matching point y calculated from the map point, the offset is larger than the current stereo matching point, and the error is too large to skip.
		    //If the interest point in the current frame already has a corresponding MapPoint, exit the loop
		    if(F.mvpMapPoints[idx])
			if(F.mvpMapPoints[idx]->Observations() > 0)//Skip if observation frame is already found
			    continue;
		    // The error between the tracked matching point coordinates and the actual stereo matching is too large, skipping
		    if(F.mvuRight[idx]>0)// 双目 / 深度相机
		    {
			const float er = fabs(pMP->mTrackProjXR  -  F.mvuRight[idx]);//
			if(er > r*F.mvScaleFactors[nPredictedLevel])// The error between the tracked matching point coordinates and the actual stereo matching is too large
			    continue;
		    }

		    const cv::Mat &d = F.mDescriptors.row(idx);// Descriptor for each candidate matching point

		    const int dist = DescriptorDistance(MPdescriptor,d);// Descriptor distance between the local map point and the current frame map point
	  	    // Step 7: According to the descriptor distance, find the feature points with the smallest and second smallest distances
		    if(dist<bestDist)
		    {
			bestDist2=bestDist;// next closest distance
			bestDist=dist;// the closest distance
			bestLevel2 = bestLevel;
			bestLevel = F.mvKeysUn[idx].octave;// Pyramid level corresponding to keypoints
			bestIdx=idx;
		    }
		    else if(dist<bestDist2)
		    {
			bestLevel2 = F.mvKeysUn[idx].octave;
			bestDist2=dist;// next closest distance
		    }
		}

		// Apply ratio to second match (only if best and second are in the same scale level)
		if(bestDist<=TH_HIGH)
		{
		    if(bestLevel == bestLevel2 && bestDist > mfNNratio*bestDist2)
			continue;// 最The best match and the second best match are at the same pyramid level and the shortest distance is not less than 80% of the second shortest distance
			
		    // Step 7: Add the corresponding MapPoint to the interest point in the Frame
		    F.mvpMapPoints[bestIdx]=pMP;// 
		    nmatches++;
		}
	    }

	    return nmatches;
	}


	float ORBmatcher::RadiusByViewingCos(const float &viewCos)
	{
	    if(viewCos>0.998)
		return 2.5;
	    else
		return 4.0;
	}

	  /**
	  * @brief  Checks if the pair of points given at the match is within the epipolar range
	  * @Param  kp1   keypoint kp1 on frame 1
	  * @Param  kp2   keypoint kp2 on frame 2 pKF2
	  * @Param  F12   fundamental matrix F12 of frame 1 to frame 2, p2 transpose * F * p1 = 0
	  * @Param  pKF2  frame 2 pKF2   
	  * @return kp2   The distance from kp1 to the epipolar line on the frame 2 image is small enough within a reasonable range, and it is considered that it is possible to match
	  */
	bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
	{
	    // Epipolar line in second image  l = kp1(Homogeneous representation) transpose F12 = [a b c]
	    // Find the epipolar line corresponding to the key point kp1 on the key frame image pKF2
	    const float a = kp1.pt.x * F12.at<float>(0,0) + kp1.pt.y * F12.at<float>(1,0) + F12.at<float>(2,0);
	    const float b = kp1.pt.x * F12.at<float>(0,1) + kp1.pt.y * F12.at<float>(1,1) + F12.at<float>(2,1);
	    const float c = kp1.pt.x * F12.at<float>(0,2)  + kp1.pt.y * F12.at<float>(1,2) + F12.at<float>(2,2);
	  
	    // Calculate the distance from the kp2 feature point to the epipolar line:
	    // Polar line l：ax + by + c = 0
	    // The distance from (u,v) to l is：d = |au+bv+c| / sqrt(a^2+b^2) 
	    // d^2 = |au+bv+c|^2/(a^2+b^2)
	    const float num = a*kp2.pt.x + b*kp2.pt.y + c;
	    const float den = a*a + b*b;
	    if(den==0)
		return false;
	    const float dsqr = num*num/den;

	    return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];// Distance is within reason 3.84 chi-square constraint
	}

  
	  /**
	  * @brief Track the map points of the reference key frame through the bag of words
	  * 
	  * Quickly match the point descriptors in pKF and F through bow (feature points that do not belong to the same node (dictionary word) skip the matching directly) \n
	  * Match feature points belonging to the same node (dictionary word) by descriptor distance \n
	  * According to the matching, update the MapPoints corresponding to the feature points in the current frame F with the MapPoints corresponding to the feature points in the reference key frame pKF \n
	  * Each feature point corresponds to a MapPoint, so the MapPoint of each feature point in pKF is also the MapPoint of the corresponding point in F \n
	  * Eliminate false matches by distance threshold, scale threshold and angle voting
	  * @param  pKF                KeyFrame           reference keyframe
	  * @param  F                  Current Frame      current frame
	  * @param  vpMapPointMatches  Map Points that match the key points in the current frame F, NULL means no match
	  * @return                    number of successful matches
	  */
	int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
	{
	    // Map points with reference to keyframes
	    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
	    // The number of key points in the current frame Matching points (corresponding to the map points in the original key frame)
	    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
	    // feature vector of the map point descriptor of the reference keyframe
	    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

	    int nmatches=0;

	    vector<int> rotHist[HISTO_LENGTH];// Direction Vector Histogram
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;

	    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
	    // Both the key frame and the current frame are represented linearly by dictionary words
	    // The descriptors of the corresponding words must be relatively similar. Matching the descriptors of the corresponding words can speed up the matching
	    // Match ORB features belonging to the same node (specific layer)
	    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();// reference key frame feature point descriptor dictionary feature vector start
	    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();// current frame feature point descriptor dictionary feature vector start
	    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();// Reference key frame Feature point descriptor Dictionary feature vector End
	    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();// The end of the current frame feature point descriptor dictionary feature vector

	    while(KFit != KFend && Fit != Fend)
	    {
		//Step 1: Take out the ORB feature points belonging to the same node respectively (only those belonging to the same node (word) can be matching points) 
		if(KFit->first == Fit->first)// Descriptors under the same word
		{
		    const vector<unsigned int> vIndicesKF = KFit->second;
		    const vector<unsigned int> vIndicesF = Fit->second;
		    
		  // Step 2: Traverse the map points belonging to the node in the key frame KF, which corresponds to a descriptor
		  for(size_t iKF=0; iKF < vIndicesKF.size(); iKF++)// Each reference keyframe map point
		    {
			const unsigned int realIdxKF = vIndicesKF[iKF];
			MapPoint* pMP = vpMapPointsKF[realIdxKF];// Take out the MapPoint corresponding to the feature in KF
			// Eliminate bad map points
			if(!pMP)
			    continue;
			if(pMP->isBad())
			    continue;    
			// Take out the descriptor corresponding to the feature in the key frame KF
			const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);  

			int bestDist1=256;// Best distance (minimum distance)
			int bestIdxF =-1 ;
			int bestDist2=256;// Best distance (minimum distance)
			
			// Step 3: Traverse the feature points belonging to the node in the current frame F and find the best matching point
			for(size_t iF=0; iF<vIndicesF.size(); iF++)//every current frame
			{
			    const unsigned int realIdxF = vIndicesF[iF];
			    
			    // Indicates that this feature point has been matched, no longer matched, speed up
			    if(vpMapPointMatches[realIdxF])
				continue;
			    // Take out the descriptor corresponding to the feature in the current frame F
			    const cv::Mat &dF = F.mDescriptors.row(realIdxF); 
			    const int dist =  DescriptorDistance(dKF,dF);// distance between descriptors

			    //  Step 4: Find the matching points corresponding to the shortest distance and the second shortest distance
			    if(dist<bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
			    {
				bestDist2=bestDist1;// 次最短的距离
				bestDist1=dist;// 最短的距离
				bestIdxF=realIdxF;
			    }
			    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
			    {
				bestDist2=dist;
			    }
			}
			// Step 5: Voting out false matches based on threshold and angle
			if(bestDist1<= TH_LOW)// The shortest distance is less than the threshold 
			{
			    // trick!
			    // The best match is obviously better than the next best match, then the best match is really reliable
			    if(static_cast<float>(bestDist1) < mfNNratio*static_cast<float>(bestDist2)) 
			    {
				// Step 6: Update the map point MapPoint corresponding to the feature point of the current frame		      
				vpMapPointMatches[bestIdxF]=pMP;// Map point in the matched reference key

				const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];//The pixel point of the map point in the reference keyframe

				if(mbCheckOrientation)// Check if the direction is right
				{
				    // trick!
				    // angle：The rotation main direction angle of each feature point when extracting the descriptor, if the image is rotated, this angle will change
				    // The angle change of all feature points should be consistent, and the most accurate angle change value is obtained through histogram statistics
				    float rot = kp.angle - F.mvKeys[bestIdxF].angle;// The direction of the key point of the current frame and the direction of the matching point change
				    if(rot<0.0)
					rot+=360.0f;
				    int bin = round(rot*factor);
				    if(bin==HISTO_LENGTH)
					bin=0;			
				    // The angle difference of each pair of matching points can be put into the range of a bin (360/HISTO_LENGTH)
				    assert(bin>=0 && bin<HISTO_LENGTH);
				    rotHist[bin].push_back(bestIdxF);// Orientation histogram
				}
				nmatches++;
			    }
			}

		    }

		    KFit++;
		    Fit++;
		}
		else if(KFit->first < Fit->first)
		{
		    KFit = vFeatVecKF.lower_bound(Fit->first);
		}
		else
		{
		    Fit = F.mFeatVec.lower_bound(KFit->first);
		}
	    }

	    // Step 7: Eliminate mismatched points based on orientation
	    if(mbCheckOrientation)
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
	       // Statistical direction deviation histogram The three bins with the highest frequency are retained, and the matching points in other ranges are eliminated.
	       // In addition, if the highest one is more than 10 times higher than the second highest, only the matching points in the highest bin are kept.
	       // If the highest is more than 10 times higher than the third highest, the matching points in the highest and second highest bins are kept.
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		for(int i=0; i<HISTO_LENGTH; i++)
		{
		    // If the rotation angle variation of the feature point belongs to these three groups, keep
		    if(i==ind1 || i==ind2 || i==ind3)// Three bins with the highest statistical histogram
			continue;
		    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
		    {
		        // Remove the matching points except ind1, ind2, ind3
			vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);// Matching points in other ranges are eliminated.
			nmatches--;
		    }
		}
	    }

	    return nmatches;
	}

	  /**
	  * @brief   For the 2D feature points in the key frame pKF that have not been matched to the 3D map points, match the map points from the given map points
	  * Convert to Euclidean transformation according to Sim3 transformation,
	  * Project each vpPoints to the image pixel coordinate system of the reference key frame pKF, and determine a search area according to the scale, 
	  * and match the feature points in the area according to the descriptor of the MapPoint. 
	  * If the matching error is less than TH_LOW, the matching is successful. update vpMatched
	  * @param  pKF           KeyFrame            reference keyframe
	  * @param  Scw           Similarity transformation with reference keyframes   [s*R t] 
	  * @param  vpPoints      Map point
	  * @param  vpMatched     Refer to the matching points corresponding to the keyframe feature points
	  * @param  th            Match distance threshold
	  * @return               Number of successful matches
	  */
	int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
	{
	    // Get Calibration Parameters for later projection
	    // In-camera parameters
	    const float &fx = pKF->fx;
	    const float &fy = pKF->fy;
	    const float &cx = pKF->cx;
	    const float &cy = pKF->cy;

	    // Step 1: Convert the similarity transformation to Euclidean transformation, and normalize the similarity transformation matrix Decompose Scw
	    // | s*R  t|
	    // |   0    1|
	    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);// similarity transformation  rotation matrix
	    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));// Calculate the scale s of the similarity transformation matrix
	    cv::Mat Rcw = sRcw/scw;// normalized rotation matrix
	    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;//  Normalized Computed Similarity Transformation Matrix
	    cv::Mat Ow = -Rcw.t()*tcw;// In the pKF coordinate system, the displacement from the world coordinate system to pKF, the direction from the world coordinate system to pKF
	    // Rwc * twc  Used to calculate the distance of map points from the camera to infer possible scales in the image pyramid

	    // Set of MapPoints already found in the KeyFrame
	    // Step 2: Use the set type and remove the unmatched points to quickly retrieve whether a MapPoint has a match
	    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
	    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

	    int nmatches=0;

	    // For each Candidate MapPoint Project and Match
	    // Step 3: Traverse all map points MapPoints
	    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
	    {
		MapPoint* pMP = vpPoints[iMP];

		// Discard Bad MapPoints and already found
		if(pMP->isBad() || spAlreadyFound.count(pMP))
		    continue;

		// Step 4: The map point is transferred to the current frame according to the transformation, under the camera coordinate system	
		// Get 3D Coords.
		// world coordinates of the map point
		cv::Mat p3Dw = pMP->GetWorldPos();
		// Transform into Camera Coords.
		cv::Mat p3Dc = Rcw*p3Dw + tcw;//  Go to the current frame, under the camera coordinate system
		// Depth must be positive
		if(p3Dc.at<float>(2) < 0.0)
		    continue;

		// Project into Image
		// Step 5: Project to the image pixel coordinate system of the current frame according to the internal parameters of the camera	
		const float invz = 1/p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// Point must be inside the image
		// The map point is projected, if it is not within the image range, there is no matching point
		if(!pKF->IsInImage(u,v))
		    continue;

		//Step 6: Determine whether the distance is within the scale covariance range
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw-Ow;
	    	// The distance of the map point from the camera and then infer the possible scale in the image pyramid, the farther the scale is, the smaller the scale is, and the closer the scale is.
		const float dist = cv::norm(PO);
		if(dist<minDistance || dist>maxDistance)
		    continue;
		// Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();
		if(PO.dot(Pn) < 0.5*dist)
		    continue;

		// Step 7: Determine the search radius according to the scale and then determine the candidate key points on the image	
		int nPredictedLevel = pMP->PredictScale(dist,pKF);//The image pyramid scale at which the prediction point is farther away
		// Search in a radius
		const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
		// Identify candidates and keypoints on an image
		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
		if(vIndices.empty())
		    continue;
		// Match to the most similar keypoint in the radius
		
		// Step 8: Traverse the candidate key points, map points and candidate key points on the key frame, perform descriptor matching, and calculate the matching distance retaining the closest distance
		const cv::Mat dMP = pMP->GetDescriptor();// Descriptors corresponding to map points
		int bestDist = 256;//distance ceiling
		int bestIdx = -1;
	        // Traverse all feature points in the search area and match the descriptor of the MapPoint
		// vector<size_t>::const_iterator
		for( auto vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
		{
		    const size_t idx = *vit;
		    if(vpMatched[idx])
			continue;
		    const int &kpLevel= pKF->mvKeysUn[idx].octave;
		    if(kpLevel < nPredictedLevel-1 || kpLevel > nPredictedLevel)
			continue;
	  
	      	    // Calculate the distance and save the key point corresponding to the shortest distance
		    const cv::Mat &dKF = pKF->mDescriptors.row(idx);// Descriptors corresponding to key points
		    const int dist = DescriptorDistance(dMP,dKF);

		    if(dist<bestDist)
		    {
			bestDist = dist;
			bestIdx = idx;
		    }
		}

		if(bestDist<=TH_LOW)// <50
		{
		    vpMatched[bestIdx]=pMP;// The map point to which the feature point matches
		    nmatches++;
		}
	    }

	    return nmatches;
	}

  
	  /**
	  * @brief   When the monocular is initialized, the key points of the first two frames are matched 2D-2D to calculate the transformation matrix. In frame 2, find matching points for the feature points of frame 1
	  * Project each vpPoints to the image pixel coordinate system of the reference key frame pKF, 
	  * and determine a search area according to the scale, and match the feature points in the area according to the descriptor of the MapPoint
	  * 
	  * If the matching error is less than TH_LOW, the matching is successful, and vpMatched is updated.
	  * @param  F1                   Frame         normal frame
	  * @param  F2		         Frame         normal frame
	  * @param  vbPrevMatched        Matches already in the previous frame F1
	  * @param  vpMatches12          Matching points of frame 1 feature points in frame 2
	  * @param  windowSize           Search area box size on frame 2  
	  * @return                      Number of successful matches
	  */
	int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
	{
	    int nmatches=0;
	    // Initialize the number of matching points in the frame for frame 1
	    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);
	    // Count the direction difference of matching point pairs, the same matching direction is not much different
	    vector<int> rotHist[HISTO_LENGTH];//  Angle histogram 30
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);// Histogram can record 500 points per bar
	    const float factor = 1.0f/HISTO_LENGTH;

	    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);// Matching point pair distance for frame 2
	    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);// Match point of frame 2
	    // Step 1: Find matching points in frame 2 for each keypoint in frame 1
	    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
	    {
		cv::KeyPoint kp1 = F1.mvKeysUn[i1];// Every keypoint of frame 1
		int level1 = kp1.octave;//The number of layers on the image pyramid
		if(level1 > 0)
		    continue;
		
	      	//Feature point candidate matching points corresponding to the square area on the 2 map
		vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

		if(vIndices2.empty())
		    continue;
		
		// Step 2: Descriptor matching to calculate distance	
	        // Get the feature descriptor of the keypoint of frame 1
		cv::Mat d1 = F1.mDescriptors.row(i1);// 1 Descriptor of the feature points of the graph
		int bestDist = INT_MAX;// initial minimum distance
		int bestDist2 = INT_MAX;
		int bestIdx2 = -1;
	        // vector<size_t>::iterator vit
	        // Traverse the possible feature points in frame 2
		for(auto vit = vIndices2.begin(); vit!=vIndices2.end(); vit++)// The key points corresponding to each of the corresponding regions in the 2 graphs
		{
		    size_t i2 = *vit;//key point subscript
		    // The feature descriptor corresponding to the key points of the candidate region of frame 2   
		    cv::Mat d2 = F2.mDescriptors.row(i2);// descriptor
		    // distance between descriptors    
		    int dist = DescriptorDistance(d1,d2);

		    if(vMatchedDistance[i2] <= dist)// The distance is too large, skip this point directly
			continue;
		    
		    // Step 3: Keep the matching points corresponding to the smallest and next smallest distances
		    if(dist<bestDist)//shortest distance
		    {
			bestDist2=bestDist;
			bestDist=dist;// smaller distance 
			bestIdx2=i2;
		    }
		    else if(dist<bestDist2)// next short distance
		    {
			bestDist2=dist;// smaller distance
		    }
		}

		// Step 4: Make sure the minimum distance is less than the threshold
		if(bestDist<=TH_LOW)//<50
		{
		     // trick!
		    // The best match is obviously better than the next best match, then the best match is really reliable
		    if(bestDist < (float)bestDist2*mfNNratio)
		    {
		      // If it has been matched, it means that the current feature has already been matched, then there will be two correspondences, remove the match
			if(vnMatches21[bestIdx2] >= 0)
			{
			    vnMatches12[ vnMatches21[bestIdx2] ] = -1;
			    nmatches--;
			}
			vnMatches12[i1]=bestIdx2;// frame 1 key point matched key point in frame 2 subscript
			vnMatches21[bestIdx2 ] = i1;// The subscript of the key point in frame 1 to which the key point of frame 2 is matched
			vMatchedDistance[bestIdx2]=bestDist;//distance
			nmatches++;

			if(mbCheckOrientation)
			{
			    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
			    if(rot<0.0)
				rot += 360.0f;
			    int bin = round(rot*factor);
			    if(bin==HISTO_LENGTH)
				bin=0;
			    assert(bin>=0 && bin<HISTO_LENGTH);
			    rotHist[bin].push_back(i1);//get direction histogram
			}
		    }
		}
	    }
	    
	    // Step 5: In this way, other angles that are too different can be eliminated.
	    if(mbCheckOrientation)
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
		// The three highest bins in the statistical histogram are retained, and matching points in other ranges are eliminated.
		// In addition, if the highest one is more than 10 times higher than the second highest, only the matching points in the highest bin are kept.
		// If the highest is more than 10 times higher than the third, then keep the matching points in the highest and second highest bins.
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
	
		for(int i=0; i<HISTO_LENGTH; i++)
		{
		    // No consideration is given to possible consistent directions
		    if(i==ind1 || i==ind2 || i==ind3)
			continue;
		    // Eliminate matches with inconsistent directions
		    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
		    {
			int idx1 = rotHist[i][j];
			if(vnMatches12[idx1]>=0)
			{
			    vnMatches12[idx1]=-1;
			    nmatches--;
			}
		    }
		}

	    }
	    // Step 6: Update Matches    
	    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
		if(vnMatches12[i1]>=0)// Frame 1 matches to frame 2
		    vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;//Corresponding to the feature point coordinates of frame 2

	    return nmatches;
	}

	  /**
	  * @brief Through the bag of words, the feature points of the key frames are tracked, and this function is used to match the feature points between two key frames during closed-loop detection.
	  * 
	  * Quickly match the feature points in pKF1 and pKF2 through bow (feature points that do not belong to the same node (word) directly skip the matching) \n
	  * Match feature points belonging to the same node by descriptor distance \n
	  * According to the match, update vpMatches12 \n
	  * Eliminate false matches by distance threshold, scale threshold and angle voting
	  * @param  pKF1               KeyFrame1
	  * @param  pKF2               KeyFrame2
	  * @param  vpMatches12        MapPoint in pKF2 that matches pKF1, null means no match
	  * @return                    Number of successful matches
	  */
	int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
	{
	    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;  // key frame 1 feature point
	    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;// Key frame 1 feature point dictionary description vector
	    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();// Map points matched by keyframe 1 feature points
	    const cv::Mat &Descriptors1 = pKF1->mDescriptors;// Descriptor matrix of keyframe 1 feature points

	    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;  // key frame 2 feature points
	    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;// Key frame 2 feature point dictionary description vector
	    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();// Map points matched by keyframe 2 feature points
	    const cv::Mat &Descriptors2 = pKF2->mDescriptors;// Descriptor matrix of keyframe 2 feature points

	    // map point for keyframe 1 initialization match point
	    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
	    vector<bool> vbMatched2(vpMapPoints2.size(),false);// keyframe map point match marker

	    // Count the direction difference of matching point pairs, the same matching direction is not much different
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;

	    int nmatches = 0;
	    
	  // 将属于同一节点(特定层)的ORB特征进行匹配
	    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
	    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
	    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
	    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	    while(f1it != f1end && f2it != f2end)
	    {
//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)     
		if(f1it->first == f2it->first)
		{
// 步骤2：遍历KF1中属于该node的特征点	  
		    for(size_t i1=0, iend1=f1it->second.size(); i1 < iend1; i1++)
		    {
			const size_t idx1 = f1it->second[i1];
		      // 取出KF1 中该特征对应的 地图点MapPoint
			MapPoint* pMP1 = vpMapPoints1[idx1];
			// 没有匹配的地图点跳过
			if(!pMP1)
			    continue;
			// 是坏点 跳过
			if(pMP1->isBad())
			    continue;
		    // 取出KF1中该特征对应的描述子
			const cv::Mat &d1 = Descriptors1.row(idx1);
			int bestDist1=256;// 最好的距离（最小距离）
			int bestIdx2 =-1 ;
			int bestDist2=256; // 倒数第二好距离（倒数第二小距离）
			
// 步骤3：遍历KF2中属于该node的特征点，找到了最佳匹配点
			for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
			{
			    const size_t idx2 = f2it->second[i2];
			  // 对应 的 地图点
			    MapPoint* pMP2 = vpMapPoints2[idx2];
		      // 已经和 KF1 中某个点匹配过了 跳过  或者 该特征点 无匹配地图点 或者 该地图点是坏点  跳过
			    if(vbMatched2[idx2] || !pMP2)
				continue;
			    if(pMP2->isBad())
				continue;
// 步骤4：求描述子的距离 保留最小和次小距离对应的 匹配点
			    const cv::Mat &d2 = Descriptors2.row(idx2);// 取出F中该特征对应的描述子
			    int dist = DescriptorDistance(d1,d2);
			    if(dist<bestDist1)// dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
			    {
				bestDist2=bestDist1;
				bestDist1=dist;// 最段的距离
				bestIdx2=idx2;// 对应  KF2 地图点 下标
			    }
			    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，更新bestDist2
			    {
				bestDist2=dist;// 次短的距离
			    }
			}
			
// 步骤4：根据阈值 和 角度投票剔除误匹配
			if(bestDist1<TH_LOW)
			{
			    // trick!
			    // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
			    if(static_cast<float>(bestDist1) < mfNNratio*static_cast<float>(bestDist2))
			    {
				vpMatches12[idx1]=vpMapPoints2[bestIdx2];// 匹配到的 对应  KF2 中的 地图点 
				vbMatched2[bestIdx2]=true;// KF2 中的地图点 已经和 KF1中的某个地图点匹配

				if(mbCheckOrientation)
				{
				  // trick!
				  // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
				  // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
				    float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;//匹配点方向差
				    if(rot<0.0)
					rot+=360.0f;
				    int bin = round(rot*factor);
				    if(bin==HISTO_LENGTH)
					bin=0;
				    assert(bin>=0 && bin<HISTO_LENGTH);
				    rotHist[bin].push_back(idx1);//匹配点方向差 直方图
				}
				nmatches++;
			    }
			}
		    }

		    f1it++;
		    f2it++;
		}
		else if(f1it->first < f2it->first)
		{
		    f1it = vFeatVec1.lower_bound(f2it->first);
		}
		else
		{
		    f2it = vFeatVec2.lower_bound(f1it->first);
		}
	    }
	    
// 根据方向差一致性约束 剔除误匹配的点
	    if(mbCheckOrientation)
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
	// 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
	// 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
	// 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		for(int i=0; i<HISTO_LENGTH; i++)
		{
		  // 如果特征点的旋转角度变化量属于这三个组，则保留 该匹配点对
		    if(i==ind1 || i==ind2 || i==ind3)
			continue;
		    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
		    {
		      // 将除了ind1 ind2 ind3以外的匹配点去掉
			vpMatches12[rotHist[i][j]] =static_cast<MapPoint*>(NULL);
			nmatches--;
		    }
		}
	    }

	    return nmatches;
	}


  /**
  * @brief 利用基本矩阵F12，在两个关键帧之间  两帧特征点均未有匹配的地图点 中 产生 2d-2d 匹配点对  
  * 关键帧1的每个特征点 与 关键帧2 特征点 同属于 词典同一个node(包含多的类似单词) 
  * 节点下的 特征点进行描述子匹配，在符合对极几何约束的条件下，选择距离最近的匹配
  * 最后在进行 匹配点 方向差 一致性约束 检测 去除一些误匹配
  * @param pKF1          关键帧1
  * @param pKF2          关键帧2
  * @param F12            基础矩阵F    p2转置 × F  × p1 = 0
  * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
  * @param bOnlyStereo     在双目和rgbd情况下，要求特征点在右图存在匹配
  * @return                            成功匹配的数量
  */
	int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
					      vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
	{    
	    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;// 关键帧pKF1  描述子 的 词典向量表示
	    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;// 关键帧pKF2  描述子 的 词典向量表示

	    //Compute epipole in second image
	// 计算KF1的相机中心在KF2图像平面的坐标，即极点坐标   
	    cv::Mat Cw = pKF1->GetCameraCenter();// KF1 O1 世界坐标系下
	    cv::Mat R2w = pKF2->GetRotation();// 世界坐标系 -----> KF2 旋转矩阵
	    cv::Mat t2w = pKF2->GetTranslation();// 世界坐标系 -----> KF2 平移向量
	    cv::Mat C2 = R2w*Cw+t2w;// KF1 O1 在 KF2坐标系下坐标
	    const float invz = 1.0f/C2.at<float>(2);//深度归一化坐标
	    // KF1 O1 投影到KF2像素坐标系上
	    const float ex =pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;//
	    const float ey =pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;//

	    // Find matches between not tracked keypoints
	    // Matching speed-up by ORB Vocabulary
	    // Compare only ORB that share the same node

	    int nmatches=0;
	    vector<bool> vbMatched2(pKF2->N,false);// pKF2 关键帧2  地图点是否被 pKF1 地图点匹配标志
	    vector<int> vMatches12(pKF1->N,-1);// 帧1 pKF1 地图点 在pKF2中的 匹配 地图点

	    // 匹配点 方向差 一致性约束
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;
	    
	    // 将属于同一节点(特定层)的ORB特征进行匹配
	    // FeatureVector的数据结构类似于：{(node1,feature_vector1) (node2,feature_vector2)...}
	    // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
	    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
	    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
	    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
	    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
	    
// 步骤1：遍历pKF1和pKF2中词典线性表示的特征向量树中 的node节点
	    while(f1it!=f1end && f2it!=f2end)
	    {
	      // 如果f1it和f2it属于同一个node节点
	      // 分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)     
		if(f1it->first == f2it->first)
		{
// 步骤2：遍历该node节点下关键帧1 pKF1 (f1it->first)的所有特征点	  
		    for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
		    {
			// 获取pKF1中属于该node节点的所有特征点索引
			const size_t idx1 = f1it->second[i1];
		// 步骤2.1：通过特征点索引idx1在pKF1中取出对应的 地图点 MapPoint      
			MapPoint* pMP1 = pKF1->GetMapPoint(idx1);    
			// If there is already a MapPoint skip
			// 特征点已经存在 地图点不用计算了 直接跳过
			// 由于寻找的是未匹配的特征点，所以pMP1应该为NULL
			if(pMP1)
			    continue;
			// 如果mvuRight（右图像 有匹配点）中的值大于0，表示是双目/深度 ，且该特征点有深度值
			const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;
			if(bOnlyStereo)
			    if(!bStereo1)
				continue;// 非双目/深度 跳过
		  // 步骤2.2：通过特征点索引idx1在pKF1中取出对应的特征点     
			const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
			
		  // 步骤2.3：通过特征点索引idx1在pKF1中取出对应的特征点的描述子	
			const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
			
			int bestDist = TH_LOW;//50
			int bestIdx2 = -1;//匹配点 下标
// 步骤3：遍历该node节点下关键帧2 pKF2 (f2it->first)的所有特征点         
			for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
			{
			  // 获取pKF2中属于该node节点的所有特征点索引
			    size_t idx2 = f2it->second[i2];
			    
		      // 步骤3.1：通过特征点索引idx2在pKF2中取出对应的MapPoint  
			    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);      
			    // If we have already matched or there is a MapPoint skip
			    // 如果关键帧2 pKF2当前特征点索引idx2已经被匹配过或者对应的3d点非空
			    // 那么这个索引idx2就不能被考虑
			    if(vbMatched2[idx2] || pMP2)
				continue;// pMP2 的特征点 也不能有匹配的地图点 有的话 就已经匹配过了 生成了地图点了
			    // 如果mvuRight（右图像 有匹配点）中的值大于0，表示是双目/深度 ，且该特征点有深度值
			    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;
			    if(bOnlyStereo)
				if(!bStereo2)
				    continue;
				
		    // 步骤3.2：通过特征点索引idx2在pKF2中取出对应的特征点的描述子        
			    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
			    
		    // 计算idx1与idx2在两个关键帧中对应特征点的描述子距离        
			    const int dist = DescriptorDistance(d1,d2);
			    
			    if(dist>TH_LOW || dist>bestDist)// 距离过大 直接跳过
				continue;
			    
		  // 步骤3.3：通过特征点索引idx2在pKF2中取出对应的特征点
			    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

			    if(!bStereo1 && !bStereo2)
			    {
			    // KF1 O1 投影到KF2像素坐标系上  ex  ey
				const float distex = ex - kp2.pt.x;
				const float distey = ey - kp2.pt.y;
				// 该特征点距离极点 太近，表明kp2对应的MapPoint距离pKF1相机太近
				if(distex*distex + distey*distey < 100 * pKF2->mvScaleFactors[kp2.octave])
				    continue;
			    }
			    
// 步骤4：计算特征点kp2 到 kp1极线（kp1对应pKF2的一条极线）的距离是否小于阈值
			    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
			    {
				bestIdx2 = idx2;// 保留匹配点
				bestDist = dist;
			    }
			}
	// 步骤1、2、3、4总结下来就是：将左图像的每个特征点与右图像同一node节点的所有特征点
	// 依次检测，判断是否满足对极几何约束，满足约束就是匹配的特征点
			if(bestIdx2 >= 0)// KF1 特征点 在 KF2中匹配点的下标 初始化为 -1
			  // > 0  找到了匹配点
			{
			    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];// 匹配点在 KF2中的 像素坐标
			    vMatches12[idx1]=bestIdx2;
			    nmatches++;
// 步骤5：    匹配点 方向差 一致性约束
			    if(mbCheckOrientation)
			    {
				float rot = kp1.angle-kp2.angle;// 匹配点 方向差
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(idx1);// 匹配点 方向差 直方图
			    }
			}
		    }

		    f1it++;
		    f2it++;
		}
		else if(f1it->first < f2it->first)
		{
		    f1it = vFeatVec1.lower_bound(f2it->first);
		}
		else
		{
		    f2it = vFeatVec2.lower_bound(f1it->first);
		}
	    }

	// 根据方向差一致性约束 剔除误匹配的点
	    if(mbCheckOrientation)
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
	// 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
	// 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
	// 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		for(int i=0; i<HISTO_LENGTH; i++)
		{
		  // 方向差一致性最高的匹配点 保留
		    if(i==ind1 || i==ind2 || i==ind3)
			continue;
		  // 其他匹配点 清除
		    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
		    {
			vMatches12[rotHist[i][j]]=-1;
			nmatches--;
		    }
		}

	    }

	    vMatchedPairs.clear();
	    vMatchedPairs.reserve(nmatches);

	    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
	    {
		if(vMatches12[i]<0)// 无匹配点
		    continue;
		vMatchedPairs.push_back(make_pair(i,vMatches12[i]));//保留匹配点对关系
	    }

	    return nmatches;
	}


  /**
  * @brief 将MapPoints投影到关键帧pKF中，并判断是否有重复的MapPoints
  * 1.如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
  * 2.如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint
  * @param  pKF          相邻关键帧
  * @param  vpMapPoints 需要融合的 当前帧上的 MapPoints
  * @param  th             搜索半径的因子
  * @return                   重复MapPoints的数量
  */
	int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
	{
	    // 关键帧 的 旋转矩阵 和 平移矩阵  欧式变换
	    cv::Mat Rcw = pKF->GetRotation();
	    cv::Mat tcw = pKF->GetTranslation();
	    // 相机内参数
	    const float &fx = pKF->fx;
	    const float &fy = pKF->fy;
	    const float &cx = pKF->cx;
	    const float &cy = pKF->cy;
	    const float &bf = pKF->mbf;// 基线×f
	    // 关键帧 的 相机坐标中心点坐标 
	    cv::Mat Ow = pKF->GetCameraCenter();

	    int nFused=0;// 融合地图点的数量
	    const int nMPs = vpMapPoints.size();// 需要融合的 地图点数量
	//步骤1：  遍历所有的MapPoints
	    for(int i=0; i<nMPs; i++)
	    {
		MapPoint* pMP = vpMapPoints[i];// 地图点
		
	//步骤2： 跳过不好的地图点
		if(!pMP)// 不存在
		    continue;
		// 地图点 是坏点  地图点 被 关键帧观测到 已经匹配好了 不用 融合
		if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
		    continue;

	// 步骤3： 将地图点 投影在 关键帧 图像像素坐标上	
		cv::Mat p3Dw = pMP->GetWorldPos();// 地图点在 世界坐标系下的坐标
		cv::Mat p3Dc = Rcw*p3Dw + tcw;// 地图点在 关键帧下相机坐标系下的 坐标
		// Depth must be positive
		// 深度值必须为正  在 帧(相机)的前方
		if(p3Dc.at<float>(2)<0.0f)
		    continue;
		const float invz = 1/p3Dc.at<float>(2);// 深度归一化 因子
		const float x = p3Dc.at<float>(0)*invz;// 相机归一化尺度下的坐标
		const float y = p3Dc.at<float>(1)*invz;
		// 像素坐标
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// 像素坐标 必须在 图像尺寸范围内 Point must be inside the image
		if(!pKF->IsInImage(u,v))
		    continue;
		
	//步骤4：  判断距离是否在尺度协方差范围内
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw-Ow;
		//地图点 距离相机的距离 进而推断 在图像金字塔中可能的尺度 越远尺度小 越近尺度大
		const float dist3D = cv::norm(PO);
		// Depth must be inside the scale pyramid of the image
		if(dist3D<minDistance || dist3D>maxDistance )
		    continue;//  剔除

	//步骤5：观察视角 必须小于 60度  Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();
		if(PO.dot(Pn)<0.5*dist3D)
		    continue;
		// 根据深度 预测 地图点 在 帧 图像上 的尺度   深度大尺度小  深度小尺度大
		int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

	// 步骤6： 根据尺度确定搜索半径 进而在图像上确定 候选 关键点	
		const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
		if(vIndices.empty())
		    continue;
		
	//  步骤7：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();// 地图点描述子
		int bestDist = 256;
		int bestIdx = -1;
		// vector<size_t>::const_iterator 
		for(auto vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
		{
		    const size_t idx = *vit;
		//  关键点的尺度 需要在 预测尺度 之上
		    const cv::KeyPoint &kp = pKF->mvKeysUn[idx];// 关键帧 候选关键点
		    const int &kpLevel= kp.octave;// 关键点尺度
		    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
			continue;
	// 步骤8：计算MapPoint投影的坐标与这个区域特征点的距离，如果偏差很大，直接跳过特征点匹配	    
		//   深度/双目相机  有 右图像 匹配点横坐标差值
		    if(pKF->mvuRight[idx]>=0)
		    {
			const float ur = u - bf*invz;//和 匹配点 横坐标  深度相机和 双目相机 有
			// Check reprojection error in stereo
			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float &kpr = pKF->mvuRight[idx];
			const float ex = u - kpx;// 横坐标差值
			const float ey = v - kpy;//纵坐标差值
			const float er = ur - kpr;// 右图像 匹配点横坐标差值 
			const float e2 = ex*ex + ey*ey + er*er;

			if(e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
			    continue;// 差值过大 直接跳过
		    }
	      //  单目相机      无右图像 匹配点横坐标差值
		    else
		    {
			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float ex = u-kpx;
			const float ey = v-kpy;
			const float e2 = ex*ex+ey*ey;
			    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
			if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
			    continue;
		    }
	// 步骤9：计算地图点和 关键帧 特征点 描述子之间的距离 选出最近距离的 关键点
		    const cv::Mat &dKF = pKF->mDescriptors.row(idx);// 关键帧 特征点 描述子
		    const int dist = DescriptorDistance(dMP,dKF);// 地图点和 关键帧 特征点 描述子之间的距离
		    if(dist<bestDist)
		    {
			bestDist = dist;// 最短的距离
			bestIdx = idx;// 对应的 特征点 下标
		    }
		}

		// If there is already a MapPoint replace otherwise add new measurement
		// 找到了地图点MapPoint在该区域最佳匹配的特征点        
		if(bestDist<=TH_LOW)// 距离<50
		{
		  // 该帧 上 特征点 对应的 地图点 初始化值为 null指针
		    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
// 步骤10： 如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）   
		  // 本身已经 匹配到 地图点
		    if(pMPinKF)
		    {
			if(!pMPinKF->isBad())//是好点
			{
		// 地图点 和 关键帧  特征点对应的地图点 匹配了 保留被观测到次数多的 地图点
			    if(pMPinKF->Observations() > pMP->Observations())//帧地图点观测次数多 保留帧地图点
				pMP->Replace(pMPinKF);//原地图点 用帧地图点替代
			    else
				pMPinKF->Replace(pMP);// 帧地图点替代用 原地图点替代
			}
		    }
// 步骤11：  如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint           
		  //    关键帧  特征点 还没有匹配的地图点  把匹配到的地图点 对应上去
		    else
		    {
			pMP->AddObservation(pKF,bestIdx);// pMP 地图点 观测到了 帧pKF 上第 bestIdx 个 特征点
			pKF->AddMapPoint(pMP,bestIdx);// 帧 的 第 bestIdx 个 特征点 对应pMP地图点
		    }
		    nFused++;// 融合次数++ 
		}
	    }
	    return nFused;
	}


  /**
  * @brief 将MapPoints投影到 关键帧pKF 中，并判断是否有重复的MapPoints
  * Scw为世界坐标系到pKF机体坐标系的Sim3 相似变换变换 ，
  * 需要先将相似变换转换到欧式变换SE3 下  将世界坐标系下的vpPoints变换到机体坐标系
  * 1 地图点匹配到 帧 关键点 关键点有对应的地图点时， 用帧关键点对应的地图点 替换 原地图点
  * 2 地图点匹配到 帧 关键点 关键点无对应的地图点时，为该特征点 添加匹配到的地图点MapPoint
  * @param  pKF          相邻关键帧
  * @param  Scw          世界坐标系到pKF机体坐标系的Sim3 相似变换变换  [s*R t]
  * @param  vpPoints 需要融合的 地图点 MapPoints
  * @param  th             搜索半径的因子
  *@param    vpReplacePoint
  * @return                   重复MapPoints的数量
  */
	int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
	{
	    // Get Calibration Parameters for later projection
	    // 相机内参数
	    const float &fx = pKF->fx;
	    const float &fy = pKF->fy;
	    const float &cx = pKF->cx;
	    const float &cy = pKF->cy;

	    // Decompose Scw
	    // 相似变换Sim3 转换到 欧式变换SE3
	    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
	    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));//相似变换里的 旋转矩阵的 相似尺度因子
	    cv::Mat Rcw = sRcw/scw;// 欧式变换 里 的旋转矩阵
	    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;// 欧式变换 里 的 平移向量
	    cv::Mat Ow = -Rcw.t()*tcw;//相机中心在 世界坐标系下的 坐标

	    // Set of MapPoints already found in the KeyFrame
	    // 关键帧已有的 匹配地图点
	    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();
	    int nFused=0;// 融合计数
	    const int nPoints = vpPoints.size();// 需要融合的 地图点 数量

	    // For each candidate MapPoint project and match
// 步骤1： 遍历所有需要融合的 地图点 MapPoints   
	    for(int iMP=0; iMP<nPoints; iMP++)
	    {
		MapPoint* pMP = vpPoints[iMP];// 需要融合的 地图点 
//步骤2： 跳过不好的地图点
		if(!pMP)// 不存在
		    continue;
		// Discard Bad MapPoints and already found
		// 地图点 是坏点  地图点 被 关键帧观测到 已经匹配好了 不用 融合
		if(pMP->isBad() || spAlreadyFound.count(pMP))
		    continue; 
//步骤3： 地图点 投影到 关键帧 像素平面上 不在平面内的 不考虑
		// Get 3D Coords.
		cv::Mat p3Dw = pMP->GetWorldPos();// 地图点世界坐标系坐标
		// Transform into Camera Coords.
		cv::Mat p3Dc = Rcw*p3Dw+tcw;// 地图点在 帧坐标系 下的 坐标
		// Depth must be positive
		if(p3Dc.at<float>(2)<0.0f)// 地图点在相机前方 深度不能为负值
		    continue;
		// Project into Image 投影到像素平面
		const float invz = 1.0/p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// Point must be inside the image
		if(!pKF->IsInImage(u,v))// 不在图像内 跳过
		    continue;
	//步骤4：  判断距离是否在尺度协方差范围内
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw-Ow;
		//地图点 距离相机的距离 进而推断 在图像金字塔中可能的尺度 越远尺度小 越近尺度大
		const float dist3D = cv::norm(PO);
		// Depth must be inside the scale pyramid of the image
		if(dist3D<minDistance || dist3D>maxDistance )
		    continue;//  剔除

	//步骤5：观察视角 必须小于 60度  Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();
		if(PO.dot(Pn)<0.5*dist3D)
		    continue;
		// 根据深度 预测 地图点 在 帧 图像上 的尺度   深度大尺度小  深度小尺度大
		int nPredictedLevel = pMP->PredictScale(dist3D,pKF);
	// 步骤6： 根据尺度确定搜索半径 进而在图像上确定 候选 关键点	
		const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
		if(vIndices.empty())
		    continue;

//  步骤7：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();// 地图点 对应的 描述子
		int bestDist = INT_MAX;
		int bestIdx = -1;
		// vector<size_t>::const_iterator
		for(auto vit=vIndices.begin(); vit!=vIndices.end(); vit++)
		{
		    const size_t idx = *vit;
		    //  关键点的尺度 需要在 预测尺度 之上
		    const int &kpLevel = pKF->mvKeysUn[idx].octave;
		    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
			continue;// 不符合的 跳过
			
// 步骤8：计算地图点和 关键帧 特征点 描述子之间的距离 选出最近距离的 关键点
		    const cv::Mat &dKF = pKF->mDescriptors.row(idx);// 关键点描述子
		    int dist = DescriptorDistance(dMP,dKF);// 地图点 对应的 描述子 和  关键点描述子 之间的 汉明匹配距离
		    if(dist<bestDist)// 最近的距离
		    {
			bestDist = dist;
			bestIdx = idx;// 对应的 描述子的 下标
		    }
		}

		// 找到了地图点MapPoint在该区域最佳匹配的特征点        
		if(bestDist<=TH_LOW)// 距离<50
		{
		    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
// 步骤10： 如果MapPoint能匹配关键帧的特征点，并且该特征点有对应的MapPoint，
		    if(pMPinKF)
		    {
			if(!pMPinKF->isBad())
			    vpReplacePoint[iMP] = pMPinKF;// 用关键点对应的地图点 替换 原地图点
		    }
// 步骤11：  如果MapPoint能匹配关键帧的特征点，并且该特征点没有对应的MapPoint，那么为该特征点点添加地图点MapPoint           
		  //    关键帧  特征点 还没有匹配的地图点  把匹配到的地图点 对应上去		    
		    else
		    {
			pMP->AddObservation(pKF,bestIdx);// pMP 地图点 观测到了 帧pKF 上第 bestIdx 个 特征点
			pKF->AddMapPoint(pMP,bestIdx);// 帧 的 第 bestIdx 个 特征点 对应pMP地图点		      
		    }
		    nFused++;
		}
	    }

	    return nFused;
	}
	
  /**
  * @brief  通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，
  * 同理，确定pKF2的特征点在pKF1中的大致区域
  * 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，
  * 更新vpMatches12（之前使用SearchByBoW进行特征点匹配时会有漏匹配）
  * @param pKF1          关键帧1
  * @param pKF2          关键帧2
  * @param vpMatches12   两帧原有匹配点  帧1 特征点 匹配到 帧2 的地图点
  * @param s12              帧2->帧1 相似变换 尺度
  * @param R12             帧2->帧1  欧式变换 旋转矩阵
  * @param t12              帧2->帧1 欧式变换 平移向量
  * @param th       		 搜索半径参数
  * @return                     成功匹配的数量
  */
	int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
				    const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
	{
	  
// 步骤1：变量初始化----------------------------------------------------- 
	  //相机内参数
	    const float &fx = pKF1->fx;
	    const float &fy = pKF1->fy;
	    const float &cx = pKF1->cx;
	    const float &cy = pKF1->cy;
	    
	    // 世界坐标系 到 帧1 的 欧式变换 Camera 1 from world
	    cv::Mat R1w = pKF1->GetRotation();
	    cv::Mat t1w = pKF1->GetTranslation();
	    
	   // 世界坐标系 到 帧2 的 欧式变换 Camera 2 from world
	    cv::Mat R2w = pKF2->GetRotation();
	    cv::Mat t2w = pKF2->GetTranslation();

	    //Transformation between cameras
	    // 相似变换 旋转矩阵 平移向量
	    cv::Mat sR12 = s12*R12;// 帧2->帧1 相似变换旋转矩阵 = 帧2->帧1相似变换尺度 * 帧2->帧1欧式变换旋转矩阵
	    cv::Mat sR21 = (1.0/s12)*R12.t();// 帧1->帧2相似变换旋转矩阵 = 帧1->帧2相似变换尺度 * 帧1->帧2欧式变换旋转矩阵
	    cv::Mat t21 = -sR21*t12;// 帧1->帧2相似变换 平移向量
	    
            // 帧1地图点数量  关键点数量 
	    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	    const int N1 = vpMapPoints1.size();
            // 帧2地图点数量 关键点数量
	    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
	    const int N2 = vpMapPoints2.size();

	    // 来源于 两帧 先前 已有的 匹配
	    vector<bool> vbAlreadyMatched1(N1,false);// 帧1 在帧2中 是否有 匹配
	    vector<bool> vbAlreadyMatched2(N2,false);// 帧2 在帧1中 是否有 匹配
	    
// 步骤2：用vpMatches12更新 已有的匹配 vbAlreadyMatched1和vbAlreadyMatched2------------------------------------
	    for(int i=0; i<N1; i++)
	    {
	        MapPoint* pMP = vpMatches12[i];// 帧1 特征点 匹配到 帧2 的地图点
		if(pMP)// 存在
		{
		    vbAlreadyMatched1[i]=true;//  帧1 特征点  已经有匹配到的 地图点了
		    int idx2 = pMP->GetIndexInKeyFrame(pKF2);//  帧2 的地图点 在帧2中对应的 下标
		    if(idx2>=0 && idx2<N2)// 在 帧2特征点个数范围内的话
			vbAlreadyMatched2[idx2]=true;// 帧2 地图点 在 帧1中也已经有匹配
		}
	    }
	    
	    // 新寻找的匹配
	    vector<int> vnMatch1(N1,-1);
	    vector<int> vnMatch2(N2,-1);
// 步骤3：通过Sim变换，确定pKF1的地图点在pKF2帧图像中的大致区域，
	    //         在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12
	    //         （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
	    // 每一个帧1中的地图点 投影到 帧2 上
	    for(int i1=0; i1<N1; i1++)
	    {
       //步骤3.1： 跳过已有的匹配 和 不存在的点 以及坏点 
		MapPoint* pMP = vpMapPoints1[i1];
		if(!pMP || vbAlreadyMatched1[i1])
		    continue;// 点不存在 已经匹配过了 直接跳过
		if(pMP->isBad())
		    continue;// 坏点跳过               SE3						Sim3
       //步骤3.2： 帧1地图点(世界坐标系)-------> 帧1地图点(帧1坐标系)-------> 帧1地图点(帧2坐标系)---->帧2像素坐标系下
		// 帧1  pKF1 地图点在世界坐标系中的点坐标  
		cv::Mat p3Dw = pMP->GetWorldPos();// 帧1地图点(世界坐标系)
		// 帧1   pKF1 地图点在帧1 坐标系下的点坐标
		cv::Mat p3Dc1 = R1w*p3Dw + t1w;// 帧1地图点(帧1坐标系)
		//  帧1   pKF1 地图点在帧1 坐标系下的点坐标 通过帧1到帧2的相似变换 变换到 帧2坐标系下
		cv::Mat p3Dc2 = sR21*p3Dc1 + t21;// 帧1地图点(帧2坐标系)
		// Depth must be positive
		if(p3Dc2.at<float>(2)<0.0)// 深度值必须为正 相机前方
		    continue;
		// 投影到 帧2 像素平面上
		const float invz = 1.0/p3Dc2.at<float>(2);
		const float x = p3Dc2.at<float>(0)*invz;
		const float y = p3Dc2.at<float>(1)*invz;
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// Point must be inside the image
		if(!pKF2->IsInImage(u,v))// 坐标必须在 图像平面尺寸内
		    continue;
		
       //步骤3.3：  判断帧1地图点距帧2的距离 是否在尺度协方差范围内
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const float dist3D = cv::norm(p3Dc2);
		// Depth must be inside the scale pyramid of the image
		if(dist3D<minDistance || dist3D>maxDistance )
		    continue;//  剔除
		    
       // 步骤3.4： 根据深度确定尺度 再根据 尺度确定搜索半径 进而在图像上确定 候选 关键点		    
		// Compute predicted octave 根据深度 预测 地图点 在 帧 图像上 的尺度   深度大尺度小  深度小尺度大
		const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);// 尺度 也就是在 金字塔哪一层
		// Search in a radius
		const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];// 再根据 尺度确定搜索半径
		const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);//进而在图像上确定 候选 关键点	
		if(vIndices.empty())
		    continue;
       // 步骤3.5：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();// 帧1 地图点 描述子 
		int bestDist = INT_MAX;
		int bestIdx = -1;
		// 遍历搜索 帧2区域内的所有特征点，与帧1地图点pMP进行描述子匹配
		// vector<size_t>::const_iterator
		for(auto  vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
		{
		    const size_t idx = *vit;
		   //  关键点的尺度 需要在 预测尺度nPredictedLevel 之上
		    const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];// 帧2候选区域内的 关键点
		    if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
			continue;
		   // 帧2 关键点描述子 
		    const cv::Mat &dKF = pKF2->mDescriptors.row(idx);
		    const int dist = DescriptorDistance(dMP,dKF);// 帧1 地图点描述子 和 帧2关键点 描述子 距离
		    if(dist<bestDist)
		    {
			bestDist = dist;
			bestIdx = idx;
		    }
		}
		if(bestDist<=TH_HIGH)// <=100
		{
		    vnMatch1[i1]=bestIdx;// 帧1  地图点 匹配到的 帧2 的关键点(也对应一个地图点)
		}
	    }

// 步骤4：通过Sim变换，确定pKF2的地图点在pKF1帧图像中的大致区域，
	    //         在该区域内通过描述子进行匹配捕获pKF2和pKF1之前漏匹配的特征点，更新vpMatches12
	    //         （之前使用SearchByBoW进行特征点匹配时会有漏匹配）
	    // 每一个帧2中的地图点 投影到 帧1 上
	    for(int i2=0; i2<N2; i2++)
	    {
       //步骤4.1： 跳过已有的匹配 和 不存在的点 以及坏点 
		MapPoint* pMP = vpMapPoints2[i2];// 帧2 关键点匹配的 地图点
		if(!pMP || vbAlreadyMatched2[i2])// 不存在匹配的地图点 或者 已经和 帧1匹配了 跳过
		    continue;
		if(pMP->isBad())// 帧2地图点是坏点
		    continue;// 坏点跳过               SE3					   Sim3
       //步骤4.2： 帧2地图点(世界坐标系)-------> 帧2地图点(帧2坐标系)-------> 帧2地图点(帧1坐标系)---->帧1像素坐标系下
		cv::Mat p3Dw = pMP->GetWorldPos();// 帧2  pKF1 地图点在世界坐标系中的点坐标 
		cv::Mat p3Dc2 = R2w*p3Dw + t2w; //  帧2地图点(帧2坐标系)
		cv::Mat p3Dc1 = sR12*p3Dc2 + t12;//  帧2地图点(帧1坐标系) 相似变换
		// Depth must be positive
		if(p3Dc1.at<float>(2)<0.0)// 深度值 为正
		    continue;
		// 帧2地图点 投影到帧1 像素平面上
		const float invz = 1.0/p3Dc1.at<float>(2);
		const float x = p3Dc1.at<float>(0)*invz;
		const float y = p3Dc1.at<float>(1)*invz;
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// Point must be inside the image
		if(!pKF1->IsInImage(u,v))// 必须在 图像平面内
		    continue;
       //步骤4.3：  判断帧2地图点距帧1的距离 是否在尺度协方差范围内
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const float dist3D = cv::norm(p3Dc1);
		// Depth must be inside the scale pyramid of the image
		if(dist3D<minDistance || dist3D>maxDistance)
		    continue;
       // 步骤4.4： 根据深度确定尺度 再根据 尺度确定搜索半径 进而在图像上确定 候选 关键点		    
		// Compute predicted octave 根据深度 预测 地图点 在 帧 图像上 的尺度   深度大尺度小  深度小尺度大
		const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);// 尺度
		// Search in a radius of 2.5*sigma(ScaleLevel)
		const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];// 半径
		const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);// 在搜索区域的 候选点
		if(vIndices.empty())
		    continue;
		
       // 步骤4.5：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
		// Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();// 帧2 地图点描述子
		int bestDist = INT_MAX;
		int bestIdx = -1;
		// vector<size_t>::const_iterator
		// 遍历搜索 帧1区域内的所有特征点，与帧2地图点pMP进行描述子匹配
		for(auto vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
		{
		    const size_t idx = *vit;
		    //  关键点的尺度 需要在 预测尺度nPredictedLevel 之上
		    const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];
		    if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
			continue;
		    // 帧1 关键点描述子 
		    const cv::Mat &dKF = pKF1->mDescriptors.row(idx);
		    // 帧2 地图点描述子 和 帧1关键点 描述子 距离
		    const int dist = DescriptorDistance(dMP,dKF);
		    if(dist<bestDist)
		    {
			bestDist = dist;
			bestIdx = idx;
		    }
		}

		if(bestDist<=TH_HIGH)// <100
		{
		    vnMatch2[i2]=bestIdx;// 帧2  地图点 匹配到的 帧1 的关键点(也对应一个地图点)
		}
	    }

	    // Check agreement
// 步骤5 检查两者的匹配 是否对应起来  
	    int nFound = 0;

	    for(int i1=0; i1<N1; i1++)
	    {
		int idx2 = vnMatch1[i1];// 帧1  地图点 匹配到的 帧2 的关键点(也对应一个地图点) 下标

		if(idx2>=0)// 帧1  地图点 匹配到的 帧2 关键点 下标
		{
		    int idx1 = vnMatch2[idx2];// 帧2 关键点  对应的 帧1 地图点下标
		    if(idx1==i1)// 匹配 相互符合
		    {
			vpMatches12[i1] = vpMapPoints2[idx2];// 更新帧1 在帧2 中匹配的 地图点
			nFound++;
		    }
		}
	    }

	    return nFound;
	}
	

	// b. 匹配上一帧的地图点，即前后两帧匹配，用于TrackWithMotionModel
	//运动模型（Tracking with motion model）跟踪   速率较快  假设物体处于匀速运动
	// 用 上一帧的位姿和速度来估计当前帧的位姿使用的函数为TrackWithMotionModel()。
	//这里匹配是通过投影来与上一帧看到的地图点匹配，使用的是matcher.SearchByProjection()。
/**
 * @brief 通过投影，对上一帧的特征点(地图点)进行跟踪
 * 运动跟踪模式
 * 上一帧中包含了MapPoints，对这些MapPoints进行跟踪tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame 当前帧
 * @param  LastFrame       上一帧
 * @param  th                      搜索半径参数
 * @param  bMono             是否为单目
 * @return                           成功匹配的数量
 * @see SearchByBoW()
 */	
	int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
	{
	    int nmatches = 0;
// 步骤1：变量初始化----------------------------------------------------------
	    // Rotation Histogram (to check rotation consistency)
	   // 匹配点 观测方向差 直方图 统计 用来筛选 最好的 匹配
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;

	 // 当前帧 旋转 平移矩阵	    
	    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat twc = -Rcw.t()*tcw;// // twc(w)
	// 上一帧 旋转 平移矩阵
	    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat tlc = Rlw*twc + tlw;//当前帧到上一帧的 平移向量
	    
	// 判断前进还是后退
	    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	     // 非单目情况，如果Z>0且大于基线，则表示前进
	    
	    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	    // 非单目情况，如果Z<0,且绝对值大于基线，则表示前进
	    
// 步骤2：遍历 上一帧 所有的关键点(对应 地图点)-------------------------------------------
	    for(int i=0; i<LastFrame.N; i++)
	    {
		MapPoint* pMP = LastFrame.mvpMapPoints[i];//上一帧  地图点

		if(pMP)// 地图点存在
		{
		    if(!LastFrame.mvbOutlier[i])// 该地图点也不是外点 是内点 复合变换关系的点
		    {
			// Project
// 步骤3： 上一帧  地图点 投影到 当前帧 像素平面上-----------------------------------------    
			cv::Mat x3Dw = pMP->GetWorldPos();// 上一帧地图点（世界坐标系下）
			cv::Mat x3Dc = Rcw*x3Dw+tcw;//上一帧地图点（当前帧坐标系下）
			const float xc = x3Dc.at<float>(0);
			const float yc = x3Dc.at<float>(1);
			const float invzc = 1.0/x3Dc.at<float>(2);// 深度>0 逆深度>0
			if(invzc<0)
			    continue;
			float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
			float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
			if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
			    continue;// 需要在 图像尺寸内
			if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
			    continue;

// 步骤4： 在当前帧上确定候选点-----------------------------------------------------			
			// NOTE 尺度越大,图像越小
			// 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
			// 当前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
			// 因此m>=n，对应前进的情况，nCurOctave>=nLastOctave。后退的情况可以类推
			int nLastOctave = LastFrame.mvKeys[i].octave;//  上一帧  地图点 对应特征点所处的 尺度(金字塔层数)
			// Search in a window. Size depends on scale
			float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];//尺度越大，搜索范围越大
			vector<size_t> vIndices2;// 当前帧 上 投影点附近的 候选点
			if(bForward)// 前进,则上一帧兴趣点在所在的尺度nLastOctave <= nCurOctave< 8(更近了 尺度大 层数高也可以看见)
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
			else if(bBackward)// 后退,则上一帧兴趣点在所在的尺度0<= nCurOctave <= nLastOctave（远了 尺度降低）
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
			else// 没怎么运动 在上一帧 尺度附加搜索
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);
			if(vIndices2.empty())
			    continue;
			
// 步骤5：遍历候选关键点  计算与地图点  描述子匹配 计算距离 保留最近距离的匹配
			const cv::Mat dMP = pMP->GetDescriptor();// 上一帧地图点描述子
			int bestDist = 256;
			int bestIdx2 = -1;	
			// vector<size_t>::const_iterator
			for(auto vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
			{
			    const size_t i2 = *vit;
			    if(CurrentFrame.mvpMapPoints[i2])// 如果当前帧关键帧有地图点
				if(CurrentFrame.mvpMapPoints[i2]->Observations() > 0)//该对应地图点也有观测帧 则跳过
				    continue;//跳过不用在匹配地图点
				    
                           // 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
			    if(CurrentFrame.mvuRight[i2]>0)
			    {
				const float ur = u - CurrentFrame.mbf*invzc;//匹配点 右图的横坐标
				const float er = fabs(ur - CurrentFrame.mvuRight[i2]);// 误差
				if(er > radius)
				    continue;
			    }

			    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);// 当前帧 关键点描述子
			    const int dist = DescriptorDistance(dMP,d);// 描述子匹配距离
			    if(dist<bestDist)
			    {
				bestDist=dist;//最短的距离
				bestIdx2=i2;// 对应的 当前帧关键点下标
			    }
			}

			if(bestDist<=TH_HIGH)// 最短距离小于 <100
			{
			    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// 为当前帧关键点匹配上一帧的地图点
			    nmatches++;
                         // 匹配点 观测方向差 一致性检测
			    if(mbCheckOrientation)
			    {                    // 上一帧 地图点的观测方向     -    当前帧 特征点 的观测方向              
				float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle; // 匹配点 观测方向差
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(bestIdx2);//统计到对应的 方向直方图上
			    }
			}
		    }
		}
	    }
// 步骤6：根据方向差一致性约束 剔除误匹配的点
	    //Apply rotation consistency
	    if(mbCheckOrientation)
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
	// 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
	// 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
	// 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		for(int i=0; i<HISTO_LENGTH; i++)
		{
		    if(i!=ind1 && i!=ind2 && i!=ind3)
		    {
			for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
			{
			    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
			    nmatches--;
			}
		    }
		}
	    }

	    return nmatches;
	}

	// 
	// 在 关键帧地图点对应的描述子和 当前帧关键点描述子匹配后的 匹配点数少
	// 把 关键帧地图点 根据当前帧个世界的变换关系 转换到当前帧坐标系下
	// 再根据相机内参数K投影到当前帧的像素坐标系下
	// 根据像素坐标所处的 格子区域 和估算的金字塔层级信息 得到和地图点匹配的  当前帧 候选特征点
	// 计算匹配距离

	// 1. 获取pKF对应的地图点vpMPs，遍历
	//     (1). 若该点为NULL、isBad或者在SearchByBow中已经匹配上（Relocalization中首先会通过SearchByBow匹配一次），抛弃；
	// 2. 通过当前帧的位姿，将世界坐标系下的地图点坐标转换为当前帧坐标系（相机坐标系）下的坐标
	//     (2). 投影点(u,v)不在畸变矫正过的图像范围内，地图点的距离dist3D不在地图点的可观测距离内（根据地图点对应的金字塔层数，
	//           也就是提取特征的neighbourhood尺寸），抛弃
	// 3. 通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel，并获取搜索window大小（th*scale），在以上约束的范围内，
	//    搜索得到候选匹配点集合向量vIndices2
	//     const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
	// 4. 计算地图点的描述子和候选匹配点描述子距离，获得最近距离的最佳匹配，但是也要满足距离<ORBdist。
	// 5. 最后，还需要通过直方图验证描述子的方向是否匹配
/**
 * @brief 通过投影，对上一参考关键帧的特征点(地图点)进行跟踪
 * 重定位模式中的 跟踪关键帧模式    重定位中先通过 SearchByBow 在关键帧数据库中找到候选关键帧  再与每一个 参考关键帧匹配 找的匹配效果最好的 完成定位
 * 上一参考关键帧中包含了MapPoints，对这些MapPoints进行跟踪tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一参考关键帧的MapPoints投影到当前帧 
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame   当前帧
 * @param  pKF                    上一帧参考关键帧
 * @param  sAlreadyFound 当前帧关键点匹配到地图点的情况
 * @param  th                       搜索半径参数
 * @param  ORBdist             匹配距离阈值
 * @return                             成功匹配的数量
 * @see SearchByBoW()
 */		
	int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
	{
	    int nmatches = 0;
	  //  当前帧旋转平移矩阵向量 相机坐标点
	    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat Ow = -Rcw.t()*tcw;

	    // Rotation Histogram (to check rotation consistency)
	    // 匹配点对观测方向一致性检测
	    // 匹配点对观测方向差值 方向直方图
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;
	    
// 步骤1：获取关键帧pKF对应的地图点vpMPs，遍历
	    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();// 所有关键帧中的地图点
	    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)// 获取关键帧 对应的地图点vpMPs，遍历
	    {
		MapPoint* pMP = vpMPs[i];//关键帧中的地图点
		if(pMP)// 地图点存在
		{
	       // 1). 若该点为NULL、isBad或者
		  // 在SearchByBow中已经匹配上（Relocalization中首先会通过SearchByBow匹配一次），抛弃；
		    if(!pMP->isBad() && !sAlreadyFound.count(pMP))
		    {
			//Project
 // 步骤2：关键帧 对应的有效地图点投影到 当前帧 像素平面上 查看是否在视野内
			cv::Mat x3Dw = pMP->GetWorldPos();// 关键帧地图点在 世界坐标系下 的坐标
			cv::Mat x3Dc = Rcw*x3Dw+tcw;// 关键帧地图点在 当前帧坐标系（相机坐标系）下的坐标
                       // 得到归一化相机平面上的点
			const float xc = x3Dc.at<float>(0);
			const float yc = x3Dc.at<float>(1);
			const float invzc = 1.0/x3Dc.at<float>(2);//归一化
			// 有相机内参数 得到在像素平面上的投影点(u,v) 像素坐标
			const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
			const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;	
		      //  投影点(u,v)不在畸变矫正过的图像范围内 抛弃
			if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
			    continue;
			if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
			    continue;

			// Compute predicted scale level
// 步骤2： 地图点的距离dist3D不在地图点的可观测距离内（根据地图点对应的金字塔层数，
			//也就是提取特征的neighbourhood尺寸），抛弃
			cv::Mat PO = x3Dw-Ow;// 关键帧地图点到 当前帧相机中的的相对坐标
			float dist3D = cv::norm(PO);//关键帧地图点 距离当前帧相机中心的距离
			const float maxDistance = pMP->GetMaxDistanceInvariance();
			//地图点的可观测距离内（根据地图点对应的金字塔层数，也就是提取特征的neighbourhood尺寸）
			const float minDistance = pMP->GetMinDistanceInvariance();
			// Depth must be inside the scale pyramid of the image
			// 地图点的距离dist3D不在地图点的可观测距离内   
			if(dist3D < minDistance || dist3D > maxDistance)
			    continue;
			
// 步骤3：通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel，得到搜索半径，得到候选匹配点
		    // 并获取搜索window大小（th*scale），在以上约束的范围内，
		  // 搜索得到候选匹配点集合向量vIndices2
			int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);//  通过地图点的距离dist3D，预测特征对应金字塔层nPredictedLevel
			// Search in a window 并获取搜索window大小（th*scale），
			const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];
			//  在以上约束的范围内，搜索得到候选匹配点集合向量vIndices2
			// 对于 特征点格子内 图像金字塔的 相应层上 的候选特征点
			const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
			if(vIndices2.empty())
			    continue;

			const cv::Mat dMP = pMP->GetDescriptor();//关键帧地图点的描述子
			int bestDist = 256;
			int bestIdx2 = -1;
// 步骤4：计算地图点的描述子和候选匹配点描述子距离，获得最近距离的最佳匹配，但是也要满足距离<ORBdist。		
			// vector<size_t>::const_iterator
			for(auto vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)//每一个候选匹配点
			{
			    const size_t i2 = *vit;
			    if(CurrentFrame.mvpMapPoints[i2])//当前帧每一个候选匹配点 已经匹配到了 地图点 跳过
				continue;

			    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);/// 候选匹配点描述子
			    const int dist = DescriptorDistance(dMP,d);//  计算地图点的描述子和候选匹配点描述子距离

			    if(dist<bestDist)// 获得最近距离的最佳匹配，
			    {
				bestDist=dist;//最短的距离
				bestIdx2=i2;//对应的 当前帧 关键点下标
			    }
			}
			
// 步骤5：最短距离阈值检测  要满足最短距离距离<ORBdist。
			if(bestDist <= ORBdist)//但是也要满足距离<ORBdist 100
			{
			    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// 为当前帧生成和关键帧匹配上的地图点
			    nmatches++;

			    if(mbCheckOrientation)//  最后，还需要通过直方图验证描述子的方向是否匹配
			    {
				float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;//匹配点观测方向差
				//将关键帧与当前帧匹配点的观测方向angle相减
				// 得到rot（0<=rot<360），放入一个直方图中
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				// 对于每一对匹配点的角度差，均可以放入一个bin的范围内（360/HISTO_LENGTH）
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(bestIdx2);// 方向直方图
			    }
			}

		    }
		}
	    }
	    
// 步骤6：匹配点对 观测方向一致性 检测
	// 其中角度直方图是用来剔除不满足两帧之间角度旋转的外点的，也就是所谓的旋转一致性检测
	    if(mbCheckOrientation)//  最后，还需要通过直方图验证描述子的方向是否匹配
	    {
		int ind1=-1;
		int ind2=-1;
		int ind3=-1;
	// 统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
	// 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
	// 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
		ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		for(int i=0; i<HISTO_LENGTH; i++)
		{
		    if(i!=ind1 && i!=ind2 && i!=ind3)//最高的三个bin保留
		    {
			for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
			{
			    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;// 其他范围内的匹配点剔除
			    nmatches--;
			}
		    }
		}
	    }

	    return nmatches;
	}

	
/**
 * @brief  统计直方图最高的三个bin保留，其他范围内的匹配点剔除。
		 另外，若最高的比第二高的高10倍以上，则只保留最高的bin中的匹配点。
		 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
 * @param  histo  直方图
 * @param  L         直方图的大小
 * @param  ind1   数量最高的一个bin
 * @param  ind2   数量次高的一个bin
 * @param  ind3   数量第三高的一个bin
 */		
	void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
	    int max1=0;
	    int max2=0;
	    int max3=0;
// 步骤1：遍历直方图的 每一个 bin 查看其统计计数情况
	    for(int i=0; i<L; i++)
	    {
		const int s = histo[i].size();
// 步骤2：保留最高的bin的 下标		
		if(s>max1)
		{
		    max3=max2;// 第三高 计数值
		    max2=max1;// 第二高 计数值
		    max1=s;// 第一高 计数值
		    ind3=ind2;
		    ind2=ind1;
		    ind1=i;// 第一高 计数值 对应的 直方图 序列下标 0~360 >>> 0~30 
		}
// 步骤3：保留次高的bin的 下标			
		else if(s>max2)
		{
		    max3=max2;// 第三高 计数值
		    max2=s;// 第二高 计数值
		    ind3=ind2;
		    ind2=i;
		}
// 步骤4：保留第三高的bin的 下标			
		else if(s>max3)
		{
		    max3=s;// 第三高 计数值
		    ind3=i;
		}
	    }
//步骤5：若最高的比第二高的 高10倍以上，则只保留最高的bin中的匹配点
	    if(max2 < 0.1f*(float)max1)  
	    {
		ind2=-1;
		ind3=-1;
	    }
//步骤6： 若最高的比第 三高的高10倍以上，则 保留最高的和第二高bin中的匹配点。
	  else if(max3<0.1f*(float)max1)
	    {
		ind3=-1;
	    }
	}


	// Bit set count operation from
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
/**
 * @brief     二进制向量之间 相似度匹配距离
 * @param  a  二进制向量
 * @param  b  二进制向量
 * @return 
 */	
	int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
	    const int *pa = a.ptr<int32_t>();//指针
	    const int *pb = b.ptr<int32_t>();

	    int dist=0;

	    for(int i=0; i<8; i++, pa++, pb++)//只计算了前八个 二进制位 的差异
	    {
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	    }

	    return dist;
	}

} //namespace ORB_SLAM
