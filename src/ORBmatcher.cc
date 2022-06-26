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
		    if(F.mvuRight[idx]>0)// Binocular / depth camera
		    {
			const float er = fabs(pMP->mTrackProjXR  -  F.mvuRight[idx]);
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
	    
	    // Match ORB features belonging to the same node (specific layer)
	    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
	    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
	    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
	    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	    while(f1it != f1end && f2it != f2end)
	    {
		// Step 1: Take out the ORB feature points belonging to the same node respectively (only those belonging to the same node can be matching points)  
		if(f1it->first == f2it->first)
		{
		    // Step 2: Traverse the feature points belonging to the node in KF1	  
		    for(size_t i1=0, iend1=f1it->second.size(); i1 < iend1; i1++)
		    {
			const size_t idx1 = f1it->second[i1];
		        // Take out the map point MapPoint corresponding to the feature in KF1
			MapPoint* pMP1 = vpMapPoints1[idx1];
			// No matching map points to skip
			if(!pMP1)
			    continue;
			// is a bad point continue
			if(pMP1->isBad())
			    continue;
		    	// Take out the descriptor corresponding to the feature in KF1
			const cv::Mat &d1 = Descriptors1.row(idx1);
			int bestDist1=256;// Best distance (minimum distance)
			int bestIdx2 =-1 ;
			int bestDist2=256; // The penultimate best distance (the penultimate shortest distance)
			
			// Step 3: Traverse the feature points belonging to the node in KF2 and find the best matching point
			for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
			{
			    const size_t idx2 = f2it->second[i2];
			    // corresponding map point
			    MapPoint* pMP2 = vpMapPoints2[idx2];
			    if(vbMatched2[idx2] || !pMP2)
				continue;
			    if(pMP2->isBad())
				continue;
			    // Step 4: Find the distance of the descriptor and keep the matching points corresponding to the smallest and second smallest distances
			    const cv::Mat &d2 = Descriptors2.row(idx2);// Take out the descriptor corresponding to the feature in F
			    int dist = DescriptorDistance(d1,d2);
			    if(dist<bestDist1)// dist < bestDist1 < bestDist2，update bestDist1 bestDist2
			    {
				bestDist2=bestDist1;
				bestDist1=dist;// the longest distance
				bestIdx2=idx2;// Corresponding to KF2 map point subscript
			    }
			    else if(dist<bestDist2)// bestDist1 < dist < bestDist2，update bestDist2
			    {
				bestDist2=dist;// next shortest distance
			    }
			}
			
			// Step 4: Voting out false matches based on threshold and angle
			if(bestDist1<TH_LOW)
			{
			    // trick!
			    // The best match is obviously better than the next best match, then the best match is really reliable
			    if(static_cast<float>(bestDist1) < mfNNratio*static_cast<float>(bestDist2))
			    {
				vpMatches12[idx1]=vpMapPoints2[bestIdx2];// The matched map point in KF2 
				vbMatched2[bestIdx2]=true;// The map point in KF2 has been matched with a map point in KF1

				if(mbCheckOrientation)
				{
				  // trick!
				    // angle：The rotation main direction angle of each feature point when extracting the descriptor, if the image is rotated, this angle will change
				    // The angle change of all feature points should be consistent, and the most accurate angle change value is obtained through histogram statistics
				    float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;//Match point direction difference
				    if(rot<0.0)
					rot+=360.0f;
				    int bin = round(rot*factor);
				    if(bin==HISTO_LENGTH)
					bin=0;
				    assert(bin>=0 && bin<HISTO_LENGTH);
				    rotHist[bin].push_back(idx1);//Match point direction difference histogram
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
	    
	    // According to the consistency constraint of the direction difference, the incorrect matching points are eliminated
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
		    // If the rotation angle variation of the feature point belongs to these three groups, keep the matching point pair
		    if(i==ind1 || i==ind2 || i==ind3)
			continue;
		    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
		    {
		        // Remove the matching points except ind1, ind2, ind3
			vpMatches12[rotHist[i][j]] =static_cast<MapPoint*>(NULL);
			nmatches--;
		    }
		}
	    }

	    return nmatches;
	}


	  /**
	  * @brief Using the fundamental matrix F12, a 2d-2d matching point pair is generated between the two keyframes and the map points where the feature points of the two frames are not matched 
	  * Each feature point of key frame 1 and the feature point of key frame 2 belong to the same node of the dictionary (including many similar words)
	  * The feature points under the node are matched with descriptors, and the closest matching distance is selected under the condition that the epipolar geometric constraints are met
	  * After matching points, direction difference consistency constraints, detection and removal of some mismatches
	  * @param pKF1          keyframe 1
	  * @param pKF2          keyframe 2
	  * @param F12           fundamental matrix F    p2 transpose × F  × p1 = 0
	  * @param vMatchedPairs Store matching feature point pairs, the feature points are represented by their indices in the keyframe
	  * @param bOnlyStereo   In the case of binocular and rgbd, the feature points are required to have matching in the right image
	  * @return              Number of successful matches
	  */
	int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
					      vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
	{    
	    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;// Dictionary vector representation of keyframe pKF1 descriptors
	    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;// Dictionary vector representation of keyframe pKF2 descriptors

	    //Compute epipole in second image
	    // Calculate the coordinates of the camera center of KF1 in the KF2 image plane, that is, the coordinates of the pole
	    cv::Mat Cw = pKF1->GetCameraCenter();// In the KF1 O1 world coordinate system
	    cv::Mat R2w = pKF2->GetRotation();// World coordinate system -----> KF2 rotation matrix
	    cv::Mat t2w = pKF2->GetTranslation();// World coordinate system -----> KF2 translation vector
	    cv::Mat C2 = R2w*Cw+t2w;// The coordinates of KF1 O1 in the KF2 coordinate system
	    const float invz = 1.0f/C2.at<float>(2);//depth normalized coordinates
	    // KF1 O1 is projected onto the KF2 pixel coordinate system
	    const float ex =pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
	    const float ey =pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

	    // Find matches between not tracked keypoints
	    // Matching speed-up by ORB Vocabulary
	    // Compare only ORB that share the same node
	    int nmatches=0;
	    vector<bool> vbMatched2(pKF2->N,false);// Whether the pKF2 keyframe 2 map point is marked by the pKF1 map point match
	    vector<int> vMatches12(pKF1->N,-1);// Frame 1 pKF1 map points match map points in pKF2

	    // Match Points, Orientation Differences and Consistency Constraints
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;
	    
	    // Match ORB features belonging to the same node (specific layer)
	    // The data structure of FeatureVector is similar to: {(node1,feature_vector1) (node2,feature_vector2)...}
	    // f1it->first corresponds to the node number, and f1it->second corresponds to the number of all feature points belonging to the node
	    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
	    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
	    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
	    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
	    
	    // Step 1: Traverse the node nodes in the feature vector tree of the linear representation of the dictionary in pKF1 and pKF2
	    while(f1it!=f1end && f2it!=f2end)
	    {
	        // If f1it and f2it belong to the same node
	        // Take out the ORB feature points belonging to the same node respectively (only if they belong to the same node, can they be matching points)    
		if(f1it->first == f2it->first)
		{
		    // Step 2: Traverse all feature points of key frame 1 pKF1 (f1it->first) under the node node	  
		    for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
		    {
			// Get the indices of all feature points belonging to the node in pKF1
			const size_t idx1 = f1it->second[i1];
			// Step 2.1: Take out the corresponding map point MapPoint in pKF1 through the feature point index idx1      
			MapPoint* pMP1 = pKF1->GetMapPoint(idx1);    
			// If there is already a MapPoint skip
			// The feature point already exists, the map point does not need to be calculated and skipped directly
			// Since it is looking for unmatched feature points, pMP1 should be NULL
			if(pMP1)
			    continue;
			// If the value in mvuRight (right image has a matching point) is greater than 0, it means binocular/depth, and the feature point has a depth value
			const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;
			if(bOnlyStereo)
			    if(!bStereo1)
				continue;
		  	// Step 2.2: Take out the corresponding feature point in pKF1 through the feature point index idx1     
			const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
			
		  	// Step 2.3: Extract the descriptor of the corresponding feature point in pKF1 through the feature point index idx1	
			const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
			
			int bestDist = TH_LOW;//50
			int bestIdx2 = -1;//matching point subscript
			// Step 3: Traverse all feature points of key frame 2 pKF2 (f2it->first) under the node node         
			for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
			{
			    // Get the indices of all feature points belonging to the node in pKF2
			    size_t idx2 = f2it->second[i2];
			    
		      	    // Step 3.1: Take out the corresponding MapPoint in pKF2 through the feature point index idx2  
			    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);      
			    // If we have already matched or there is a MapPoint skip
			    if(vbMatched2[idx2] || pMP2)
				continue;
			    // If the value in mvuRight (right image has a matching point) is greater than 0, it means binocular/depth, and the feature point has a depth value
			    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;
			    if(bOnlyStereo)
				if(!bStereo2)
				    continue;
				
		    	    // Step 3.2: Extract the descriptor of the corresponding feature point in pKF2 through the feature point index idx2      
			    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
			    
		   	    // Calculate the descriptor distance of the corresponding feature points of idx1 and idx2 in the two key frames        
			    const int dist = DescriptorDistance(d1,d2);
			    
			    if(dist>TH_LOW || dist>bestDist)
				continue;
			    
		  	    // Step 3.3: Take out the corresponding feature point in pKF2 through the feature point index idx2
			    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

			    if(!bStereo1 && !bStereo2)
			    {
			    	// KF1 O1 is projected onto the KF2 pixel coordinate system ex ey
				const float distex = ex - kp2.pt.x;
				const float distey = ey - kp2.pt.y;
				// The feature point is too close to the pole, indicating that the MapPoint corresponding to kp2 is too close to the pKF1 camera
				if(distex*distex + distey*distey < 100 * pKF2->mvScaleFactors[kp2.octave])
				    continue;
			    }
			    
			    // Step 4: Calculate whether the distance from the feature point kp2 to the kp1 polar line (kp1 corresponds to a polar line of pKF2) is less than the threshold
			    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
			    {
				bestIdx2 = idx2;// keep matching points
				bestDist = dist;
			    }
			}
			// Steps 1, 2, 3, and 4 are summed up as follows: each feature point of the left image and all feature points of the same node of the right image
			// Check in turn to determine whether the epipolar geometric constraints are met. Satisfying the constraints is the matching feature point.
			if(bestIdx2 >= 0)// The subscript of KF1 feature point matching point in KF2, initialized to -1
			  // > 0  found a match
			{
			    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];// Pixel coordinates of matching points in KF2
			    vMatches12[idx1]=bestIdx2;
			    nmatches++;
			    // Step 5: Match Consistency Constraints for Point Orientation Differences
			    if(mbCheckOrientation)
			    {
				float rot = kp1.angle-kp2.angle;// Match point  Direction difference
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(idx1);// Match point  Direction difference  Histogram
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

	    // According to the consistency constraint of the direction difference, the incorrect matching points are eliminated
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
		    // The matching point with the highest consistency of direction difference is retained
		    if(i==ind1 || i==ind2 || i==ind3)
			continue;
		  // Other match points clear
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
		if(vMatches12[i]<0)// no match point
		    continue;
		vMatchedPairs.push_back(make_pair(i,vMatches12[i]));//Keep matching point-to-point relationships
	    }

	    return nmatches;
	}


	  /**
	  * @brief Project MapPoints into keyframe pKF and determine if there are duplicate MapPoints
	  * 1. If the MapPoint can match the feature point of the key frame, and the feature point has a corresponding MapPoint, then merge the two MapPoints (select the one with more observations)
	  * 2. If MapPoint can match the feature point of the key frame, and the feature point does not have a corresponding MapPoint, then add a map point MapPoint for the feature point
	  * @param  pKF          adjacent keyframes
	  * @param  vpMapPoints  mapPoints on the current frame that need to be fused
	  * @param  th           factor for search radius
	  * @return              number of repeating MapPoints
	  */
	int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
	{
	    // Rotation and translation matrices of keyframes    Euclidean transformation
	    cv::Mat Rcw = pKF->GetRotation();
	    cv::Mat tcw = pKF->GetTranslation();
	    // In-camera parameters
	    const float &fx = pKF->fx;
	    const float &fy = pKF->fy;
	    const float &cx = pKF->cx;
	    const float &cy = pKF->cy;
	    const float &bf = pKF->mbf;// Baseline ×f
	    // The camera coordinates of the key frame center point coordinates
	    cv::Mat Ow = pKF->GetCameraCenter();

	    int nFused=0;// The number of fused map points
	    const int nMPs = vpMapPoints.size();// The number of map points to be fused
	    // Step 1: Iterate over all MapPoints
	    for(int i=0; i<nMPs; i++)
	    {
		MapPoint* pMP = vpMapPoints[i];// map point
		
		//Step 2: Skip Bad Map Points
		if(!pMP)// does not exist
		    continue;
		// The map point is a bad point. The map point is observed by the key frame. It has been matched and does not need to be fused.
		if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
		    continue;

		// Step 3: Project map points on keyframe image pixel coordinates	
		cv::Mat p3Dw = pMP->GetWorldPos();// The coordinates of the map point in the world coordinate system
		cv::Mat p3Dc = Rcw*p3Dw + tcw;// The coordinates of the map point in the camera coordinate system under the key frame
		// Depth must be positive
		if(p3Dc.at<float>(2)<0.0f)
		    continue;
		const float invz = 1/p3Dc.at<float>(2);// depth normalization factor
		const float x = p3Dc.at<float>(0)*invz;// Coordinates at the normalized scale of the camera
		const float y = p3Dc.at<float>(1)*invz;
		// pixel coordinates
		const float u = fx*x+cx;
		const float v = fy*y+cy;
		// Point must be inside the image
		if(!pKF->IsInImage(u,v))
		    continue;
		
		// Step 4: Determine whether the distance is within the scale covariance range
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw-Ow;
		// The distance of the map point from the camera and then infer the possible scale in the image pyramid, the farther the scale is, the smaller the scale is, and the closer the scale is.
		const float dist3D = cv::norm(PO);
		// Depth must be inside the scale pyramid of the image
		if(dist3D<minDistance || dist3D>maxDistance )
		    continue;//  cull

		//Step 5:Viewing angle must be less than 60 deg
		cv::Mat Pn = pMP->GetNormal();
		if(PO.dot(Pn)<0.5*dist3D)
		    continue;
		// Predict the scale of the map point on the frame image according to the depth, the large scale of the depth is small, and the small scale of the depth is large.
		int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

		// Step 6: Determine the search radius according to the scale and then determine the candidate key points on the image
		const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
		if(vIndices.empty())
		    continue;
		
		// Step 7: Match to the most similar keypoint in the radius
		const cv::Mat dMP = pMP->GetDescriptor();// map point descriptor
		int bestDist = 256;
		int bestIdx = -1;
		// vector<size_t>::const_iterator 
		for(auto vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
		{
		    const size_t idx = *vit;
		    //  The scale of keypoints needs to be above the predicted scale
		    const cv::KeyPoint &kp = pKF->mvKeysUn[idx];// keyframe candidate keypoint
		    const int &kpLevel= kp.octave;// keypoint scale
		    if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
			continue;
		    // Step 8: Calculate the distance between the coordinates of the MapPoint projection and the feature points in this area. If the deviation is large, skip the feature point matching directly	    
		    if(pKF->mvuRight[idx]>=0)
		    {
			const float ur = u - bf*invz;
			// Check reprojection error in stereo
			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float &kpr = pKF->mvuRight[idx];
			const float ex = u - kpx;// abscissa difference
			const float ey = v - kpy;// vertical coordinate difference
			const float er = ur - kpr;// Right image matching point abscissa difference
			const float e2 = ex*ex + ey*ey + er*er;

			if(e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
			    continue;// If the difference is too large, skip it directly
		    }
	      	     //  Monocular camera No right image Match point abscissa difference
		    else
		    {
			const float &kpx = kp.pt.x;
			const float &kpy = kp.pt.y;
			const float ex = u-kpx;
			const float ey = v-kpy;
			const float e2 = ex*ex+ey*ey;
			// Threshold calculated based on chi-square test (assuming measurement has a one-pixel deviation)
			if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
			    continue;
		    }
		    // Step 9: Calculate the distance between map points and key frame feature points, and select the key point with the closest distance between the descriptors
		    const cv::Mat &dKF = pKF->mDescriptors.row(idx);// keyframe feature point descriptor
		    const int dist = DescriptorDistance(dMP,dKF);// The distance between map points and keyframe feature points, descriptors
		    if(dist<bestDist)
		    {
			bestDist = dist;// shortest distance
			bestIdx = idx;// Corresponding feature point subscript
		    }
		}

		// If there is already a MapPoint replace otherwise add new measurement      
		if(bestDist<=TH_LOW)// Distance < 50
		{
		   // The map point corresponding to the feature point on this frame, the initialization value is a null pointer
		    MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
		    // Step 10: If the MapPoint can match the feature point of the key frame, and the feature point has a corresponding MapPoint, then merge the two MapPoints (select the one with more observations)  
		    // itself has been matched to the map point
		    if(pMPinKF)
		    {
			if(!pMPinKF->isBad())// good point
			{
			    // Map points and key frames, map points corresponding to feature points, match the map points that have been observed more frequently
			    if(pMPinKF->Observations() > pMP->Observations())//Frame map points are observed more frequently, and frame map points are retained
				pMP->Replace(pMPinKF);//The original map point is replaced by the frame map point
			    else
				pMPinKF->Replace(pMP);// Replace the frame map point with the original map point
			}
		    }
		    // Step 11: If the MapPoint can match the feature point of the key frame, and the feature point does not have a corresponding MapPoint, then add a map point MapPoint for the feature point         
		   //    Keyframe Feature points Map points that have not yet been matched Match the matched map points
		    else
		    {
			pMP->AddObservation(pKF,bestIdx);// The pMP map point observes the bestIdx-th feature point on frame pKF
			pKF->AddMapPoint(pMP,bestIdx);// The bestIdx feature point of the frame corresponds to the pMP map point
		    }
		    nFused++;// Fusion times++
		}
	    }
	    return nFused;
	}


	  /**
	  * @brief Project MapPoints into keyframe pKF and determine if there are duplicate MapPoints
	  * Scw is the Sim3 similarity transformation transformation from the world coordinate system to the pKF body coordinate system. 
	  * It is necessary to convert the similarity transformation to the Euclidean transformation SE3 to transform the vpPoints in the world coordinate system to the body coordinate system.
	  * 1 The map point is matched to the frame key point. When the key point has a corresponding map point, replace the original map point with the map point corresponding to the frame key point.
	  * 2 The map point is matched to the frame key point. When the key point has no corresponding map point, add the matched map point MapPoint for the feature point
	  * @param  pKF               adjacent keyframes
	  * @param  Scw               sim3 similarity transformation transformation from world coordinate system to pKF body coordinate system [s*R t]
	  * @param  vpPoints          map Points that need to be fused
	  * @param  th                factor for search radius
	  *@param    vpReplacePoint
	  * @return                   number of repeating MapPoints
	  */
	int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
	{
	    // Get Calibration Parameters for later projection
	    // In-camera parameters
	    const float &fx = pKF->fx;
	    const float &fy = pKF->fy;
	    const float &cx = pKF->cx;
	    const float &cy = pKF->cy;

	    // Decompose Scw
	    // Similarity transformation Sim3 converted to Euclidean transformation SE3
	    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
	    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));//Similarity scale factor for rotation matrices in similarity transformations
	    cv::Mat Rcw = sRcw/scw;// Rotation matrix in Euclidean transformation
	    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;// translation vector in euclidean transformation
	    cv::Mat Ow = -Rcw.t()*tcw;//The coordinates of the camera center in the world coordinate system

	    // Set of MapPoints already found in the KeyFrame
	    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();
	    int nFused=0;// fusion count
	    const int nPoints = vpPoints.size();// The number of map points to be fused

	    // step 1: For each candidate MapPoint project and match  
	    for(int iMP=0; iMP<nPoints; iMP++)
	    {
		MapPoint* pMP = vpPoints[iMP];// Map points that need to be merged
		//Step 2: Skip Bad Map Points
		if(!pMP)// does not exist
		    continue;
		// Discard Bad MapPoints and already found
		if(pMP->isBad() || spAlreadyFound.count(pMP))
		    continue; 
		//Step 3: Project the map point to the keyframe pixel plane. Those not in the plane are not considered
		// Get 3D Coords.
		cv::Mat p3Dw = pMP->GetWorldPos();// map point world coordinate system coordinates
		// Transform into Camera Coords.
		cv::Mat p3Dc = Rcw*p3Dw+tcw;// The coordinates of the map point in the frame coordinate system
		// Depth must be positive
		if(p3Dc.at<float>(2)<0.0f)
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
	    // Step 5: Check whether the two match 
	    int nFound = 0;

	    for(int i1=0; i1<N1; i1++)
	    {
		int idx2 = vnMatch1[i1];// The key point of frame 2 (also corresponding to a map point) matched by the map point of frame 1 is subscripted

		if(idx2>=0)// The frame 2 key point subscript matched by the map point of frame 1
		{
		    int idx1 = vnMatch2[idx2];// The subscript of the frame 1 map point corresponding to the frame 2 key point
		    if(idx1==i1)// match with each other
		    {
			vpMatches12[i1] = vpMapPoints2[idx2];// Update the map point that frame 1 matches in frame 2
			nFound++;
		    }
		}
	    }

	    return nFound;
	}
	
	
	/**
	 * @brief Through projection, the feature points (map points) of the previous frame are tracked
	 * Through projection, the feature points (map points) of the previous frame are tracked
	 * The previous frame contains MapPoints, and these MapPoints are tracked, thereby increasing the MapPoints of the current frame \n
	 * 1. Project the MapPoints of the previous frame to the current frame (the Tcw of the current frame can be estimated according to the velocity model)
	 * 2. Select matching according to the descriptor distance near the projection point, and eliminate the final direction voting mechanism
	 * @param  CurrentFrame      current frame
	 * @param  LastFrame         previous frame
	 * @param  th                search radius parameter
	 * @param  bMono             is it monocular
	 * @return                   number of successful matches
	 * @see SearchByBoW()
	 */	
	int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
	{
	    int nmatches = 0;
	    // Step 1: Variable initialization
	    // Rotation Histogram (to check rotation consistency)
	   // The histogram statistics of the observed direction difference of the matching points are used to filter the best matches
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;

	    // current frame rotation translation matrix
	    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat twc = -Rcw.t()*tcw;// // twc(w)
	    // Previous Frame Rotation Translation Matrix
	    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat tlc = Rlw*twc + tlw;//Translation vector from the current frame to the previous frame
	    
	    // Decide whether to go forward or backward
	    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	     // Non-monocular case, if Z>0 and greater than the baseline, it means forward
	    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	    // Non-monocular case, if Z<0, and the absolute value is greater than the baseline, it means forward
	    
	    // Step 2: Traverse all key points in the previous frame (corresponding to map points)
	    for(int i=0; i<LastFrame.N; i++)
	    {
		MapPoint* pMP = LastFrame.mvpMapPoints[i];//previous frame map point

		if(pMP)// map point exists
		{
		    if(!LastFrame.mvbOutlier[i])// The map point is also not an outer point but an inner point compound transformation relation point
		    {
			// Project
			// Step 3: Project the map points of the previous frame to the pixel plane of the current frame
			cv::Mat x3Dw = pMP->GetWorldPos();// The map point of the previous frame (under the world coordinate system)
			cv::Mat x3Dc = Rcw*x3Dw+tcw;// The map point of the previous frame (under the current frame coordinate system)
			const float xc = x3Dc.at<float>(0);
			const float yc = x3Dc.at<float>(1);
			const float invzc = 1.0/x3Dc.at<float>(2);// depth>0 inverse depth>0
			if(invzc<0)
			    continue;
			float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
			float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;
			if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
			    continue;// needs to be within the image size
			if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
			    continue;

			// Step 4: Determine candidate points on the current frame		
			// NOTE The larger the scale, the smaller the image
			// The following can be understood as follows. For example, a dot with a certain area is a feature point at a certain scale n.
			// When moving forward, the area of the dot increases. At a certain scale m, it is a feature point. Due to the increased area, it needs to be detected at a higher scale.
			// Therefore, m>=n, corresponding to the forward situation, nCurOctave>=nLastOctave. The case of going backward can be deduced by analogy
			int nLastOctave = LastFrame.mvKeys[i].octave;//  The scale at which the feature point corresponds to the map point of the previous frame (the number of pyramid layers)
			// Search in a window. Size depends on scale
			float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];//The larger the scale, the larger the search range
			vector<size_t> vIndices2;// Candidate points near the projected point on the current frame
			if(bForward)// Moving forward, the interest point of the previous frame is at the scale nLastOctave <= nCurOctave < 8 (closer, large scale, high layer number can also be seen)
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
			else if(bBackward)// Backwards, the interest point in the previous frame is at the scale 0<= nCurOctave <= nLastOctave (far away, the scale is reduced)
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
			else// Not much motion Additional search at previous frame scale
			    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);
			if(vIndices2.empty())
			    continue;
			
			// Step 5: Traverse the candidate key points Calculate the matching with the map point descriptor Calculate the distance Keep the closest distance matching
			const cv::Mat dMP = pMP->GetDescriptor();// Last frame map point descriptor
			int bestDist = 256;
			int bestIdx2 = -1;	
			// vector<size_t>::const_iterator
			for(auto vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
			{
			    const size_t i2 = *vit;
			    if(CurrentFrame.mvpMapPoints[i2])// If the current frame keyframe has a map point
				if(CurrentFrame.mvpMapPoints[i2]->Observations() > 0)//If the corresponding map point also has an observation frame, skip it
				    continue;//Skip does not match map points
				    
                           // In the case of binocular and rgbd, it is necessary to ensure that the points on the right are also within the search radius
			    if(CurrentFrame.mvuRight[i2]>0)
			    {
				const float ur = u - CurrentFrame.mbf*invzc;//Matching point Abscissa of the right image
				const float er = fabs(ur - CurrentFrame.mvuRight[i2]);// deviation
				if(er > radius)
				    continue;
			    }

			    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);// current frame key descriptor
			    const int dist = DescriptorDistance(dMP,d);// descriptor matching distance
			    if(dist<bestDist)
			    {
				bestDist=dist;//shortest distance
				bestIdx2=i2;// Corresponding current frame key point subscript
			    }
			}

			if(bestDist<=TH_HIGH)// The shortest distance is less than <100
			{
			    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// Match the map point of the previous frame for the key point of the current frame
			    nmatches++;
                         // Matching point Observation direction difference Consistency detection
			    if(mbCheckOrientation)
			    {                    // The observation direction of the map point in the previous frame - the observation direction of the feature point in the current frame             
				float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle; // Matching point Observation direction difference
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(bestIdx2);//Statistics on the corresponding direction histogram
			    }
			}
		    }
		}
	    }
	     // Step 6: Eliminate the incorrectly matched points according to the direction difference consistency constraint
	    //Apply rotation consistency
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

	/**
	 * @brief Through projection, the feature points (map points) of the previous reference key frame are tracked
	 * Tracking Keyframe Mode in Retarget Mode    In relocation, first find candidate keyframes in the keyframe database through SearchByBow, and then match each reference keyframe to find the best matching effect to complete the positioning
	 * The previous reference key frame contains MapPoints, and these MapPoints are tracked, thereby increasing the MapPoints of the current frame \n
	 * 1. Project the MapPoints of the previous reference keyframe to the current frame
	 * 2. Select matching according to the descriptor distance near the projection point, and eliminate the final direction voting mechanism
	 * @param  CurrentFrame           current frame
	 * @param  pKF                    previous frame reference keyframe
	 * @param  sAlreadyFound          the current frame key point is matched to the map point
	 * @param  th                     search radius parameter
	 * @param  ORBdist                match distance threshold
	 * @return                        number of successful matches
	 * @see SearchByBoW()
	 */		
	int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
	{
	    int nmatches = 0;
	    // current frame rotation translation matrix vector
	    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
	    const cv::Mat Ow = -Rcw.t()*tcw;

	    // Rotation Histogram (to check rotation consistency)
	    // Matching point pair observation direction consistency detection
	    // Match point pair observation direction difference direction histogram
	    vector<int> rotHist[HISTO_LENGTH];
	    for(int i=0;i<HISTO_LENGTH;i++)
		rotHist[i].reserve(500);
	    const float factor = 1.0f/HISTO_LENGTH;
	    
	    // Step 1: Obtain the map point vpMPs corresponding to the key frame pKF, and foreach
	    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();// map points in all keyframes
	    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)// Get the map point vpMPs corresponding to the key frame, foreach
	    {
		MapPoint* pMP = vpMPs[i];//map points in keyframes
		if(pMP)// map point exists
		{
	       	  // 1). If the point is NULL, isBad or
		  // It has been matched in SearchByBow (relocalization will first match once through SearchByBow), discard
		    if(!pMP->isBad() && !sAlreadyFound.count(pMP))
		    {
			//Project
 			// Step 2: Project the corresponding valid map point of the key frame to the pixel plane of the current frame to check whether it is within the field of view
			cv::Mat x3Dw = pMP->GetWorldPos();// The coordinates of the keyframe map point in the world coordinate system
			cv::Mat x3Dc = Rcw*x3Dw+tcw;// The coordinates of the keyframe map point in the current frame coordinate system (camera coordinate system)
                        // get points on normalized camera plane
			const float xc = x3Dc.at<float>(0);
			const float yc = x3Dc.at<float>(1);
			const float invzc = 1.0/x3Dc.at<float>(2);//Normalized
			// There are camera internal parameters to get the pixel coordinates of the projected point (u, v) on the pixel plane
			const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
			const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;	
		      	//  Projection points (u, v) are not discarded within the distortion-corrected image
			if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
			    continue;
			if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
			    continue;

			// Compute predicted scale level
			// Step 2: The distance dist3D of the map point is not within the observable distance of the map point (according to the number of pyramid layers corresponding to the map point, 
			// that is the neighborhood size of the extracted feature)
			cv::Mat PO = x3Dw-Ow;
			float dist3D = cv::norm(PO);
			const float maxDistance = pMP->GetMaxDistanceInvariance();
			//Within the observable distance of the map point (according to the number of pyramid layers corresponding to the map point, that is, the neighborhood size of the extracted feature)
			const float minDistance = pMP->GetMinDistanceInvariance();
			// Depth must be inside the scale pyramid of the image
			// The distance of the map point dist3D is not within the observable distance of the map point   
			if(dist3D < minDistance || dist3D > maxDistance)
			    continue;
			
			// Step 3: Through the distance dist3D of the map points, the predicted feature corresponds to the pyramid layer nPredictedLevel, the search radius is obtained, and the candidate matching point is obtained
		    	// And get the search window size (th*scale), within the bounds of the above constraints
		  	// Search to get the candidate matching point set vector vIndices2
			int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);//  Through the distance dist3D of map points, the predicted feature corresponds to the pyramid level nPredictedLevel
			// Search in a window and get the search window size (th*scale)
			const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];
			//  Within the scope of the above constraints, the search obtains the candidate matching point set vector vIndices2
			//For the feature point grid, the candidate feature points on the corresponding layers of the image pyramid
			const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
			if(vIndices2.empty())
			    continue;

			const cv::Mat dMP = pMP->GetDescriptor();//Descriptors for keyframe map points
			int bestDist = 256;
			int bestIdx2 = -1;
			// Step 4: Calculate the distance between the descriptor of the map point and the descriptor of the candidate matching point to obtain the best match with the closest distance, but also satisfy the distance < ORBdist		
			// vector<size_t>::const_iterator
			for(auto vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)//each candidate matching point
			{
			    const size_t i2 = *vit;
			    if(CurrentFrame.mvpMapPoints[i2])//Each candidate matching point in the current frame has been matched to a map point and skipped
				continue;

			    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);/// candidate matching point descriptor
			    const int dist = DescriptorDistance(dMP,d);//  Calculate the distance between the descriptor of the map point and the descriptor of the candidate matching point

			    if(dist<bestDist)// Get the best match with the closest distance
			    {
				bestDist=dist;//shortest distance
				bestIdx2=i2;//The corresponding subscript of the current frame key point
			    }
			}
			
			// Step 5: The shortest distance threshold detection should satisfy the shortest distance distance < ORBdist
			if(bestDist <= ORBdist)//But also satisfy distance < ORBdist 100
			{
			    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;// Generate map points on the current frame and match the keyframes
			    nmatches++;

			    if(mbCheckOrientation)//  Finally, it is also necessary to verify that the orientations of the descriptors match through the histogram
			    {
				float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;//Match point observation direction difference
				// Subtract the angle of the observation direction of the keyframe and the matching point of the current frame
				// Get rot (0<=rot<360) and put it into a histogram
				if(rot<0.0)
				    rot+=360.0f;
				int bin = round(rot*factor);
				// The angle difference of each pair of matching points can be put into the range of a bin (360/HISTO_LENGTH)
				if(bin==HISTO_LENGTH)
				    bin=0;
				assert(bin>=0 && bin<HISTO_LENGTH);
				rotHist[bin].push_back(bestIdx2);// Orientation histogram
			    }
			}

		    }
		}
	    }
	    
	    // Step 6: Matching point pair observation direction consistency detection
	    // The angle histogram is used to remove outliers that do not satisfy the angle rotation between two frames, which is the so-called rotation consistency detection
	    if(mbCheckOrientation)//  Finally, it is also necessary to verify that the orientations of the descriptors match through the histogram
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
	 * @brief  The three highest bins in the statistical histogram are retained, and matching points in other ranges are eliminated.
			 In addition, if the highest one is more than 10 times higher than the second highest, only the matching points in the highest bin are kept.
			 If the highest is more than 10 times higher than the third, then keep the matching points in the highest and second highest bins.
	 * @param  histo    Histogram
	 * @param  L        the size of the histogram
	 * @param  ind1     The highest number of bins
	 * @param  ind2     A bin with the second highest number
	 * @param  ind3     The third highest number of bins
	 */		
	void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
	    int max1=0;
	    int max2=0;
	    int max3=0;
	    // Step 1: Traverse each bin of the histogram to view its statistical count
	    for(int i=0; i<L; i++)
	    {
		const int s = histo[i].size();
		// Step 2: Keep the index of the highest bin		
		if(s>max1)
		{
		    max3=max2;// Third highest count value
		    max2=max1;// Second highest count value
		    max1=s;// first high count value
		    ind3=ind2;
		    ind2=ind1;
		    ind1=i;// The histogram sequence index corresponding to the first high count value is 0~360 >>> 0~30
		}
		// Step 3: Keep the subscript of the next highest bin			
		else if(s>max2)
		{
		    max3=max2;
		    max2=s;
		    ind3=ind2;
		    ind2=i;
		}
		// Step 4: Keep the index of the third highest bin			
		else if(s>max3)
		{
		    max3=s;
		    ind3=i;
		}
	    }
	    //Step 5: If the highest is more than 10 times higher than the second highest, then only keep the matching points in the highest bin
	    if(max2 < 0.1f*(float)max1)  
	    {
		ind2=-1;
		ind3=-1;
	    }
	  //Step 6: If the highest bin is more than 10 times higher than the third highest bin, keep the matching points in the highest and second highest bins.
	  else if(max3<0.1f*(float)max1)
	    {
		ind3=-1;
	    }
	}


	// Bit set count operation from
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	/**
	 * @brief     Similarity matching distance between binary vectors
	 * @param  a  binary vector
	 * @param  b  binary vector
	 * @return 
	 */	
	int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
	    const int *pa = a.ptr<int32_t>();//pointer
	    const int *pb = b.ptr<int32_t>();

	    int dist=0;

	    for(int i=0; i<8; i++, pa++, pb++)//Only the difference of the first eight binary bits is calculated
	    {
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	    }

	    return dist;
	}

} //namespace ORB_SLAM
