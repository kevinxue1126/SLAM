/*
* This file is part of ORB-SLAM2
* 
* Monocular camera initialization
* Homography matrix H (8 motion hypotheses) for planar scenes and fundamental matrix F (4 motion hypotheses) for non-planar scenes, 
* then a scoring rule is used to select the appropriate model and restore the camera's rotation matrix R and translation vector t and corresponding 3D point (scale problem).
* 
* 
* 
 *【0】2D-2D point pairs Standardize before transforming matrix De-mean and then divide by absolute moments
* Monocular initialization feature points
* mean_x  =  sum( ui) / N   mean_y =  sum(vi)/N
* absolute moment  mean_x_dev = sum（abs(ui - mean_x)）/ N     mean_y_dev = sum（abs(vi - mean_y)）/ N 
*
* Inverse absolute moment  sX = 1/mean_x_dev     sY = 1/mean_y_dev 
* 
* Normalized point coordinates
* u = (ui - mean_x) × sX
* v =  (vi - mean_y) * sY 
* 
* normalized matrix
*     Used to calculate the transformation matrix score by calculating the symmetric transformation error from the original coordinates after computing the transformation matrix
* T = sX   0    -mean_x * sX
*        0   sY   -mean_y * sY
*        0    0         1
* 
* Normalized matrix * point coordinates = normalized coordinates
*         ui          ui × sX - mean_x * sX  = (ui - mean_x) × sX       u
*  T ×  vi    =    vi  × sY - mean_y * sY  = (vi - mean_y) × sY   =   v
*         1               1            1
* 
* point coordinates = normalized matrix inverse matrix * normalized coordinates
* 
* 
* 
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{
	  /**
	 * @brief class constructor  Given a reference frame, construct an Initializer monocular camera to initialize the reference frame
	 * 
	 * use reference  frame to initialize, this reference frame is the first frame of the official start of SLAM
	 * @param ReferenceFrame   reference frame
	 * @param sigma            measurement error
	 * @param iterations       RANSAC iterations
	 */
	Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
	{
	    mK = ReferenceFrame.mK.clone();

	    mvKeys1 = ReferenceFrame.mvKeysUn;

	    mSigma = sigma;
	    mSigma2 = sigma*sigma;
	    mMaxIterations = iterations;
	}
	
	/**
	 * @brief class initialization function
	 * Calculate the fundamental matrix and the homography matrix in parallel, select one of the models, and recover the relative pose and point cloud between the first two frames
	 * @param CurrentFrame      The current frame and the first frame reference frame match the triangular transformation to get the 3D point
	 * @param vMatches12        matching information of feature points of the current frame
	 * @param R21               rotation matrix 
	 * @param t21               translation matrix
	 * @param vP3D              the recovered 3D points
	 * @param vbTriangulated    3D point conforming to triangulation
	 */
	bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
				    vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
	{
	    // Fill structures with current keypoints and matches with reference frame
	    // Reference Frame: 1,  Current Frame: 2
	    mvKeys2 = CurrentFrame.mvKeysUn;

	    mvMatches12.clear();
	    mvMatches12.reserve(mvKeys2.size());
	    mvbMatched1.resize(mvKeys1.size());
	    
	    // Step 1: Organize Feature Point Pairs
	    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
	    {
		if(vMatches12[i]>=0)
		{
		    mvMatches12.push_back(make_pair(i,vMatches12[i]));
		    mvbMatched1[i]=true;
		}
		else
		    mvbMatched1[i]=false;
	    }
	    
            // The number of matching feature points
	    const int N = mvMatches12.size();
	    
            // Create a new container vAllIndices and generate a number from 0 to N-1 as the index of the feature point
	    vector<size_t> vAllIndices;
	    vAllIndices.reserve(N);
	    vector<size_t> vAvailableIndices;

	    for(int i=0; i<N; i++)
	    {
		vAllIndices.push_back(i);
	    }

	    // Generate sets of 8 points for each RANSAC iteration
	    // Step 2: Randomly select 8 pairs of matching feature points from all matching feature point pairs as a group, and select mMaxIterations group in total
	    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

	    DUtils::Random::SeedRandOnce(0);

	    for(int it=0; it<mMaxIterations; it++)
	    {
		vAvailableIndices = vAllIndices;
		// Select a minimum set
		for(size_t j=0; j<8; j++)
		{ 
		    // Generate random numbers from 0 to N-1
		    int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
		    // idx indicates which index corresponding feature point is selected
		    int idx = vAvailableIndices[randi];
		    mvSets[it][j] = idx;
		    
		    // The index corresponding to randi has been selected and deleted from the container
		    vAvailableIndices[randi] = vAvailableIndices.back();
		    vAvailableIndices.pop_back();
		}
	    }
	    
	    // Step 3: Call multi-threads to calculate fundamental matrix and homography respectively
	    // Launch threads to compute in parallel a fundamental matrix and a homography
	    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
	    float SH, SF;
	    cv::Mat H, F;
	    // Compute and score the homography matrix
	    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
	    // Calculate and score the fundamental matrix
	    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

	    // Wait until both threads have finished
	    threadH.join();
	    threadF.join();
	    // Step 4: Calculate the score ratio and select a model    
	    // Compute ratio of scores
	    float RH = SH/(SH+SF);// calculate
	    
	    // Step 5: Recover R,t from the homography H or the fundamental matrix F
	    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
	    if(RH>0.40)
		return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
	    else //if(pF_HF>0.6) // Bias to non-planarity using fundamental matrix recovery
		return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

	    return false;
	}

	/**
	 * @brief Calculate the homography matrix   
	 * 
	 * @param vbMatchesInliers     		Returns matching points that conform to the transformation
	 * @param score                         Transform Score
	 * @param H21                           Homography matrix
	 */
	void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
	{
	    // Number of putative matches
	    const int N = mvMatches12.size();// 2 matched point pairs in 1 Total number of matched point pairs

	    // Step 1： Normalize mvKeys1 and mvKey2 to a mean value of 0, the first-order absolute moment is 1, and the normalized matrices are T1 and T2, respectively	    
	    vector<cv::Point2f> vPn1, vPn2;
	    cv::Mat T1, T2;
	    Normalize(mvKeys1,vPn1, T1);
	    Normalize(mvKeys2,vPn2, T2);
	    cv::Mat T2inv = T2.inv();

	    // Best Results variables
	    score = 0.0;
	    vbMatchesInliers = vector<bool>(N,false);

	    // Iteration variables
	    vector<cv::Point2f> vPn1i(8);
	    vector<cv::Point2f> vPn2i(8);
	    cv::Mat H21i, H12i;
	    vector<bool> vbCurrentInliers(N,false);
	    float currentScore;
	    
	     // Step 2: Iterative solution of random sampling sequence
	    // Perform all RANSAC iterations and save the solution with highest score
	    for(int it=0; it<mMaxIterations; it++)//within the maximum number of iterations
	    {
		// Select a minimum set
		//step 3: random 8 pairs of point pairs
		for(size_t j=0; j<8; j++)
		{
		    int idx = mvSets[it][j];
		    vPn1i[j] = vPn1[mvMatches12[idx].first];
		    vPn2i[j] = vPn2[mvMatches12[idx].second];
		}                                    
		// Step 4: Calculate the homography matrix        T1*p1  ----> T2*p2     p1 ----------------> p2
		cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
		H21i = T2inv*Hn*T1;
		H12i = H21i.inv();
		
		// Step 5: Calculate the score of the homography H
		currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
		// Step 6: Keep the homography corresponding to the highest score
		if(currentScore > score)
		{
		    H21 = H21i.clone();
		    vbMatchesInliers = vbCurrentInliers;   
		    score = currentScore;
		}
	    }
	}


	/**
	 * @brief Calculate the fundamental matrix
	 *
	 * Assuming that the scene is non-planar, the Fundamental matrix (current frame 2 to reference frame 1) is obtained through the first two frames, and the score of the model is obtained
	 */
	void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
	{
	    // Number of putative matches
	    const int N = vbMatchesInliers.size();
		
	    /*
 	     * [1] 2D-2D point pair Standardize before calculating the transformation matrix
	     */
	    // Normalize coordinates
	    vector<cv::Point2f> vPn1, vPn2;
	    cv::Mat T1, T2;
	    Normalize(mvKeys1,vPn1, T1);
	    Normalize(mvKeys2,vPn2, T2);
	    cv::Mat T2t = T2.t();

	    // Best Results variables
	    score = 0.0;
	    vbMatchesInliers = vector<bool>(N,false);

	    // Iteration variables
	    vector<cv::Point2f> vPn1i(8);
	    vector<cv::Point2f> vPn2i(8);
	    cv::Mat F21i;
	    vector<bool> vbCurrentInliers(N,false);
	    float currentScore;
	    // [2] Iterative solution of random sampling sequence
	    // Perform all RANSAC iterations and save the solution with highest score
	    for(int it=0; it<mMaxIterations; it++)
	    {
        	//【3】Random 8 pairs of point pairs     
		for(int j=0; j<8; j++)
		{
		    int idx = mvSets[it][j];
		    vPn1i[j] = vPn1[mvMatches12[idx].first];
		    vPn2i[j] = vPn2[mvMatches12[idx].second];
		}
                
    		// [4] Calculate the fundamental matrix T1*p1 ----> T2*p2 p1 ----------------> p2
		cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
		F21i = T2t*Fn*T1;
             
     		// [5] The score for calculating the fundamental matrix F is obtained from the symmetric transformation error of the corresponding matching point pair.	
		currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
     		// [6] Retain the basic matrix F corresponding to the highest score
		if(currentScore>score)
		{
		    F21 = F21i.clone();// The optimal fundamental matrix F
		    vbMatchesInliers = vbCurrentInliers;//Keeping the optimal point pairs solved at each iteration are marked 1 for interior points 0 for outliers
		    score = currentScore;//current score
		}
	    }
	}

	/**
	 * @brief Finding homography (normalized DLT) from feature point matching
	 * 
	 * @param  vP1 normalized point, in reference frame
	 * @param  vP2 normalized point, in current frame
	 * @return     Homography matrix
	 * @see        Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
	 */
	cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
	{
	    const int N = vP1.size();
	    cv::Mat A(2*N,9,CV_32F);
	    for(int i=0; i<N; i++)
	    {
		const float u1 = vP1[i].x;
		const float v1 = vP1[i].y;
		const float u2 = vP2[i].x;
		const float v2 = vP2[i].y;
		
		A.at<float>(2*i,0) = 0.0;
		A.at<float>(2*i,1) = 0.0;
		A.at<float>(2*i,2) = 0.0;
		A.at<float>(2*i,3) = -u1;
		A.at<float>(2*i,4) = -v1;
		A.at<float>(2*i,5) = -1;
		A.at<float>(2*i,6) = v2*u1;
		A.at<float>(2*i,7) = v2*v1;
		A.at<float>(2*i,8) = v2;
		
		A.at<float>(2*i+1,0) = u1;
		A.at<float>(2*i+1,1) = v1;
		A.at<float>(2*i+1,2) = 1;
		A.at<float>(2*i+1,3) = 0.0;
		A.at<float>(2*i+1,4) = 0.0;
		A.at<float>(2*i+1,5) = 0.0;
		A.at<float>(2*i+1,6) = -u2*u1;
		A.at<float>(2*i+1,7) = -u2*v1;
		A.at<float>(2*i+1,8) = -u2;
	    }
	    cv::Mat u,w,vt;

	    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	    return vt.row(8).reshape(0, 3);
	}


	/**
	 * @brief Find fundamental matrix from feature point matching (normalized 8-point method)
	 * @param  vP1 normalized point, in reference frame
	 * @param  vP1 normalized point, in reference frame
	 * @return  Fundamental matrix
	 * @see    Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 
	 */
	cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
	{
	    const int N = vP1.size();

	    cv::Mat A(N,9,CV_32F);

	    for(int i=0; i<N; i++)
	    {
		const float u1 = vP1[i].x;
		const float v1 = vP1[i].y;
		const float u2 = vP2[i].x;
		const float v2 = vP2[i].y;

		A.at<float>(i,0) = u2*u1;
		A.at<float>(i,1) = u2*v1;
		A.at<float>(i,2) = u2;
		A.at<float>(i,3) = v2*u1;
		A.at<float>(i,4) = v2*v1;
		A.at<float>(i,5) = v2;
		A.at<float>(i,6) = u1;
		A.at<float>(i,7) = v1;
		A.at<float>(i,8) = 1;
	    }

	    cv::Mat u,w,vt;

	    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	    cv::Mat Fpre = vt.row(8).reshape(0, 3);// The rank of the fundamental matrix of F is 2. It is necessary to take the diagonal matrix after the decomposition and the rank is 2. In the synthesis of F

	    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	    w.at<float>(2)=0;//  The rank of the fundamental matrix is 2, an important constraint

	    return  u * cv::Mat::diag(w)  * vt;// in synthetic F
	}

	/**
	 * @brief Score the given homography matrix
	 * 
	 * @see
	 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
	 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
	 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
	 */
	float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
	{   
	    const int N = mvMatches12.size();// total number of matching point pairs
	    
	    // |h11 h12 h13|
	    // |h21 h22 h23|
	    // |h31 h32 h33|
	    const float h11 = H21.at<float>(0,0);  //  p1  ----> p2
	    const float h12 = H21.at<float>(0,1);
	    const float h13 = H21.at<float>(0,2);
	    const float h21 = H21.at<float>(1,0);
	    const float h22 = H21.at<float>(1,1);
	    const float h23 = H21.at<float>(1,2);
	    const float h31 = H21.at<float>(2,0);
	    const float h32 = H21.at<float>(2,1);
	    const float h33 = H21.at<float>(2,2);
	    

	    // |h11inv h12inv h13inv|
	    // |h21inv h22inv h23inv|
	    // |h31inv h32inv h33inv|
	    const float h11inv = H12.at<float>(0,0);
	    const float h12inv = H12.at<float>(0,1);
	    const float h13inv = H12.at<float>(0,2);
	    const float h21inv = H12.at<float>(1,0);
	    const float h22inv = H12.at<float>(1,1);
	    const float h23inv = H12.at<float>(1,2);
	    const float h31inv = H12.at<float>(2,0);
	    const float h32inv = H12.at<float>(2,1);
	    const float h33inv = H12.at<float>(2,2);

	    vbMatchesInliers.resize(N);// Whether the matching point pair is an interior point on the transformation of the transformation matrix pair

	    float score = 0;
            // Threshold calculated based on the chi-square test (assuming the measurement has a one-pixel deviation)
	    const float th = 5.991;
	    //Information matrix, inverse of squared variance
	    const float invSigmaSquare = 1.0/(sigma*sigma);//Variance Reciprocal
	    
           // N对特征匹配点
	    for(int i=0; i<N; i++)/ / Calculate the symmetric transformation error generated when the homography matrix transforms each point pair
	    {
		bool bIn = true;
                // key point coordinates
		const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
		const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

		// Step 1: Convert p2 from homography to p1 distance error and score
		const float u1 = kp1.pt.x;
		const float v1 = kp1.pt.y;
		const float u2 = kp2.pt.x;
		const float v2 = kp2.pt.y;
		// Reprojection error in first image
		// x2in1 = H12*x2
		// Homography of the feature points in image 2 to image 1
		// |u1|    |h11inv h12inv h13inv||u2|
		// |v1| = |h21inv h22inv h23inv||v2|
		// |1 |     |h31inv h32inv h33inv||1 |
		const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);// third line countdown
		const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;// p2 is converted from homography to p1'
		const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
		const float squareDist1 = (u1-u2in1)*(u1-u2in1) + (v1-v2in1)*(v1-v2in1);// sum of squares of the difference between the horizontal and vertical coordinates
		const float chiSquare1 = squareDist1*invSigmaSquare;// normalize the error according to the variance
		if(chiSquare1>th)//The distance is greater than the threshold, the effect of changing the point is poor
		    bIn = false;
		else
		    score += th - chiSquare1;// Threshold - the distance difference gets a score, the smaller the difference, the higher the score
		    
		// Step 2: Convert p1 from homography to p2 distance error and score
		// Reprojection error in second image
		const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
		const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
		const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
		
		const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
		
		const float chiSquare2 = squareDist2*invSigmaSquare;
		if(chiSquare2>th)
		    bIn = false;
		else
		    score += th - chiSquare2;
		if(bIn)
		    vbMatchesInliers[i]=true;
		else
		    vbMatchesInliers[i]=false;
	    }

	    return score;
	}

	/**
	 * @brief Score a given fundamental matrix
	 * p2 transpose * F21 * p1 = 0
	 * F21*p1 is frame 1
	 * 
	 * @see
	 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
	 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
	 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC
	 */
	float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
	{
	    const int N = mvMatches12.size();

	    const float f11 = F21.at<float>(0,0);
	    const float f12 = F21.at<float>(0,1);
	    const float f13 = F21.at<float>(0,2);
	    const float f21 = F21.at<float>(1,0);
	    const float f22 = F21.at<float>(1,1);
	    const float f23 = F21.at<float>(1,2);
	    const float f31 = F21.at<float>(2,0);
	    const float f32 = F21.at<float>(2,1);
	    const float f33 = F21.at<float>(2,2);

	    vbMatchesInliers.resize(N);

	    float score = 0;
	    
            // Threshold calculated based on the chi-square test (assuming the measurement has a one-pixel deviation)
	    const float th = 3.841;
	    const float thScore = 5.991;
            //Information matrix, inverse of squared variance
	    const float invSigmaSquare = 1.0/(sigma*sigma);

	    for(int i=0; i<N; i++)
	    {
		bool bIn = true;

		const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
		const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

		const float u1 = kp1.pt.x;
		const float v1 = kp1.pt.y;
		const float u2 = kp2.pt.x;
		const float v2 = kp2.pt.y;
		// Reprojection error in second image
		// l2=F21 x1=(a2,b2,c2)
		const float a2 = f11*u1+f12*v1+f13;
		const float b2 = f21*u1+f22*v1+f23;
		const float c2 = f31*u1+f32*v1+f33;
		
                // x2 should be on the line l: x2 point multiplied by l = 0
		const float num2 = a2*u2+b2*v2+c2;
		const float squareDist1 = num2*num2/(a2*a2+b2*b2);
		const float chiSquare1 = squareDist1*invSigmaSquare;
		if(chiSquare1>th)
		    bIn = false;
		else
		    score += thScore - chiSquare1;

		// Reprojection error in second image
		const float a1 = f11*u2+f21*v2+f31;
		const float b1 = f12*u2+f22*v2+f32;
		const float c1 = f13*u2+f23*v2+f33;
		const float num1 = a1*u1+b1*v1+c1;
		const float squareDist2 = num1*num1/(a1*a1+b1*b1);
		const float chiSquare2 = squareDist2*invSigmaSquare;
		if(chiSquare2>th)
		    bIn = false;
		else
		    score += thScore - chiSquare2;

		if(bIn)
		    vbMatchesInliers[i]=true;
		else
		    vbMatchesInliers[i]=false;
	    }

	    return score;
	}


	/**
	 * @brief Restoring R t from the fundamental matrix F
	 * 
	 * Metric Refactoring
	 * 1. Combine the Fundamental matrix with the camera internal parameter K to get the Essential matrix: \f$ E = k transpose F k \f$
	 * 2. SVD decomposition to get R t
	 * 3. Carry out the cheerality check and find the most suitable solution from the four solutions
	 * 
	 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
	 */	
	bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
				    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
	    int N=0;
	    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
		if(vbMatchesInliers[i])
		    N++;

	    // Compute Essential Matrix from Fundamental Matrix
 	    // Step 1: Calculate the essential matrix E = K transpose * F * K
	    cv::Mat E21 = K.t()*F21*K;

	    cv::Mat R1, R2, t;
	    // Recover the 4 motion hypotheses  
	    // Although this function is normalized to t, it does not determine the scale of the monocular entire SLAM process
	    DecomposeE(E21,R1,R2,t);  

	    cv::Mat t1=t;
	    cv::Mat t2=-t;

	    // Step 3: Reconstruct with the 4 hyphoteses and check
	    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
	    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
	    float parallax1,parallax2, parallax3, parallax4;

	    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
	    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
	    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
	    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

	    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

	    R21 = cv::Mat();
	    t21 = cv::Mat();
            // minTriangulated is the number of three-dimensional points that can be triangulated to restore
	    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

	    int nsimilar = 0;
	    if(nGood1>0.7*maxGood)
		nsimilar++;
	    if(nGood2>0.7*maxGood)
		nsimilar++;
	    if(nGood3>0.7*maxGood)
		nsimilar++;
	    if(nGood4>0.7*maxGood)
		nsimilar++;

	    // If there is not a clear winner or not enough triangulated points reject initialization
	    if(maxGood<nMinGood || nsimilar>1)
	    {
		return false;
	    }

	    // If best reconstruction has enough parallax initialize
	    if(maxGood==nGood1)
	    {
		if(parallax1>minParallax)
		{
		    vP3D = vP3D1;
		    vbTriangulated = vbTriangulated1;

		    R1.copyTo(R21);
		    t1.copyTo(t21);
		    return true;
		}
	    }else if(maxGood==nGood2)
	    {
		if(parallax2>minParallax)
		{
		    vP3D = vP3D2;
		    vbTriangulated = vbTriangulated2;

		    R2.copyTo(R21);
		    t1.copyTo(t21);
		    return true;
		}
	    }else if(maxGood==nGood3)
	    {
		if(parallax3>minParallax)
		{
		    vP3D = vP3D3;
		    vbTriangulated = vbTriangulated3;

		    R1.copyTo(R21);
		    t2.copyTo(t21);
		    return true;
		}
	    }else if(maxGood==nGood4)
	    {
		if(parallax4>minParallax)
		{
		    vP3D = vP3D4;
		    vbTriangulated = vbTriangulated4;

		    R2.copyTo(R21);
		    t2.copyTo(t21);
		    return true;
		}
	    }

	    return false;
	}

	
	/**
	 * @brief Restoring R t from H
	 *
	 * @see
	 * - Faugeras et al, Motion and structure from motion in a piecewise planar environment. International Journal of Pattern Recognition and Artificial Intelligence, 1988.
	 * - Deeper understanding of the homography decomposition for vision-based control
	 */
	bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
			      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
	{
	    int N=0;
	    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
		if(vbMatchesInliers[i])
		    N++;

	    // We recover 8 motion hypotheses using the method of Faugeras et al.
	    // Motion and structure from motion in a piecewise planar environment.
	    // International Journal of Pattern Recognition and Artificial Intelligence, 1988
	    
           // Because the feature point is the image coordinate system, the H matrix is converted from the camera coordinate system to the image coordinate system
	    cv::Mat invK = K.inv();
	    cv::Mat A = invK*H21*K;

	    cv::Mat U,w,Vt,V;
	    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
	    V=Vt.t();

	    float s = cv::determinant(U)*cv::determinant(Vt);

	    float d1 = w.at<float>(0);
	    float d2 = w.at<float>(1);
	    float d3 = w.at<float>(2);
	    
            // The normal case for SVD decomposition is that the eigenvalues are arranged in descending order
	    if(d1/d2<1.00001 || d2/d3<1.00001)
	    {
		return false;
	    }

	    vector<cv::Mat> vR, vt, vn;
	    vR.reserve(8);
	    vt.reserve(8);
	    vn.reserve(8);

	    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
	    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
	    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
	    float x1[] = {aux1,aux1,-aux1,-aux1};
	    float x3[] = {aux3,-aux3,aux3,-aux3};

	    //case d'=d2
	    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

	    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
	    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
	    //Calculate the rotation matrix R', calculate formula 18 in ppt
	    for(int i=0; i<4; i++)
	    {
		cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
		Rp.at<float>(0,0)=ctheta;
		Rp.at<float>(0,2)=-stheta[i];
		Rp.at<float>(2,0)=stheta[i];
		Rp.at<float>(2,2)=ctheta;

		cv::Mat R = s*U*Rp*Vt;
		vR.push_back(R);

		cv::Mat tp(3,1,CV_32F);
		tp.at<float>(0)=x1[i];
		tp.at<float>(1)=0;
		tp.at<float>(2)=-x3[i];
		tp*=d1-d3;
		
		// Although t is normalized here, it does not determine the scale of the entire SLAM process of the monocular
		cv::Mat t = U*tp;
		vt.push_back(t/cv::norm(t));

		cv::Mat np(3,1,CV_32F);
		np.at<float>(0)=x1[i];
		np.at<float>(1)=0;
		np.at<float>(2)=x3[i];

		cv::Mat n = V*np;
		if(n.at<float>(2)<0)
		    n=-n;
		vn.push_back(n);
	    }

	    //case d'=-d2
	    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

	    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
	    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};
	    
              // Calculate the rotation matrix R', calculate formula 21 in ppt
	    for(int i=0; i<4; i++)
	    {
		cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
		Rp.at<float>(0,0)=cphi;
		Rp.at<float>(0,2)=sphi[i];
		Rp.at<float>(1,1)=-1;
		Rp.at<float>(2,0)=sphi[i];
		Rp.at<float>(2,2)=-cphi;

		cv::Mat R = s*U*Rp*Vt;
		vR.push_back(R);

		cv::Mat tp(3,1,CV_32F);
		tp.at<float>(0)=x1[i];
		tp.at<float>(1)=0;
		tp.at<float>(2)=x3[i];
		tp*=d1+d3;

		cv::Mat t = U*tp;
		vt.push_back(t/cv::norm(t));

		cv::Mat np(3,1,CV_32F);
		np.at<float>(0)=x1[i];
		np.at<float>(1)=0;
		np.at<float>(2)=x3[i];

		cv::Mat n = V*np;
		if(n.at<float>(2)<0)
		    n=-n;
		vn.push_back(n);
	    }


	    int bestGood = 0;
	    int secondBestGood = 0;    
	    int bestSolutionIdx = -1;
	    float bestParallax = -1;
	    vector<cv::Point3f> bestP3D;
	    vector<bool> bestTriangulated;

	    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
	    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
	    for(size_t i=0; i<8; i++)
	    {
		float parallaxi;
		vector<cv::Point3f> vP3Di;
		vector<bool> vbTriangulatedi;
		int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
                // Keep the best and the next best
		if(nGood>bestGood)
		{
		    secondBestGood = bestGood;
		    bestGood = nGood;
		    bestSolutionIdx = i;
		    bestParallax = parallaxi;
		    bestP3D = vP3Di;
		    bestTriangulated = vbTriangulatedi;
		}
		else if(nGood>secondBestGood)
		{
		    secondBestGood = nGood;
		}
	    }


	    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
	    {
		vR[bestSolutionIdx].copyTo(R21);
		vt[bestSolutionIdx].copyTo(t21);
		vP3D = bestP3D;
		vbTriangulated = bestTriangulated;

		return true;
	    }

	    return false;
	}


	/**
	 * @brief Given the projection matrices P1, P2 and the points kp1, kp2 on the image, recover the 3D coordinates
	 *
	 * @param kp1 feature point, in reference frame
	 * @param kp2 feature points, in current frame
	 * @param P1 projection matrix P1
	 * @param P2 projection matrix P2
	 * @param x3D 3D point
	 * @see       Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
	 */
	void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
	{
	    // Normalize t in DecomposeE function and ReconstructH function
	    cv::Mat A(4,4,CV_32F);

	    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
	    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
	    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
	    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

	    cv::Mat u,w,vt;
	    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
	    x3D = vt.row(3).t();
	    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
	}

	
	/**
	 * ＠brief Normalize feature points to the same scale (as input to normalize DLT)
	 *
	 * [x' y' 1]' = T * [x y 1]' \n
	 * After normalization, the mean of x', y' is 0, sum(abs(x_i'-0))=1, sum(abs((y_i'-0))=1
	 * 
	 * @param vKeys             The coordinates of the feature points on the image
	 * @param vNormalizedPoints Normalized coordinates of feature points
	 * @param T                 matrix to normalize feature points
	 */
	void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
	{

	    const int N = vKeys.size();
	    vNormalizedPoints.resize(N);
	    
	    float meanX = 0;
	    float meanY = 0;
	    for(int i=0; i<N; i++)
	    {
		meanX += vKeys[i].pt.x;
		meanY += vKeys[i].pt.y;
	    }
	    meanX = meanX/N;
	    meanY = meanY/N;

	    float meanDevX = 0;
	    float meanDevY = 0;
	    
           // Subtract the center coordinates from all vKeys points so that the mean x and y coordinates are 0
	    for(int i=0; i<N; i++)
	    {
		vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
		vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
		meanDevX += fabs(vNormalizedPoints[i].x);
		meanDevY += fabs(vNormalizedPoints[i].y);
	    }
	    meanDevX = meanDevX/N;//mean absolute moment
	    meanDevY = meanDevY/N;

	    float sX = 1.0/meanDevX;
	    float sY = 1.0/meanDevY;
	    
           // Scale the x-coordinate and y-coordinate respectively, so that the first-order absolute moments of the x-coordinate and y-coordinate are 1, respectively
	    for(int i=0; i<N; i++)
	    {
		vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
		vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
	    }
		// |sX  0  -meanx*sX|
		// |0   sY -meany*sY|
		// |0   0          1         |	    
	    T = cv::Mat::eye(3,3,CV_32F);
	    T.at<float>(0,0) = sX;
	    T.at<float>(1,1) = sY;
	    T.at<float>(0,2) = -meanX*sX;
	    T.at<float>(1,2) = -meanY*sY;
	}



	/**
	 * @brief Perform cheerality check to further find the most suitable solution after F decomposition
	 * Check whether the obtained R t complies with
	 * Accepts R,t , a set of successful matches. 
	 * The final result is how many matches in this set of matches can be correctly triangulated under this set of R,t (that is, Z is greater than 0), 
	 * and output these triangulated 3D points.
	 */
	int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
			      const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
			      const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
	{
	    // Calibration parameters
	    const float fx = K.at<float>(0,0);
	    const float fy = K.at<float>(1,1);
	    const float cx = K.at<float>(0,2);
	    const float cy = K.at<float>(1,2);

	    vbGood = vector<bool>(vKeys1.size(),false);
	    vP3D.resize(vKeys1.size());// Corresponding 3D point

	    vector<float> vCosParallax;
	    vCosParallax.reserve(vKeys1.size());

	    // Camera 1 Projection Matrix K[I|0]
	   //Step 1: Get a camera's projection matrix
	    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
	    K.copyTo(P1.rowRange(0,3).colRange(0,3));
            // The coordinates of the optical center of the first camera in the world coordinate system
	    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);
	    
	    // Step 2: Get the projection matrix of the second camera
	    cv::Mat P2(3,4,CV_32F);
	    R.copyTo(P2.rowRange(0,3).colRange(0,3));
	    t.copyTo(P2.rowRange(0,3).col(3));
	    P2 = K*P2;
            //Camera 2 origin R inverse * - t R is an orthogonal matrix inverse = transpose
	    cv::Mat O2 = -R.t()*t;

	    int nGood=0;

	    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
	    {
		if(!vbMatchesInliers[i])
		    continue;
               // kp1 and kp2 are matching feature points
		const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
		const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
		cv::Mat p3dC1;
		
		// Step 3: Use trigonometry to restore the 3D point p3dC1
		// kp1 = P1 * p3dC1     kp2 = P2 * p3dC1   
		Triangulate(kp1,kp2,P1,P2,p3dC1);

		if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
		{// The obtained 3d point coordinates are valid
		    vbGood[vMatches12[i].first]=false;
		    continue;
		}
		
		// Step 4: Calculate the cosine of the parallax angle
		// Check parallax
		cv::Mat normal1 = p3dC1 - O1;
		float dist1 = cv::norm(normal1);

		cv::Mat normal2 = p3dC1 - O2;
		float dist2 = cv::norm(normal2);

		float cosParallax = normal1.dot(normal2)/(dist1*dist2);
		
		 // Step 5: Determine if the 3D point is in front of the two cameras
		// Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
	   	// Step 5.1: 3D point depth is negative, behind the first camera, out
		if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
		    continue;
		
           	// Step 5.2: 3D point depth is negative, behind the second camera, out
		// Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
		cv::Mat p3dC2 = R*p3dC1+t;

		if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
		    continue;
		
		// Step 6: Calculate the reprojection error
		// Check reprojection error in first image
		float im1x, im1y;
		float invZ1 = 1.0/p3dC1.at<float>(2);
		im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
		im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
		float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
		
         	// Step 6.1: Reprojection error is too large, skip elimination
		if(squareError1>th2)
		    continue;
		
		// Check reprojection error in second image
		float im2x, im2y;
		float invZ2 = 1.0/p3dC2.at<float>(2);
		im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
		im2y = fy*p3dC2.at<float>(1)*invZ2+cy;
		float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        	 // Step 6.2: Reprojection error is too large, skip elimination
		if(squareError2>th2)
		    continue;
		
         	// Step 7: Count the number of tested 3D points and record the 3D point parallax angle
		vCosParallax.push_back(cosParallax);
		vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
		//nGood++;

		if(cosParallax<0.99998){
		    vbGood[vMatches12[i].first]=true;
		  nGood++;
		 }
	    }
	    
	    // Step 8: Get Larger Parallax Angles in 3D Points
	    if(nGood>0)
	    {
		sort(vCosParallax.begin(),vCosParallax.end());
		
		size_t idx = min(50,int(vCosParallax.size()-1));
		parallax = acos(vCosParallax[idx])*180/CV_PI;
	    }
	    else
		parallax=0;

	    return nGood;
	}

	
/*
 * Recover rotation matrix R and translation vector t from essential matrix
 *  Singular value decomposition of the essential matrix E gives possible solutions
 * t = u * RZ(90) * u transpose
 * R= u * RZ(90) * v transpose
 * There are four combinations
 */

/**
 * @brief Decompose Essential Matrix
 * 
 * The Essential matrix can be obtained by combining the internal parameters of the F matrix, and 4 sets of solutions will be obtained by decomposing the E matrix
 * The four sets of solutions are [R1,t],[R1,-t],[R2,t],[R2,-t]
 * @param E  Essential Matrix
 * @param R1 Rotation Matrix 1
 * @param R2 Rotation Matrix 2
 * @param t  Translation
 * @see Multiple View Geometry in Computer Vision - Result 9.19 p259
 */
	void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
	{
            // [1] Perform singular value decomposition on the essential matrix E
	    cv::Mat u,w,vt;
	    cv::SVD::compute(E,w,u,vt);// where u and v represent two mutually orthogonal matrices, and w represents a diagonal matrix
	    
	    // There is normalization to t, but this place does not determine the scale of the entire SLAM process of the monocular
	    u.col(2).copyTo(t);
	    t=t/cv::norm(t);
	    // The rotation matrix obtained by rotating 90 degrees along the Z axis (counterclockwise is the positive direction)
            // The z-axis is still the original z-axis   
	    // The y-axis becomes the negative of the original x-axis   
	    // The x-axis becomes the original y-axis 
	    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
	    W.at<float>(0,1)=-1;
	    W.at<float>(1,0)=1;
	    W.at<float>(2,2)=1;

	    R1 = u*W*vt;
	    if(cv::determinant(R1)<0)
		R1=-R1;

	    R2 = u*W.t()*vt;
	    if(cv::determinant(R2)<0)
		R2=-R2;
	}

} //namespace ORB_SLAM
