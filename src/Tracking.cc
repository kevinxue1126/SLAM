/* 
*This file is part of ORB-SLAM2.
* Tracking Thread Depth Binocular Initialization Pose Motion Model Keyframe Mode Relocalization Local Map Tracking Keyframe
* mpMap is our entire pose and map (which can be imagined as the interface world of ORB-SLAM runtime), 
* and MapPoint and KeyFrame are included in this mpMap. 
* Therefore, when creating these three objects (map, map point, keyframe), the relationship between the three cannot be missing in the constructor.
* 
* 
* In addition, since the feature points extracted from a key frame correspond to a map point set, 
* it is necessary to note down the number of each map point in the frame;
* 
* In the same way, a map point will be observed by multiple keyframes, 
* and it also needs a few times the number of each keyframe in the point.
* 
* For map points, two operations need to be completed. 
* The first is to select the descriptor with the highest degree of 
* discrimination among the multiple feature points (corresponding to multiple key frames) of the observed map point as the descriptor of the map point.
* pNewMP->ComputeDistinctiveDescriptors();
* 
* The second is to update the range of the average observation direction and observation distance of the map point. 
* These are all preparations for the subsequent fusion of the descriptors.
* pNewMP->UpdateNormalAndDepth();
*
* 
* track
* Each frame of image Frame ---> Extract ORB key point features -----> Calculate R t according to the position estimation of the previous frame (or initialize the position by global relocation)
* ------> Track the local map and optimize the pose -------> Whether to add keyframes
* 
* Tracking thread
* Frame
* 1】initialization
*       Monocular initialization MonocularInitialization()
*       Binocular initialization StereoInitialization
* 
* 2】Camera pose tracking P
*       Simultaneous tracking and positioning Simultaneous tracking and positioning, no keyframe insertion, local mapping does not work
*       Tracking and positioning separation mbOnlyTracking(false)  
*        Pose tracking TrackWithMotionModel()  TrackReferenceKeyFrame()  reset Relocalization()
*   
* a Tracking with motion model (Tracking with motion model) has a faster tracking rate. 
*      Assuming that the object is in uniform motion, the pose and speed of the previous frame are used to estimate the pose of the current frame. 
*      The function used is TrackWithMotionModel().
*      Here the matching is done by projecting to match the map points seen in the previous frame, using
*      matcher.SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, ...)。
*      
* b keyframe mode     TrackReferenceKeyFrame()
*     When the number of feature points matched by the motion mode is small, the key frame mode is selected. That is, try to match with the most recent keyframe.
*     For fast matching, this paper utilizes bag of words (BoW) to speed up.
*     First, calculate the BoW of the current frame, and set the initial pose as the pose of the previous frame;
*     Second, find feature matching according to the pose and BoW dictionary, use the function matcher.SearchByBoW(KeyFrame *pKF, Frame &F, ...);
*     What is matched is the map point in the reference keyframe.
*     Finally, the pose is optimized using the matched features.
*     
* c Initialize pose estimation via global relocalization Relocalization()
*    If the above method is used, the matching of the current frame and the nearest neighbor keyframe also fails, 
*    which means that relocation is required to continue tracking.
*    The relocation entry is as follows: bOK = Relocalization();
*    At this point, it is only necessary to match all keyframes to see if a suitable position can be found.
*    First, the BOW vector of the current frame is calculated, and several key frames are selected as candidates in the key frame dictionary database.
*         The function used is as follows: vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
*    Second, look for keyframes with enough feature points to match; finally, use RANSAC to iterate, and then use the PnP algorithm to solve the pose. This part is also in Tracking::Relocalization()
*
*    
* 3】local map tracking
*       Update local map UpdateLocalMap() Update keyframes and update map points  UpdateLocalKeyFrames()   UpdateLocalPoints
*       Search for map points  Get a match between the local map and the current frame
*       Optimize the pose    Minimize reprojection error  3D point-2D point pair  si * pi = K * T * Pi = K * exp(f) * Pi 
* 
* 4】whether to generate keyframes
*       Conditions to join:
*       No keyframes inserted for a long time
*       Partial map is idle
*       Tracking is about to lose
*       The proportion of MapPoints map points of the tracking map is relatively small
* 
* 5】Generate keyframes
*       KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB)
*       Construct some MapPoints for binocular or RGBD cameras, add properties to MapPoints
* 
* Enter the LocalMapping thread
* 
* 
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>// orb feature detection extraction

// user
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"// 3d-2d point pair Solving for R t

#include<iostream>
#include<mutex>//Multithreading


using namespace std;

// If the first letter of the variable name in the program is "m", it means a member variable in the class，member
// First and second letters:
// "p" indicates the pointer data type
// "n" means int type
// "b" means bool type
// "s" means set type
// "v" indicates the vector data type
// "l" means list data type
// "KF" for KeyPoint data type

namespace ORB_SLAM2
{
    /**
      * @brief  Tracking object initialization function  default constructor
      *
      */
	// Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
        Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer,
                           MapDrawer *pMapDrawer, Map *pMap,
                           shared_ptr<PointCloudMapping> pPointCloud,
                           KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
	    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), 
            mbVO(false), mpORBVocabulary(pVoc),
            mpPointCloudMapping( pPointCloud ),
	    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), 
            mpSystem(pSys), mpViewer(NULL),
	    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), 
            mpMap(pMap), mnLastRelocFrameId(0)
	{
	    // Load camera parameters from settings file

	    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);//read configuration file
	    
	     //【1】------------------ In-camera parameter matrix K------------------------
	    //     |fx  0   cx|
	    // K = |0   fy  cy|
	    //     |0   0   1 |
	    float fx = fSettings["Camera.fx"];
	    float fy = fSettings["Camera.fy"];
	    float cx = fSettings["Camera.cx"];
	    float cy = fSettings["Camera.cy"];
	    cv::Mat K = cv::Mat::eye(3,3,CV_32F);// initialized to a diagonal matrix
	    K.at<float>(0,0) = fx;
	    K.at<float>(1,1) = fy;
	    K.at<float>(0,2) = cx;
	    K.at<float>(1,2) = cy;
	    K.copyTo(mK);// Copy to In-class variable mK is an accessible variable in the class
	    
 	    // 【2】-------Distortion Correction Parameters----------------------------------------
	    cv::Mat DistCoef(4,1,CV_32F);// Camera Distortion Correction Matrix
	    DistCoef.at<float>(0) = fSettings["Camera.k1"];
	    DistCoef.at<float>(1) = fSettings["Camera.k2"];
	    DistCoef.at<float>(2) = fSettings["Camera.p1"];
	    DistCoef.at<float>(3) = fSettings["Camera.p2"];
	    const float k3 = fSettings["Camera.k3"];
	    if(k3!=0)
	    {
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	    }
	    DistCoef.copyTo(mDistCoef);// copy to class variable

	    mbf = fSettings["Camera.bf"];// baseline * fx 
            //----------------Shooting frame rate---------------------------
	    float fps = fSettings["Camera.fps"];
	    if(fps==0)
		fps=30;

	    // Max/Min Frames to insert keyframes and to check relocalisation
	    // key frame interval
	    mMinFrames = 0;
	    mMaxFrames = fps;
	    // 【3】------------------Display parameters--------------------------
	    cout << endl << "Camera Parameters: " << endl;
	    cout << "-- fx: " << fx << endl;
	    cout << "-- fy: " << fy << endl;
	    cout << "-- cx: " << cx << endl;
	    cout << "-- cy: " << cy << endl;
	    cout << "-- k1: " << DistCoef.at<float>(0) << endl;
	    cout << "-- k2: " << DistCoef.at<float>(1) << endl;
	    if(DistCoef.rows==5)
		cout << "-- k3: " << DistCoef.at<float>(4) << endl;
	    cout << "-- p1: " << DistCoef.at<float>(2) << endl;
	    cout << "-- p2: " << DistCoef.at<float>(3) << endl;
	    cout << "-- fps: " << fps << endl;
	    int nRGB = fSettings["Camera.RGB"];// Image channel order 1 RGB order 0 BGR order
	    mbRGB = nRGB;
	    if(mbRGB)
		cout << "-- color order: RGB (ignored if grayscale)" << endl;
	    else
		cout << "-- color order: BGR (ignored if grayscale)" << endl;

	     //【4】-----------Load ORB parameters------------------------------------
	    // The number of feature points extracted per frame is 1000
	    int nFeatures = fSettings["ORBextractor.nFeatures"];          //The total number of feature points extracted from each image is 2000
	    // Scale of image changes when building pyramids 1.2
	    float fScaleFactor = fSettings["ORBextractor.scaleFactor"]; // Scale factor 1.2 Image pyramid Scale factor
	    // Levels of the scale pyramid 8
	    int nLevels = fSettings["ORBextractor.nLevels"];
	    // The default threshold for extracting fast feature points is 20
	    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
	     // If the default threshold does not extract enough fast feature points, use the minimum threshold of 8
	    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
           
	    // 【5】-------------------Create ORB Feature Extraction Object---------------------------------------------------------
           // The tracking process will use mpORBextractorLeft as the feature point extractor
	    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
	    // If it is binocular, mpORBextractorRight will also be used as the right-eye feature point extractor during the tracking process.
	    if(sensor==System::STEREO)//binocular camera
		mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
	    // When the monocular is initialized, mpIniORBextractor is used as the feature point extractor, and the number of extracted feature points is set to be twice that of the normal frame.
	    if(sensor==System::MONOCULAR)// Monocular Camera First Frame Feature Extractor
                // In order for the monocular to be successfully initialized (monocular initialization requires a scale factor normalized by translational motion)
	       // During initialization, the number of feature points extracted by mpIniORBextractor is set to be twice that of a normal frame.
		mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
	     // 【6】--------------Display feature extraction parameter information------------------------------------
	    cout << endl  << "ORB Extractor Parameters: " << endl;
	    cout << "-- Number of Features:   " << nFeatures << endl;
	    cout << "-- Scale Levels:                                 " << nLevels << endl;
	    cout << "-- Scale Factor:                                 " << fScaleFactor << endl;
	    cout << "-- Initial Fast Threshold: " << fIniThFAST << endl;
	    cout << "-- Minimum Fast Threshold:             " << fMinThFAST << endl;

 	    //【7】 binocular or depth camera depth threshold
	    if(sensor==System::STEREO || sensor==System::RGBD)
	    {
	       // Threshold for judging how far/near a 3D point is mbf * 35 / fx
		mThDepth = mbf*(float)fSettings["ThDepth"]/fx;//depth threshold
		cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
	    }	    
	    // depth camera
	    if(sensor==System::RGBD)
	    {
	        // Depth camera disparity disparity is converted to depth, the factor when depth
		mDepthMapFactor = fSettings["DepthMapFactor"];//map depth factor
		if(fabs(mDepthMapFactor)<1e-5)
		    mDepthMapFactor=1;
		else
		    mDepthMapFactor = 1.0f/mDepthMapFactor;
	    }

	}
       // Set up local mapping
	void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
	{
	    mpLocalMapper=pLocalMapper;// set class object value
	}
       // Set up loopback detection
	void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
	{
	    mpLoopClosing=pLoopClosing;// set class object value
	}
      // Set up visualization
	void Tracking::SetViewer(Viewer *pViewer)
	{
	    mpViewer=pViewer;// set class object value
	}
	
	
	/**
	  * @brief  Binocular camera initialization Get camera pose
	  * Input left and right eye images, which can be RGB, BGR, RGBA, GRAY
	  * 1. Convert the image to mImGray and imGrayRight and initialize mCurrentFrame
	  * 2. Carry out the tracking process
	  * Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
	  */
	cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
	{
	    //-----[1] Whether the picture is RGB, BGR, or RGBA, BGRA, it is converted into a grayscale image, giving up color information.----------------------------------	 
	    mImGray = imRectLeft;
	    cv::Mat imGrayRight = imRectRight;   
            // Convert color map to gray map
 	    // Step 1: Convert RGB or RGBA image to grayscale
	    if(mImGray.channels()==3)
	    {
		if(mbRGB)//  Original image Channel RGB order
		{
		    cvtColor(mImGray,mImGray,CV_RGB2GRAY);
		    cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
		}
		else//  Original image Channel BGR sequence
		{
		    cvtColor(mImGray,mImGray,CV_BGR2GRAY);
		    cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
		}
	    }
      	    // Color image with four channels of transparency converted to grayscale
	    else if(mImGray.channels()==4)
	    {
		if(mbRGB)
		{
		    cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
		    cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
		}
		else
		{
		    cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
		    cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
		}
	    }
	    // Step 2: Construct the Frame	    
            //---Create frame, grayscale left image, grayscale right image, timestamp, ORB feature extractor for left and right images, ORB dictionary, in-camera parameter mk, distortion correction parameter mDistCoef, near and far threshold and depth scale
            // Frame object key points, key points are matched to corresponding depth values, and the coordinate values of matching points are divided into key points
	    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

	    // Step 3: Track
	    // Pose can be obtained after tracking 
	    // initialization-------------------------------
	    // The number of feature points in the current frame is greater than 500, and initialization is performed
	    // Set the first frame as a key frame and the pose as [I 0] 
	    // Calculate the 3D point based on the depth obtained from the parallax of the first frame
	    // Generate map, add map point, map point observation frame, best descriptor of map point, update direction and distance of map point
	    // map points for keyframes, add map points to the current frame, add map points to the map
	    // show map
	    // later frame -------------------
	    Track(); 

	    return mCurrentFrame.mTcw.clone();
	}
	
	/**
	  * @brief   depth camera  get camera pose
	  * Input left eye RGB or RGBA image and depth map
	  * 1. Convert the image to mImGray and imDepth, and initialize mCurrentFrame
	  * 2. Carry out the tracking process
	  * Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
	  */
	cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
	{
	  
	    // --------------[1] Whether the picture is RGB, BGR, or RGBA, BGRA, it is converted into a grayscale image, giving up color information---------------
	    mImGray = imRGB; // Grayscale
            mImRGB  = imRGB; // original color map
	    // cv::Mat imDepth = imD;// depth map local variables
            mImDepth = imD; // Initialized as a class member variable
     	    // Convert color map to gray map
	    if(mImGray.channels()==3)
	    {
		if(mbRGB)
		    cvtColor(mImGray,mImGray,CV_RGB2GRAY);
		else
                   {
		     cvtColor(mImGray,mImGray,CV_BGR2GRAY);
                     cvtColor(mImRGB,mImRGB,CV_BGR2RGB);
                   }
	    }
            // Color image with transparency, four channels converted to grayscale
	    else if(mImGray.channels()==4)
	    {
		if(mbRGB)
                    {
		       cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
                       cvtColor(mImRGB,mImRGB,CV_RGBA2RGB);
                    }
		else
                    {
		       cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
                       cvtColor(mImRGB,mImRGB,CV_BGRA2RGB);
                    }
	    }
	    // -------------【2】depth information---------------------------------
            // depth camera depth map, precision conversion
	    if((fabs(mDepthMapFactor-1.0f)>1e-5) || mImDepth.type()!=CV_32F)
		mImDepth.convertTo(mImDepth,CV_32F,mDepthMapFactor);// 32-bit float precision is calculated by dividing by 1000
	    
	    //--------------[3] The key points of the frame object, the key points match the corresponding depth values, and the matching point coordinate values are divided into key points-----------------------
	    mCurrentFrame = Frame(mImGray,mImDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
	    // -------------【4】track-------------------
	    Track();// Pose can be obtained after tracking
	    // -------------【5】Back to camera motion
	    return mCurrentFrame.mTcw.clone();
	}
	
	/**
	  * @brief Monocular camera  Get camera pose
	  * Input left eye RGB or RGBA image
	  * 1. Convert the image to mImGray and initialize mCurrentFrame
	  * 2. Carry out the tracking process
	  * Output the transformation matrix from the world coordinate system to the camera coordinate system of the frame
	  */
	cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
	{
	    mImGray = im;// image
	   //--------------【1】Whether the picture is RGB, BGR, or RGBA, BGRA, it is converted into a grayscale image, giving up color information.--------------------	    
            // Convert color map to gray map
	    if(mImGray.channels()==3)
	    {
		if(mbRGB)
		    cvtColor(mImGray,mImGray,CV_RGB2GRAY);
		else
		    cvtColor(mImGray,mImGray,CV_BGR2GRAY);
	    }
	    // Color image with transparency, four-channel conversion to grayscale
	    else if(mImGray.channels()==4)
	    {
		if(mbRGB)
		    cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
		else
		    cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
	    }
	    // --------------------【2】Then encapsulate the current read frame as an mCurrentFrame object of type Frame----------------------------
	    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
	      // monocular, first frame, extractormpIniORBextractor
		mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
	    else
	      // post frame extractor mpORBextractorLeft
		mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
	      //  【3】 track
	   // Motion tracking (tracking the previous frame map point) / reference frame tracking (tracking the previous reference keyframe map point) / relocation (tracking all keyframe map points) to get the pose
	   // Local map point tracking and re-optimized pose
	    Track();// Pose can be obtained after tracking

	    return mCurrentFrame.mTcw.clone();
	}
/*
// Track key points to calculate camera pose
Tracking this part mainly uses several models:
sports model track（Tracking with motion model）、
reference keyframe track（Tracking with reference key frame）和
reset（Relocalization） track。

【1】sports model track（Tracking with motion model）
        Track the map point of the previous frame
        
        The map points of the previous frame are back-projected to the pixel coordinates of the current frame image and the key points of the current frame, 
	which fall in the same grid for descriptor matching and searching, which can speed up the matching. 
        
	Assuming that the object is moving at a uniform speed, the pose and velocity of the previous frame can be used to estimate the pose of the current frame.
	The velocity of the previous frame can be calculated from the pose of the previous frames.
	This model is suitable for situations where the movement speed and direction are relatively consistent and there is no large rotation, such as cars, robots, and people moving at a uniform speed.
	And for targets with more casual movements, of course, it will fail. At this point, the following two models are used.

【2】reference keyframe track（Tracking with reference key frame）
	If the motion model has failed, then you can first try to match with the latest keyframe (match the map point in the keyframe).
	After all, the distance between the current frame and the previous keyframe is not very far.
	The author utilizes bag of words (BoW) to speed up matching.
	
	Both the key frame and the current frame are represented by a dictionary word linear vector
        The descriptors of the corresponding words must be relatively similar, and matching the descriptors of the corresponding words can speed up the matching.
        
	First, calculate the BoW of the current frame, and set the initial pose as the pose of the previous frame;
	Second, find feature matching according to the pose and BoW dictionary (see ORB-SLAM (6) loop closure detection);
	Finally, the pose is optimized using the matched features (see ORB-SLAM (5) optimization).

【3】reset（Relocalization） track
       The current frame is calculated with a dictionary, the dictionary word linear representation vector
       All keyframes are computed using a dictionary, a dictionary word linear representation vector
       
       Calculate the dictionary word linearity of the current frame, which represents the dictionary word linearity between the vector and all key frames, and represents the distance between the vectors. Select some candidate key frames with short distances
       The current frame and the candidate key frame, respectively, perform descriptor matching
       
       	Both the key frame and the current frame are represented by a dictionary word linear vector
        The descriptors of the corresponding words must be relatively similar. Matching the descriptors of the corresponding words can speed up the matching.
       
      If the matching between the current frame and the nearest neighbor keyframe also fails, it means that the current frame has been lost at this time, and its true position cannot be determined.
      At this point, it is only necessary to match all keyframes to see if a suitable position can be found. First, compute the Bow vector of the current frame.
      Second, use the BoW dictionary to select several key frames as candidates (see ORB-SLAM (6) loop closure detection);
      Third, look for key frames with enough feature point matching; finally, use feature point matching to iteratively solve the pose (under the RANSAC framework, 
      because the relative pose may be relatively large, there will be more outliers).
      If a keyframe has enough interior points, select the pose optimized by that keyframe.

　1）Prefer to pass constant velocity motion model, from LastFrame (last normal frame)
　      Directly predict (multiply by a fixed pose transformation matrix) the pose of the current frame;
    2）If it is a static state or the motion model matching fails 
           (after using the constant speed model, it is found that the map points of the LastFrame and the feature points of the CurrentFrame rarely match), 
	   by increasing the matching range of the map point backprojection of the reference frame, after obtaining more matches , calculate the current pose;
    3）If both of these fail, it means that the tracking fails, mState!=OK, 
          then use Bow in KeyFrameDataBase to search for the feature point matching of CurrentFrame, perform global relocation GlobalRelocalization, 
	  and use EPnP to solve the current pose under the RANSAC framework.
	  
      Once we have obtained the initial camera pose and initial feature matches through the above three models, 
      we can project the complete map into the current frame to search for more matches. However, 
      projecting a complete map is computationally expensive and unnecessary in large scale scenarios. 
      Therefore, a local map LocalMap is used here for projection matching.
      
LocalMap contains:
    The key frame K1 connected to the current frame, and the key frame K2 connected to K1 (first-level and second-level connected key frames);
    Map points corresponding to K1 and K2; refer to the key frame Kf.
    
The matching process is as follows:
        for local map points
　　1. Abandon the projection range beyond the camera screen;
　　2. Discard the difference between the observation angle and the average observation direction of the map points by more than 60o;
　　3. Discard the mismatch between the scale of feature points and the scale of map points (represented by the number of Gaussian pyramid layers);
　　4. Calculate the scale of the feature points in the current frame;
　　5. To match the descriptor of the map point with the descriptor of the ORB feature of the current frame, it is necessary to search around the rough x projection position obtained by the initial pose according to the scale of the map point;
　　6. Perform PoseOptimization optimization based on all matching points.
　　
These three tracking models are all to obtain a rough initial value of the camera pose, 
and then BundleAdjustment will be performed on the pose by tracking the local map TrackLocalMap to further optimize the pose.

*/
/**
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * Tracking thread
 */
	void Tracking::Track()
	{
	    // track contains two parts: estimated motion (motion transformation matrix of two frames before and after), tracking local map (positioning in the map)
     	    // mState is the state machine for tracking
           // SYSTME_NOT_READY , NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
           // NO_IMAGE_YET state if the image has been reset or run for the first time
	    if(mState == NO_IMAGES_YET)
	    {
		mState = NOT_INITIALIZED;// Uninitialized
	    }
	    
            // mLastProcessedState stores the latest state of Tracking for drawing in FrameDrawer
	    mLastProcessedState = mState;

	    // lock the map Get Map Mutex -> Map cannot be changed
	    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	    // Step 1: Tracking of the previous frame, the system is not initialized, initialize it, and get the initialization pose (tracking and estimated motion)
	    if(mState == NOT_INITIALIZED)
	    {
                // 1. Monocular / binocular / RGBD initialization to get the 3d point seen in the first frame
		if(mSensor==System::STEREO || mSensor==System::RGBD)
		    // The number of feature points in the current frame is greater than 500 for initialization
		    // Set the first frame as a key frame and pose as [I 0]
		    // Calculate the 3D point based on the depth obtained from the disparity of the first frame
		    // Generate map Add map point Map point observation frame Map point best descriptor Update map point direction and distance
		    // Map points for keyframes Add map points to the current frame Add map points to the map
		    // show map 
		    StereoInitialization();// binocular / deep initialization
		else
		      // The number of feature points in two consecutive frames is greater than 100 and the number of pairs of key points orb feature matching points in two frames is greater than 100
		      // Initial frame [I 0] Second frame Fundamental matrix/homography recovery [R t] Global optimization and corresponding 3D points
		      // Create Map Map Optimization Using Minimize Reprojection Error BA Optimize Pose and Map Points
		      // Reciprocal of the median depth distance Normalized the translation vector of the second frame pose and the three-axis coordinates of the map point
		      // show update
		    MonocularInitialization();// Monocular initialization

            // 2. Visually display the current frame pose
	    	mpFrameDrawer->Update(this);// display frame
		if(mState!=OK)
		    return;
	    }  
	    
// Step 2: Tracking of subsequent frames
      // 1. Track the previous frame to get an initial estimate of the pose.
	 // The system has been initialized (there are already 3d points in the map) Track the previous frame Feature point pair Calculate camera movement Pose----
     // 2. Track local maps, map optimization to fine-tune the pose
     // 3. Processing after tracking failure (two-by-two tracking failure or partial map tracking failure)
	    else
	    {
		bool bOK; // bOK is a temporary variable used to indicate whether each function is executed successfully
		// Initial camera pose estimation using motion model or relocalization (if tracking is lost)
		// There is a switch in the viewer menuLocalizationMode , which controls whether to ActivateLocalizationMode , and finally controls mbOnlyTracking
		// mbOnlyTracking equal to false means normal VO mode (with map update), mbOnlyTracking equal to true means the user manually selects the positioning mode
     // 1. Track the previous frame=============================================================================================================
           // 1. Tracking + Mapping + Relocation==================================================================================		
		if(!mbOnlyTracking)// Tracking + Mapping + Relocating (relocating after tracking is lost)
		{
		    // Local Mapping is activated. This is the normal behaviour, unless
		    // you explicitly activate the "only tracking" mode.
                // A. Normal initialization is successful==============================================================================
		    if(mState==OK)//Status ok not lost
		    {
			// Check and update MapPoints that were replaced in the previous frame
			// Updated MapPoints replaced by Fuse function and SearchAndFuse function      
			CheckReplacedInLastFrame();// The last frame map point Whether there is a replacement point, if there is a replacement point, replace it
                     // a. Tracking reference frame mode Small moving speed========================================
                         // No movement Tracking reference keyframes (motion model is empty) or just done retargeting
                         // mCurrentFrame.mnId < mnLastRelocFrameId+2 This judgment should not be
                         // TrackWithMotionModel should be preferred as long as mVlocity is not empty
                         // mnLastRelocFrameId The frame of the last relocation
			if(mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)// The id of the latest relocation
			{
			        // Use the pose of the previous frame as the initial pose of the current frame
                                // Find the matching point of the feature point of the current frame in the reference frame by BoW
                                // The pose can be obtained by optimizing each feature point to correspond to the 3D point reprojection error
			    bOK = TrackReferenceKeyFrame();// If there are more than 10 map points in the tracking reference keyframe, return true
			}
		        // b. There is movement, move first   Tracking mode=======================================
			else
			{
			    // Set the initial pose of the current frame according to the constant velocity model
                            // Find the matching point of the feature point of the current frame in the reference frame of the previous frame by means of projection
                            // The pose can be obtained by optimizing the projection error of the 3D point corresponding to each feature point
			    bOK = TrackWithMotionModel();// Motion tracking mode, track the previous frame
			    if(!bOK)//If unsuccessful, try to track the reference frame mode
				// The position and attitude of the current frame cannot be predicted according to the fixed motion speed model, and the matching is accelerated by bow (SearchByBow)
				// Finally, the optimized pose is obtained by optimization		      
				bOK = TrackReferenceKeyFrame();// If the tracking reference frame mode is greater than 10, return true
			}
		    }
                // B. 更丢了，重定位模式===========================================================================
		    else
		    {
			bOK = Relocalization();//重定位  BOW搜索，PnP 3d-2d匹配 求解位姿
		    }
		}
            // 2. 已经有地图的情况下，则进行 跟踪 + 重定位（跟踪丢失后进行重定位）================================================		
		else
		{
	        // A.跟踪丢失 ======================================================================================
		    if(mState==LOST)
		    {
			bOK = Relocalization();//重定位  BOW搜索，PnP 3d-2d匹配 求解位姿
		    }
                // B. 正常跟踪
		    else
		    {
		        // mbVO 是 mbOnlyTracking 为true时的才有的一个变量
                        // mbVO 为0表示此帧匹配了很多的MapPoints，跟踪很正常，
                        // mbVO 为1表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏      
			if(!mbVO)
			{
		   // a. 上一帧跟踪的点足够多=============================================
                           // 1. 移动跟踪模式, 如果失败，尝试使用跟踪参考帧模式====
			    if(!mVelocity.empty())// 在移动
			    {
				bOK = TrackWithMotionModel();// 恒速跟踪上一帧 模型
				 if(!bOK)// 新添加，如果移动跟踪模式失败，尝试使用 跟踪参考帧模式 进行跟踪
				    bOK = TrackReferenceKeyFrame();
			    }
                           // 2. 使用跟踪参考帧模式===============================
			    else//未移动
			    {
				bOK = TrackReferenceKeyFrame();// 跟踪 参考帧
			    }
			}
	         // b. 上一帧跟踪的点比较少(到了无纹理区域等)，要跪的节奏，既做跟踪又做定位========   	
			else// mbVO为1，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做 运动跟踪 又做 定位	
			{
                            // 使用 运动跟踪 和 重定位模式 计算两个位姿，如果重定位成功，使用重定位得到的位姿
			    bool bOKMM = false;
			    bool bOKReloc = false;
			    vector<MapPoint*> vpMPsMM;
			    vector<bool> vbOutMM;
			    cv::Mat TcwMM;// 视觉里程计跟踪得到的 位姿 结果
			    if(!mVelocity.empty())// 有速度 运动跟踪模式
			    {
				bOKMM = TrackWithMotionModel();// 运动跟踪模式跟踪上一帧 结果
				vpMPsMM = mCurrentFrame.mvpMapPoints;// 地图点
				vbOutMM = mCurrentFrame.mvbOutlier;  // 外点
				TcwMM = mCurrentFrame.mTcw.clone();  // 保存视觉里程计 位姿 结果
			    }
			    
			    bOKReloc = Relocalization();// 重定位模式
                         // 1.重定位没有成功，但运动跟踪 成功,使用跟踪的结果===================================
			    if(bOKMM && !bOKReloc)
			    {
				mCurrentFrame.SetPose(TcwMM);// 把帧的位置设置为 视觉里程计 位姿 结果
				mCurrentFrame.mvpMapPoints = vpMPsMM;// 帧看到的地图点
				mCurrentFrame.mvbOutlier = vbOutMM;// 外点

				if(mbVO)
				{
				  // 这段代码是不是有点多余？应该放到TrackLocalMap函数中统一做
				  // 更新当前帧的MapPoints被观测程度
				    for(int i =0; i<mCurrentFrame.N; i++)
				    {
					if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
					{
					    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
					}
				    }
				}
			    }
		         // 2. 重定位模式 成功=================================================
			    else if(bOKReloc)// 只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）
			    {
				mbVO = false;//重定位成功 
			    }

			    bOK = bOKReloc || bOKMM;// 运动 跟踪 / 重定位 成功标志
			}
		    }
		}

      // 步骤2. 局部地图跟踪=======================================================================================================	
	      // 通过之前的计算，已经得到一个对位姿的初始估计，我们就能透过投影，
	      // 从已经生成的地图点 中找到更多的对应关系，来精确结果
	      // 三种模式的初始跟踪之后  进行  局部地图的跟踪
	      // 局部地图点的描述子 和 当前帧 特征点(还没有匹配到地图点的关键点) 进行描述子匹配
	      // 图优化进行优化  利用当前帧的特征点的像素坐标和 与其匹配的3D地图点  在其原位姿上进行优化
	      // 匹配优化后 成功的点对数 一般情况下 大于30 认为成功
	      // 在刚进行过重定位的情况下 需要大于50 认为成功

		mCurrentFrame.mpReferenceKF = mpReferenceKF;// 参考关键帧
	 // 步骤2.1：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
		// local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
		// 在上面两两帧跟踪（恒速模型跟踪上一帧、跟踪参考帧），
                // 这里搜索局部关键帧 后 搜集所有局部MapPoints，
		// 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
             // 有建图线程
		if(!mbOnlyTracking)// 跟踪 + 建图 + 重定位
		{
		    if(bOK)
			bOK = TrackLocalMap(); // 局部地图跟踪 g20优化 ------------------
		}
             // 无建图线程
		else// 跟踪  + 重定位
		{
		    // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
		    // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
		    // the camera we will use the local map again.
		    if(bOK && !mbVO)// 重定位成功
			bOK = TrackLocalMap();// 局部地图跟踪--------------------------
		}

		if(bOK)
		    mState = OK;
		else
		    mState=LOST;// 丢失

		// Update drawer
		//  更新显示
		mpFrameDrawer->Update(this);


       // 步骤2.2 局部地图跟踪成功, 根系运动模型，清除外点等，检查是否需要创建新的关键帧
		if(bOK)
		{
	       // a. 有运动，则更新运动模型 Update motion model 运动速度为前后两针的 变换矩阵
		    if(!mLastFrame.mTcw.empty())
		    {      
			cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
			mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
			mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
			mVelocity = mCurrentFrame.mTcw*LastTwc;//运动速度 为前后两针的 变换矩阵
		    }
		    else
			mVelocity = cv::Mat();// 无速度
                    // 显示 当前相机位姿 
		    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

               // b. 清除 UpdateLastFrame 中为当前帧 临时添加的 MapPoints	    
                    // 当前帧 的地图点的 观测帧数量小于1 的化 清掉 相应的 地图点
		    for(int i=0; i< mCurrentFrame.N; i++)
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];//当前帧 匹配到的地图点
			if(pMP)//指针存在
			    if(pMP->Observations()<1)// 其观测帧 小于 1
			    {// 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
				mCurrentFrame.mvbOutlier[i] = false;// 外点标志 0 
				mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);// 清掉 相应的 地图点
			    }
		    }
               // c. 清除临时的MapPoints,删除临时的地图点
                  // 这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）
		  // b 中只是在 当前帧 中将这些MapPoints剔除，这里从MapPoints数据库中删除
		  // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
		    //  list<MapPoint*>::iterator 
		    for(auto lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit != lend; lit++)
		    {
			MapPoint* pMP = *lit;
			delete pMP;// 删除地图点 对应的 空间
		    }
		    mlpTemporalPoints.clear();

	      // d. 判断是否需要新建关键帧
		   // 最后一步是确定是否将当前帧定为关键帧，由于在Local Mapping中，
		   // 会剔除冗余关键帧，所以我们要尽快插入新的关键帧，这样才能更鲁棒。
		    if(NeedNewKeyFrame())
			CreateNewKeyFrame();

	      // e. 外点清除 检查外点 标记(不符合 变换矩阵的 点 优化时更新)   
		    for(int i=0; i<mCurrentFrame.N;i++)
		    {
			if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
			    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
		    }
		}
		
      // 3. 跟踪失败后的处理（两两跟踪失败 or 局部地图跟踪失败）======================================================

		if(mState == LOST)
		{
		    if(mpMap->KeyFramesInMap()<=5)// 关键帧数量过少（刚开始建图） 直接退出
		    {
			cout << "跟踪丢失， 正在重置 Track lost soon after initialisation, reseting..." << endl;
			mpSystem->Reset();
			return;
		    }
		}

		if(!mCurrentFrame.mpReferenceKF)
		    mCurrentFrame.mpReferenceKF = mpReferenceKF;

		mLastFrame = Frame(mCurrentFrame);//新建关键帧
	    }

   
// 步骤3: 返回跟踪得到的位姿 信息=======================================================================
            // 计算参考帧到当前帧 的变换 Tcr = mTcw  * mTwr 
	    if(!mCurrentFrame.mTcw.empty())
	    {
	        // mTcw  * mTwr  = mTcr
		cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
		mlRelativeFramePoses.push_back(Tcr);
		mlpReferences.push_back(mpReferenceKF);
		mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
		mlbLost.push_back(mState==LOST);
	    }
	    else//跟踪丢失 会造成  位姿为空 
	    {
		// This can happen if tracking is lost
		mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
		mlpReferences.push_back(mlpReferences.back());
		mlFrameTimes.push_back(mlFrameTimes.back());
		mlbLost.push_back(mState==LOST);
	    }

  }
// 以上为 Tracking部分
	

// 当前帧 特征点个数 大于500 进行初始化
// 设置第一帧为关键帧  位姿为 [I 0] 
// 根据第一帧视差求得的深度 计算3D点
// 生成地图 添加地图点 地图点观测帧 地图点最好的描述子 更新地图点的方向和距离 
// 关键帧的地图点 当前帧添加地图点  地图添加地图点
// 显示地图

/**
 * @brief 双目和rgbd的地图初始化
 *
 * 由于具有深度信息，直接生成MapPoints
 */
// 第一帧 双目 / 深度初始化 
	void Tracking::StereoInitialization()
	{
	    if(mCurrentFrame.N>500)
  // 【0】找到的关键点个数 大于 500 时进行初始化将当前帧构建为第一个关键帧
	    {
		// Set Frame pose to the origin
       //【1】 初始化 第一帧为世界坐标系原点 变换矩阵 对角单位阵 R = eye(3,3)   t=zero(3,1)
// 步骤1：设定初始位姿
		mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

       // 【2】创建第一帧为关键帧  Create KeyFrame  普通帧      地图       关键帧数据库
		// 加入地图 加入关键帧数据库
// 步骤2：将当前帧构造为初始关键帧
		// mCurrentFrame的数据类型为Frame
		// KeyFrame包含Frame、地图3D点、以及BoW
		// KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
		// KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
		KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
		// 地图添加第一帧关键帧 关键帧存入地图关键帧set集合里 Insert KeyFrame in the map
      
		// KeyFrame中包含了地图、反过来地图中也包含了KeyFrame，相互包含
// 步骤3：在地图中添加该初始关键帧
		mpMap->AddKeyFrame(pKFini);// 地图添加 关键帧

		// Create MapPoints and asscoiate to KeyFrame
                // 【3】创建地图点 并关联到 相应的关键帧  关键帧也添加地图点  地图添加地图点 地图点描述子 距离
// 步骤4：为每个特征点构造MapPoint		
		for(int i=0; i<mCurrentFrame.N;i++)// 该帧的每一个关键点
		{
		    float z = mCurrentFrame.mvDepth[i];// 关键点对应的深度值  双目和 深度相机有深度值
		    if(z>0)// 有效深度 
		    {
		   // 步骤4.1：通过反投影得到该特征点的3D坐标  
			cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);// 投影到 在世界坐标系下的三维点坐标
		   // 步骤4.2：将3D点构造为MapPoint	
			// 每个 具有有效深度 关键点 对应的3d点 转换到 地图点对象
			MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
		  // 步骤4.3：为该MapPoint添加属性：
			// a.观测到该MapPoint的关键帧
			// b.该MapPoint的描述子
			// c.该MapPoint的平均观测方向和深度范围
			
                         // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
			pNewMP->AddObservation(pKFini,i);// 地图点添加 观测 参考帧 在该帧上可一观测到此地图点
			 // b.从众多观测到该MapPoint的特征点中挑选区分读最高的描述子
			pNewMP->ComputeDistinctiveDescriptors();// 地图点计算最好的 描述子
			// c.更新该MapPoint平均观测方向以及观测距离的范围
			// 该地图点平均观测方向与观测距离的范围，这些都是为了后面做描述子融合做准备。
			pNewMP->UpdateNormalAndDepth();
			// 更新 相对 帧相机中心 单位化相对坐标  金字塔层级 距离相机中心距离
		   // 步骤4.4：在地图中添加该MapPoint
			mpMap->AddMapPoint(pNewMP);// 地图 添加 地图点
                   // 步骤4.5：表示该KeyFrame的哪个特征点可以观测到哪个3D点
			 pKFini->AddMapPoint(pNewMP,i);
		   // 步骤4.6：将该MapPoint添加到当前帧的mvpMapPoints中
                        // 为当前Frame的特征点与MapPoint之间建立索引
			mCurrentFrame.mvpMapPoints[i]=pNewMP;//当前帧 添加地图点
		    }
		}
		cout << "新地图创建成功 new map ,具有 地图点数 : " << mpMap->MapPointsInMap() << "  地图点 points" << endl;
 // 步骤5：在局部地图中添加该初始关键帧
		// 【4】局部建图添加关键帧  局部关键帧添加关键帧     局部地图点添加所有地图点
		mpLocalMapper->InsertKeyFrame(pKFini);
               // 记录
		mLastFrame = Frame(mCurrentFrame);// 上一个 普通帧
		mnLastKeyFrameId=mCurrentFrame.mnId;// id
	 	mpLastKeyFrame = pKFini;// 上一个关键帧
               // 局部
		mvpLocalKeyFrames.push_back(pKFini);// 局部关键帧 添加 关键帧
		mvpLocalMapPoints=mpMap->GetAllMapPoints();//局部地图点  添加所有地图点
		mpReferenceKF = pKFini;// 参考帧
		mCurrentFrame.mpReferenceKF = pKFini;//当前帧 参考关键帧
                // 地图
		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);//地图 参考地图点
		mpMap->mvpKeyFrameOrigins.push_back(pKFini);// 地图关键帧
                // 可视化
		mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
		mState=OK;// 跟踪正常
	    }
	}


/**
 * @brief 单目的地图初始化    第一帧 单目初始化	
 *单目的初始化有专门的初始化器，只有连续的两帧特征点 均>100 个才能够成功构建初始化器
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
	void Tracking::MonocularInitialization()
	{
// 【1】添加第一帧 设置参考帧
	    if(!mpInitializer)// 未初始化成功 进行初始化 Initializer  得到  R  t 和 3D点
	    {
		// 设置参考帧   用作匹配的帧 Set Reference Frame
		if(mCurrentFrame.mvKeys.size()>100)// 关键点个数超过 100个 才进行初始化
		{
		    mInitialFrame = Frame(mCurrentFrame);// 初始帧
		    mLastFrame = Frame(mCurrentFrame);// 上一帧
		    mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());// 是第一帧中的所有特征点
		    for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
			mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;//匹配点横坐标
			
                    // 这两句是多余的
		   if(mpInitializer)
			delete mpInitializer;
		    
                    // 再次初始化
		    mpInitializer =  new Initializer(mCurrentFrame,1.0,200);// 方差 和 迭代次数
		    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
		    return;
		}
	    }
// 【2】添加第二帧 参考帧设置完成后 根据  当前帧关键点数量选择是否初始化
	    else// 第一帧初始化成功   当前帧和参考帧 做匹配得到 R t
	    {
		// Try to initialize
     //【3】重新初始化 设置参考帧     
		if((int)mCurrentFrame.mvKeys.size()<=100)//只有连续的两帧特征点 均>100 个才能够成功构建初始化器
		{
		    delete mpInitializer;
		    mpInitializer = static_cast<Initializer*>(NULL);// 重新初始化
		    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
		    return;
		}
    // 【4】当前帧特征点数较多 和参考帧寻找匹配点对 根据匹配点对数 确定是否 初始化
		//  寻找匹配点对   mvIniMatches
		ORBmatcher matcher(0.9,true);
		// mInitialFrame 第一帧  mCurrentFrame当前帧第二帧 
		// mvbPreMatched是第一帧中的所有特征点；
		// mvIniMatches标记匹配状态，未匹配上的标为-1；
		//如果返回nmatches<100,初始化失败，重新初始化过程
		// 100 块匹配 搜索窗口大小尺寸
		int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
     //【5】 匹配点对过少 重新初始化 Check if there are enough correspondences
		if(nmatches<100)
		{
		    delete mpInitializer;
		    mpInitializer = static_cast<Initializer*>(NULL);
		    return;
		}
               
        // 【6】匹配点对数量 较多进行初始化 计算相机的移动位姿 根据 基础矩阵 F 或者 单应矩阵 H 计算初始 R t
		cv::Mat Rcw; //当前相机 旋转矩阵 Current Camera Rotation
		cv::Mat tcw; // 平移矩阵 Current Camera Translation
		vector<bool> vbTriangulated; // 符合变换矩阵的内点 且三角化后3D三维坐标正常的点 标志
		// Triangulated Correspondences (mvIniMatches)	
 // * 单目相机初始化
//* 用于平面场景的单应性矩阵H(8中运动假设) 和用于非平面场景的基础矩阵F(4种运动假设)
//* 然后通过一个评分规则来选择合适的模型，恢复相机的旋转矩阵R和平移向量t 和 对应的3D点(尺度问题)  好坏点标志
  // 	并行计算分解基础矩阵和单应矩阵（获取的点恰好位于同一个平面），得到帧间运动（位姿），vbTriangulated标记一组特征点能否进行三角化。
  // mvIniP3D 是cv::Point3f类型的一个容器，是个存放三角化得到的 3D点 的 临时变量。
 
		if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
	     // 最关键算法是通过初始连续两帧的对极约束恢复出相机姿态和地图点 
		{
		    for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
		    {
			if(mvIniMatches[i]>=0 && !vbTriangulated[i])// 是匹配点 但是 匹配点 不在求出的变换上
			{
			    mvIniMatches[i]=-1;//此匹配点不好
			    nmatches--;//匹配点对数 - 1
			}
		    }

	 // 【7】设置初始参考帧的世界坐标位姿态  对角矩阵  Set Frame Poses
		    mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
	// 【8】设置第二帧(当前帧)的位姿	    
		    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
		    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
		    tcw.copyTo(Tcw.rowRange(0,3).col(3));		    
		    mCurrentFrame.SetPose(Tcw);
        // 【9】创建地图 使用 最小化重投影误差BA 进行 地图优化 优化位姿 和地图点
		    CreateInitialMapMonocular();
		}
	    }
	}


 /**
 * @brief CreateInitialMapMonocular
 * 初始帧设置为世界坐标系原点 初始化后 解出来的 当前帧位姿T 最小化重投影误差  BA 全局优化位姿 T
 * 为单目摄像头三角化生成MapPoints
 */
	void Tracking::CreateInitialMapMonocular()
	{
  //【1】创建关键帧 Create KeyFrames
      // 构建初始地图就是将这两关键帧以及对应的地图点加入地图（mpMap）中，需要分别构造关键帧以及地图点
	    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);// 初始关键帧 加入地图 加入关键帧数据库
	    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);//当前关键帧 第二帧
  //【2】计算帧 描述子 在 描述子词典 中的 线性表示向量
	    pKFini->ComputeBoW();
	    pKFcur->ComputeBoW();

   // 【3】地图中插入关键帧 Insert KFs in the map
	    mpMap->AddKeyFrame(pKFini);
	    mpMap->AddKeyFrame(pKFcur);

   // 【4】创建地图点 关联到 关键帧 Create MapPoints and asscoiate to keyframes
	   // 地图点中需要加入其一些属性：
	  //1. 观测到该地图点的关键帧（对应的关键点）；
	  //2. 该MapPoint的描述子；
	  //3. 该MapPoint的平均观测方向和观测距离。
	    for(size_t i=0; i<mvIniMatches.size();i++)
	    {
		if(mvIniMatches[i]<0)// 不好的匹配不要
		    continue;

	// 【5】创建地图点 Create MapPoint.
		cv::Mat worldPos(mvIniP3D[i]);// mvIniP3D 三角化得到的 3D点  vector 3d转化成 mat 3d
		MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

		pKFini->AddMapPoint(pMP,i);// 初始帧 添加地图点
		pKFcur->AddMapPoint(pMP,mvIniMatches[i]);// 当前帧 添加地图点
		
        // 【6】地图点 添加观测帧  参考帧和当前帧 均可以观测到 该地图点
		pMP->AddObservation(pKFini,i);
		pMP->AddObservation(pKFcur,mvIniMatches[i]);
		
        // 【7】 更新地图点的一些新的参数 描述子 观测方向 观测距离
		pMP->ComputeDistinctiveDescriptors();// 地图点 在 所有观测帧上的 最具有代表性的 描述子
		pMP->UpdateNormalAndDepth();// 该MapPoint的平均观测方向和观测距离。
		// 更新 地图点 相对各个 观测帧 相机中心 单位化坐标
	       // 更新 地图点 在参考帧下 各个金字塔层级 下的  最小最大距离
        // 【8】当前帧 关联到地图点
		//Fill Current Frame structure
		mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
		mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;// 是好的点  离群点标志
	// 【9】地图 添加地图点
		//Add to Map
		mpMap->AddMapPoint(pMP);
	    }

	    // Update Connections
     // 【10】跟新关键帧的 连接关系   被观测的次数
	   //还需要更新关键帧之间连接关系（以共视地图点的数量作为权重）：
	    pKFini->UpdateConnections();
	    pKFcur->UpdateConnections();

	    // Bundle Adjustment
	    cout << "新地图 New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
        // 【11】 全局优化地图 BA最小化重投影误差
	    Optimizer::GlobalBundleAdjustemnt(mpMap,20);// 对这两帧姿态进行全局优化重投影误差（LM）：
	      // 注意这里使用的是全局优化，和回环检测调整后的大回环优化使用的是同一个函数。
	    
        // 【12】设置 深度中值 为 1 Set median depth to 1
	    // 需要归一化第一帧中地图点深度的中位数；
	    float medianDepth = pKFini->ComputeSceneMedianDepth(2);//  单目 环境 深度中值
	    float invMedianDepth = 1.0f/medianDepth;
        // 【13】检测重置  如果深度<0 或者 这时发现 优化后 第二帧追踪到的地图点<100，也需要重新初始化。
	    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100 )
	    {
		cout << "初始化错误 重置 Wrong initialization, reseting..." << endl;
		Reset();
		return;
	    }
       // 【14】关键帧 位姿 平移量尺度归一化
         // 否则，将深度中值作为单位一，归一化第二帧的位姿与所有的地图点。
	    // Scale initial baseline
	    cv::Mat Tc2w = pKFcur->GetPose();// 位姿
	    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;// 平移量归一化尺度
	    pKFcur->SetPose(Tc2w);//设置新的位姿

        // 【15】地图点 尺度归一化 Scale points
	    // 地图点 归一化尺度
	    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
	    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
	    {
		if(vpAllMapPoints[iMP])
		{
		    MapPoint* pMP = vpAllMapPoints[iMP];
		    pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);//地图点尺度归一化
		}
	    }
   // 【16】 对象更新
            // 局部地图插入关键帧
	    mpLocalMapper->InsertKeyFrame(pKFini);
	    mpLocalMapper->InsertKeyFrame(pKFcur);
           // 当前帧 更新位姿
	    mCurrentFrame.SetPose(pKFcur->GetPose());
	    mnLastKeyFrameId=mCurrentFrame.mnId;// 当前帧 迭代到上一帧  为下一次迭代做准备
	    mpLastKeyFrame = pKFcur;// 指针
           // 局部关键帧 局部地图点更新
	    mvpLocalKeyFrames.push_back(pKFcur);
	    mvpLocalKeyFrames.push_back(pKFini);
	    mvpLocalMapPoints=mpMap->GetAllMapPoints();
	    mpReferenceKF = pKFcur;// 参考关键帧
	    mCurrentFrame.mpReferenceKF = pKFcur;// 当前帧的 参考帧

	    mLastFrame = Frame(mCurrentFrame);// 当前帧 迭代到上一帧  为下一次迭代做准备
           // 参考地图点
	    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
           // 地图显示
	    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
           // 地图关键帧序列
	    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
	    mState=OK;// 状态 ok
	}
	
/**
 * @brief 检查上一帧中的MapPoints是否被替换
 * 核对 替换 关键帧 地图点
 * 最后一帧 地图点 是否有替换点 有替换点的则进行替换
 * Local Mapping线程可能会将关键帧中某些MapPoints进行替换，由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 * @see LocalMapping::SearchInNeighbors()
 */
	void Tracking::CheckReplacedInLastFrame()
	{
	    for(int i =0; i<mLastFrame.N; i++)
	    {
		MapPoint* pMP = mLastFrame.mvpMapPoints[i];

		if(pMP)
		{
		    MapPoint* pRep = pMP->GetReplaced();// 有替换点
		    if(pRep)
		    {
			mLastFrame.mvpMapPoints[i] = pRep;// 进行替换
		    }
		}
	    }
	}

// 跟踪参考帧  机器人没怎么移动
// 当前帧特征点描述子 和 参考关键帧帧中的地图点 的描述子 进行 匹配
 // 保留方向直方图中最高的三个bin中 关键点 匹配的 地图点  匹配点对
// 采用 词带向量匹配
// 关键帧和 当前帧 均用 字典单词线性表示
// 对应单词的 描述子 肯定比较相近 取对应单词的描述子进行匹配可以加速匹配
// 和参考关键帧的地图点匹配  匹配点对数 需要大于15个
// 使用 图优化 根据地图点 和 帧对应的像素点  在初始位姿的基础上 优化位姿
// 同时剔除  外点
// 最终超过10个 匹配点 的 返回true 跟踪成功
/**
 * @brief 对参考关键帧的MapPoints进行跟踪
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
	bool Tracking::TrackReferenceKeyFrame()
	{ 
	    // Compute Bag of Words vector
	  // 计算当前帧 特征描述子的词带向量
	    mCurrentFrame.ComputeBoW();// 当前帧 所有特征点描述子 用字典单词线性表示

	    // We perform first an ORB matching with the reference keyframe
	    // If enough matches are found we setup a PnP solver
	    ORBmatcher matcher(0.7,true);// orb特征 匹配器   0.7 鲁棒匹配系数
	    vector<MapPoint*> vpMapPointMatches;
	    
            // 计算 当前帧 和 参考关键帧帧之间的 特征匹配 返回匹配点对个数
	    // 当前帧 和 参考关键帧 中的地图点  进行特征匹配  匹配到已有地图点
	    // 当前帧每个关键点的描述子 和 参考关键帧每个地图点的描述子匹配 
	    // 保留距离最近的匹配地图点 且最短距离和 次短距离相差不大 （ mfNNratio）
	    // 如果需要考虑关键点的方向信息
	    // 统计当前帧 关键点的方向 到30步长 的方向直方图
	    // 保留方向直方图中最高的三个bin中 关键点 匹配的 地图点  匹配点对
	    // 关键帧和 当前帧 均用 字典单词线性表示
            // 对应单词的 描述子 肯定比较相近 取对应单词的描述子进行匹配可以加速匹配
	    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

	    if(nmatches<15)// 和参考关键帧匹配 匹配点对数 需要大于15个
		return false;

	    mCurrentFrame.mvpMapPoints = vpMapPointMatches;// 地图点
	    mCurrentFrame.SetPose(mLastFrame.mTcw);// 位姿 初始为上一帧的 位姿
	    Optimizer::PoseOptimization(&mCurrentFrame);// 优化位姿 同时标记 是否符合 变换矩阵 Rt 不符合的是外点 
	    // 使用 图优化 根据地图点 和 帧对应的像素点  在初始位姿的基础上 优化位姿

	    // Discard outliers
	    // 去除外点 对应的匹配地图点  
	    int nmatchesMap = 0;
	    for(int i =0; i<mCurrentFrame.N; i++)//每个关键点
	    {
		if(mCurrentFrame.mvpMapPoints[i])// 如果有对应 匹配到的 地图点
		{
		    if(mCurrentFrame.mvbOutlier[i])//是外点需要删除  外点 不符合变换关系的点  优化时更新
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

			mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);//删除匹配点
			mCurrentFrame.mvbOutlier[i]=false;//无匹配地图点  外点标志 置为否
			pMP->mbTrackInView = false;
			pMP->mnLastFrameSeen = mCurrentFrame.mnId;
			nmatches--;
		    }
		    else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)//是内点同事 有 观测关键帧
			nmatchesMap++;
		}
	    }

	    return nmatchesMap >= 10;
	}

// 更新 上一帧
// 更新 上一帧 位姿    =  世界到 上一帧的 参考帧  再到 上一帧
// 更新上一帧 地图点
/**
 * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
 *
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints
 */
	void Tracking::UpdateLastFrame()
	{
	    // Update pose according to reference keyframe
	    KeyFrame* pRef = mLastFrame.mpReferenceKF;// 参考帧
	    cv::Mat Tlr = mlRelativeFramePoses.back();//上一帧的 参考帧 到 上一帧 的变换 Tlr
	    mLastFrame.SetPose(Tlr*pRef->GetPose());//上一帧位姿态 =  世界到 上一帧的 参考帧  再到 上一帧

	    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
		return;

	    // Create "visual odometry" MapPoints
	    // We sort points according to their measured depth by the stereo/RGB-D sensor
	    // 以下 双目/深度相机 执行
	    vector<pair<float,int> > vDepthIdx;
	    vDepthIdx.reserve(mLastFrame.N);
	    for(int i=0; i<mLastFrame.N;i++)
	    {
		float z = mLastFrame.mvDepth[i];// 关键点对应的深度
		if(z>0)
		{
		    vDepthIdx.push_back(make_pair(z,i));
		}
	    }

	    if(vDepthIdx.empty())
		return;

	    sort(vDepthIdx.begin(),vDepthIdx.end());//深度排序

	    // We insert all close points (depth < mThDepth)
	    // If less than 100 close points, we insert the 100 closest ones.
	    int nPoints = 0;
	    for(size_t j=0; j<vDepthIdx.size();j++)
	    {
		int i = vDepthIdx[j].second;

		bool bCreateNew = false;

		MapPoint* pMP = mLastFrame.mvpMapPoints[i];// 上一帧对应的 地图点
		if(!pMP)
		    bCreateNew = true;// 重新生成标志
		else if(pMP->Observations()<1)// 地图点对应的观测帧 数量1个
		{
		    bCreateNew = true;
		}

		if(bCreateNew)//重新生成 3D点
		{
		    cv::Mat x3D = mLastFrame.UnprojectStereo(i);// 生成 3D点
		    MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

		    mLastFrame.mvpMapPoints[i]=pNewMP;

		    mlpTemporalPoints.push_back(pNewMP);
		    nPoints++;
		}
		else
		{
		    nPoints++;
		}

		if(vDepthIdx[j].first>mThDepth && nPoints>100)
		    break;
	    }
	}

// 移动模式跟踪  移动前后两帧  得到 变换矩阵
// 上一帧的地图点 反投影到当前帧图像像素坐标上  和 当前帧的 关键点落在 同一个 格子内的 
// 做描述子匹配 搜索 可以加快匹配
/*
 使用匀速模型估计的位姿，将LastFrame中临时地图点投影到当前姿态，
 在投影点附近根据描述子距离进行匹配（需要>20对匹配，否则匀速模型跟踪失败，
 运动变化太大时会出现这种情况），然后以运动模型预测的位姿为初值，优化当前位姿，
 优化完成后再剔除外点，若剩余的匹配依然>=10对，
 则跟踪成功，否则跟踪失败，需要Relocalization：
 
  运动模型（Tracking with motion model）跟踪   速率较快  假设物体处于匀速运动
      用 上一帧的位姿和速度来估计当前帧的位姿使用的函数为TrackWithMotionModel()。
      这里匹配是通过投影来与上一帧看到的地图点匹配，使用的是
      matcher.SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, ...)。
 */
/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
	bool Tracking::TrackWithMotionModel()
	{
	  
	    ORBmatcher matcher(0.9,true);// 匹配点匹配器 最小距离 < 0.9*次短距离 匹配成功

	    // Update last frame pose according to its reference keyframe
	    // Create "visual odometry" points if in Localization Mode
	    // 更新 上一帧 位姿    =  世界到 上一帧的 参考帧  再到 上一帧
            // 更新上一帧 地图点
	    UpdateLastFrame();// 

	    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
	    // 当前帧位姿 mVelocity 为当前帧和上一帧的 位姿变换
            // 初始化空指针
	    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

	    // Project points seen in previous frame
	    int th;
	    if(mSensor  != System::STEREO)
		th=15;// 搜索窗口
	    else
		th=7;
	    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

	    // If few matches, uses a wider window search
	    if(nmatches<20)
	    {
		fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
		nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
	    }

	    if(nmatches<20)
		return false;

	    // Optimize frame pose with all matches
	    Optimizer::PoseOptimization(&mCurrentFrame);

	    // Discard outliers
	    int nmatchesMap = 0;
	    for(int i =0; i<mCurrentFrame.N; i++)
	    {
		if(mCurrentFrame.mvpMapPoints[i])
		{
		    if(mCurrentFrame.mvbOutlier[i])//外点
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];// 当前帧特征点 匹配到的 地图点

			mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
			mCurrentFrame.mvbOutlier[i]=false;
			pMP->mbTrackInView = false;
			pMP->mnLastFrameSeen = mCurrentFrame.mnId;
			nmatches--;
		    }
		    else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
			nmatchesMap++;
		}
	    }    

	    if(mbOnlyTracking)
	    {
		mbVO = nmatchesMap < 10;
		return nmatches > 20;
	    }

	    return nmatchesMap>=10;
	}

	
// 三种模式的初始跟踪之后  进行  局部地图的跟踪
// 局部地图点的描述子 和 当前帧 特征点(还没有匹配到地图点的关键点) 进行描述子匹配
// 图优化进行优化  利用当前帧的特征点的像素坐标和 与其匹配的3D地图点  在其原位姿上进行优化
// 匹配优化后 成功的点对数 一般情况下 大于30 认为成功
// 在刚进行过重定位的情况下 需要大于50 认为成功
/*
 以上两种仅仅完成了视觉里程计中的帧间跟踪，
 还需要进行局部地图的跟踪，提高精度：（这其实是Local Mapping线程中干的事情）
 局部地图跟踪TrackLocalMap()中需要
 首先对局部地图进行更新(UpdateLocalMap)，
 并且搜索局部地图点(SearchLocalPoint)。
 局部地图的更新又分为
 局部地图点(UpdateLocalPoints) 和
 局部关键帧(UpdateLocalKeyFrames)的更新.
 
 为了降低复杂度，这里只是在局部图中做投影。局部地图中与当前帧有相同点的关键帧序列成为K1，
 在covisibility graph中与K1相邻的称为K2。局部地图有一个参考关键帧Kref∈K1，
 它与当前帧具有最多共同看到的地图云点。针对K1, K2可见的每个地图云点，
 通过如下步骤，在当前帧中进行搜索:
 
（1）将地图点投影到当前帧上，如果超出图像范围，就将其舍弃；
（2）计算当前视线方向向量v与地图点云平均视线方向向量n的夹角，舍弃n·v < cos(60°)的点云；
（3）计算地图点到相机中心的距离d，认为[dmin, dmax]是尺度不变的区域，若d不在这个区域，就将其舍弃；
（4）计算图像的尺度因子，为d/dmin；
（5）将地图点的特征描述子D与还未匹配上的ORB特征进行比较，根据前面的尺度因子，找到最佳匹配。
  这样，相机位姿就能通过匹配所有地图点，最终被优化。
  
 */
/**
 * @brief 对Local Map的MapPoints进行跟踪
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
	bool Tracking::TrackLocalMap()
	{
	    // We have an estimation of the camera pose and some map points tracked in the frame.
	    // We retrieve the local map and try to find matches to points in the local map.
// 【1】首先对局部地图进行更新(UpdateLocalMap) 生成对应当前帧的 局部地图 
	     // 更新局部地图(与当前帧相关的帧和地图点) 用于 局部地图点的跟踪   关键帧 + 地图点
	     // 更新局部关键帧-------局部地图的一部分  共视化程度高的关键帧  子关键帧   父关键帧
	     // 局部地图点的更新比较容易，完全根据 局部关键帧来，所有 局部关键帧的地图点就构成 局部地图点
	    UpdateLocalMap();
// 【2】并且搜索局部地图点(SearchLocalPoint)
	    // 局部地图点 搜寻和当前帧 关键点描述子 的匹配 有匹配的加入到 当前帧 特征点对应的地图点中
	    SearchLocalPoints();

 // 【3】优化帧位姿 Optimize Pose
	    Optimizer::PoseOptimization(&mCurrentFrame);
	    // 优化时会更新 当前帧的位姿变换关系 同时更新地图点的内点/外点标记
	    mnMatchesInliers = 0;

 // 【4】更新地图点状态 Update MapPoints Statistics
	    for(int i=0; i<mCurrentFrame.N; i++)
	    {
		if(mCurrentFrame.mvpMapPoints[i])//特征点找到 地图点
		{
		    if(!mCurrentFrame.mvbOutlier[i])//是内点 符合 变换关系
		    {
			mCurrentFrame.mvpMapPoints[i]->IncreaseFound();// 特征点找到 地图点标志
			if(!mbOnlyTracking)
			{
			    if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
				mnMatchesInliers++;//
			}
			else
			    mnMatchesInliers++;
		    }
		    else if(mSensor == System::STEREO)// 外点 在双目下  清空匹配的地图点
			mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
		}
	    }

	    // Decide if the tracking was succesful
	    // More restrictive if there was a relocalization recently
	    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers<50)
		return false;//刚刚进行过重定位 则需要 匹配点对数大于 50 才认为 成功

	    if(mnMatchesInliers<30)//正常情况下  找到的匹配点对数 大于 30 算成功
		return false;
	    else
		return true;
   }
	
// 更新局部地图(与当前帧相关的帧和地图点) 用于 局部地图点的跟踪   关键帧 + 地图点
// 更新局部关键帧-------局部地图的一部分  共视化程度高的关键帧  子关键帧   父关键帧
// 局部地图点的更新比较容易，完全根据 局部关键帧来，所有 局部关键帧的地图点就构成 局部地图点
/**
 * @brief 断当前帧是否为关键帧
 * @return true if needed
 */
	void Tracking::UpdateLocalMap()
	{
	    // This is for visualization 可视化
	    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

	    // Update
	    UpdateLocalKeyFrames();//更新关键帧
	    UpdateLocalPoints();//更新地图点
	}
	
// 更新局部关键帧-------局部地图的一部分
// 如何去选择当前帧对应的局部关键帧
 // 始终限制关键数量不超过80
 // 可以修改  这里 比较耗时
 // 但是改小 精度可能下降
/*
 当关键帧数量较少时(<=80)，考虑加入第二部分关键帧，
 是与第一部分关键帧联系紧密的关键帧，并且始终限制关键数量不超过80。
 联系紧密体现在三类：
 1. 共视化程度高的关键帧  观测到当前帧地图点 次数多的 关键帧；
 2. 子关键帧；
 3. 父关键帧。

还有一个关键的问题是：如何判断该帧是否关键帧，以及如何将该帧转换成关键帧？
调用
NeedNewKeyFrame() 和
CreateNewKeyFrame()  两个函数来完成。
 */
/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
 */
     void Tracking::UpdateLocalKeyFrames()
	{
	    // Each map point vote for the keyframes in which it has been observed
	  // 更新地图点 的 观测帧
	    map<KeyFrame*,int> keyframeCounter;
	    for(int i=0; i<mCurrentFrame.N; i++)
	    {
		if(mCurrentFrame.mvpMapPoints[i])//当前帧 的地图点
		{
		    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
		    if(!pMP->isBad())// 被观测到
		    {
			const map<KeyFrame*,size_t> observations = pMP->GetObservations();
			for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
			    keyframeCounter[it->first]++;// 地图点的观测帧 观测地图点次数++
		    }
		    else
		    {
			mCurrentFrame.mvpMapPoints[i]=NULL;//未观测到  地图点清除
		    }
		}
	    }

	    if(keyframeCounter.empty())
		return;

	    int max=0;
	    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

	    mvpLocalKeyFrames.clear();// 局部关键帧清空
	    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

	    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
	    // map<KeyFrame*,int>::const_iterator
//  1. 共视化程度高的关键帧 观测到当前帧地图点 次数多的 关键帧；	    
	    for( auto it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
	    {
		KeyFrame* pKF = it->first;//地图点的 关键帧

		if(pKF->isBad())
		    continue;
		if(it->second > max)// 观测到 地图点数量最多的 关键帧
		{
		    max = it->second;
		    pKFmax=pKF;
		}
		mvpLocalKeyFrames.push_back(it->first);// 保存 局部关键帧
		pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
	    }

	    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
	    // vector<KeyFrame*>::const_iterator
	    // 
	    for(auto itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
	    {
		// Limit the number of keyframes
       // 始终限制关键数量不超过80
		if(mvpLocalKeyFrames.size()>80)
		    break;

		KeyFrame* pKF = *itKF;
                // 根据权重w  二分查找 有序序列 中的某写对象
                // 返回前 w个 有序关键帧
		const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
		// vector<KeyFrame*>::const_iterator
		for(auto itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
		{
		    KeyFrame* pNeighKF = *itNeighKF;
		    if(!pNeighKF->isBad())
		    {
			if(pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
			{
			    mvpLocalKeyFrames.push_back(pNeighKF);// 加入 局部关键帧
			    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
			    break;
			}
		    }
		}
  // 2. 子关键帧；
		const set<KeyFrame*> spChilds = pKF->GetChilds();
		 // set<KeyFrame*>::const_iterator
		for(auto sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
		{
		    KeyFrame* pChildKF = *sit;
		    if(!pChildKF->isBad())
		    {
			if(pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
			{
			    mvpLocalKeyFrames.push_back(pChildKF);// 加入 局部关键帧
			    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
			    break;
			}
		    }
		}
// 3. 父关键帧
		KeyFrame* pParent = pKF->GetParent();
		if(pParent)
		{
		    if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
		    {
			mvpLocalKeyFrames.push_back(pParent);// 加入 局部关键帧
			pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
			break;
		    }
		}

	    }

	    if(pKFmax)
	    {
		mpReferenceKF = pKFmax;
		mCurrentFrame.mpReferenceKF = mpReferenceKF;
	    }
	}
	
	
/**
 * @brief 更新局部关键点，called by UpdateLocalMap()
 *  更新 局部地图点
 * 局部地图点的更新比较容易，完全根据 局部关键帧来，所有 局部关键帧的地图点就构成 局部地图点
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 */
	void Tracking::UpdateLocalPoints()
	{
	    mvpLocalMapPoints.clear();
	    // vector<KeyFrame*>::const_iterator
	    for(auto itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
	    {
		KeyFrame* pKF = *itKF;// 每一个 局部关键帧
		// 局部关键帧的地图点
		const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
		//  每一个 局部关键帧 的地图点 
		// vector<MapPoint*>::const_iterator
		for( auto itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
		{
		    MapPoint* pMP = *itMP;//每一个 局部地图点 
		    if(!pMP)// 空的点直接跳过
			continue;
		    if(pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)// 已经更新过了
			continue;
		    if(!pMP->isBad())
		    {
			mvpLocalMapPoints.push_back(pMP);// 更新
			pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
		    }
		}
	    }
	}
	

// 需要 关键帧 吗
/*
确定关键帧的标准如下：
（1）在上一个全局重定位后，又过了20帧；
（2）局部建图闲置，或在上一个关键帧插入后，又过了20帧；
（3)当前帧跟踪到大于50个点；
（4）当前帧跟踪到的比参考关键帧少90%。
*/
/**
 * @brief 断当前帧是否为关键帧
 * @return true if needed
 */
	bool Tracking::NeedNewKeyFrame()
	{
 // 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
            // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
	    if(mbOnlyTracking)// 不建图 不需要关键帧
		return false;

	    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
	    // 建图线程 停止了
	    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
		return false;
	    // 地图中的 关键帧 数量
	    const int nKFs = mpMap->KeyFramesInMap();

	    // Do not insert keyframes if not enough frames have passed from last relocalisation
	    // 刚刚重定位不久不需要插入关键帧  关键帧总数超过最大值也不需要 插入关键帧
    // Do not insert keyframes if not enough frames have passed from last relocalisation
// 步骤2：判断是否距离上一次插入关键帧的时间太短
	      // mCurrentFrame.mnId是当前帧的ID
	      // mnLastRelocFrameId是最近一次重定位帧的ID
	      // mMaxFrames等于图像输入的帧率
	      // 如果关键帧比较少，则考虑插入关键帧
	      // 或距离上一次重定位超过1s，则考虑插入关键帧
	    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
		return false;
	    
// 步骤3：得到参考关键帧跟踪到的MapPoints数量
	// 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
	    // Tracked MapPoints in the reference keyframe
	    int nMinObs = 3;
	    if(nKFs <= 2)
		nMinObs = 2;
	    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
// 步骤4：查询局部地图管理器是否繁忙
	    // Local Mapping accept keyframes?
	    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
	    
// 步骤5：对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
	    // Check how many "close" points are being tracked and how many could be potentially created.
	    int nNonTrackedClose = 0;
	    int nTrackedClose= 0;
	    if(mSensor != System::MONOCULAR)// 双目或rgbd
	    {
		for(int i =0; i<mCurrentFrame.N; i++)
		{
		    if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
		    {
			if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
			    nTrackedClose++;
			else
			    nNonTrackedClose++;
		    }
		}
	    }

	    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);
	    
// 步骤6：决策是否需要插入关键帧
	    // Thresholds
	    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
	    // Thresholds 设定inlier阈值，和之前帧特征点匹配的inlier比例
	    float thRefRatio = 0.75f;
	    if(nKFs<2)
		thRefRatio = 0.4f;// 关键帧只有一帧，那么插入关键帧的阈值设置很低

	    if(mSensor==System::MONOCULAR)
		thRefRatio = 0.9f;

	    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
	    // 很长时间没有插入关键帧
	    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId+mMaxFrames;
	    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
	    // localMapper处于空闲状态
	    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
	    //Condition 1c: tracking is weak
	    // 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值
	    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
	    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
	   // 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
	    const bool c2 = ((mnMatchesInliers < nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

	    if((c1a||c1b||c1c)&&c2)
	    {
		// If the mapping accepts keyframes, insert keyframe.
		// Otherwise send a signal to interrupt BA
		if(bLocalMappingIdle)
		{
		    return true;
		}
		else
		{
		    mpLocalMapper->InterruptBA();
		    if(mSensor!=System::MONOCULAR)
		    {
			// 队列里不能阻塞太多关键帧
			// tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
			// 然后localmapper再逐个pop出来插入到mspKeyFrames
			if(mpLocalMapper->KeyframesInQueue()<3)
			    return true;
			else
			    return false;
		    }
		    else
			return false;
		}
	    }
	    else
		return false;
	}
	
/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints
 */
	void Tracking::CreateNewKeyFrame()
	{
	    if(!mpLocalMapper->SetNotStop(true))
		return;
	    // 关键帧 加入到地图 加入到 关键帧数据库
	    
// 步骤1：将当前帧构造成关键帧	    
	    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
	    
// 步骤2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
	    mpReferenceKF = pKF;
	    mCurrentFrame.mpReferenceKF = pKF;
	    
    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
// 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
	    if(mSensor != System::MONOCULAR)
	    {
	      // 根据Tcw计算mRcw、mtcw和mRwc、mOw
		mCurrentFrame.UpdatePoseMatrices();

		// We sort points by the measured depth by the stereo/RGBD sensor.
		// We create all those MapPoints whose depth < mThDepth.
		// If there are less than 100 close points we create the 100 closest.
		// 双目 / 深度
     // 步骤3.1：得到当前帧深度小于阈值的特征点
               // 创建新的MapPoint, depth < mThDepth
		vector<pair<float,int> > vDepthIdx;
		vDepthIdx.reserve(mCurrentFrame.N);
		for(int i=0; i<mCurrentFrame.N; i++)
		{
		    float z = mCurrentFrame.mvDepth[i];
		    if(z>0)
		    {
			vDepthIdx.push_back(make_pair(z,i));
		    }
		}

		if(!vDepthIdx.empty())
		{
	         // 步骤3.2：按照深度从小到大排序  
		    sort(vDepthIdx.begin(),vDepthIdx.end());
                 // 步骤3.3：将距离比较近的点包装成MapPoints
		    int nPoints = 0;
		    for(size_t j=0; j<vDepthIdx.size();j++)
		    {
			int i = vDepthIdx[j].second;

			bool bCreateNew = false;

			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
			if(!pMP)
			    bCreateNew = true;
			else if(pMP->Observations()<1)
			{
			    bCreateNew = true;
			    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
			}

			if(bCreateNew)
			{
			    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
			    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
			    // 这些添加属性的操作是每次创建MapPoint后都要做的
			    pNewMP->AddObservation(pKF,i);
			    pKF->AddMapPoint(pNewMP,i);
			    pNewMP->ComputeDistinctiveDescriptors();
			    pNewMP->UpdateNormalAndDepth();
			    mpMap->AddMapPoint(pNewMP);

			    mCurrentFrame.mvpMapPoints[i]=pNewMP;
			    nPoints++;
			}
			else
			{
			    nPoints++;
			}
                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
			if(vDepthIdx[j].first>mThDepth && nPoints>100)
			    break;
		    }
		}
	    }

	    mpLocalMapper->InsertKeyFrame(pKF);

	    mpLocalMapper->SetNotStop(false);


    // 为点云建图线程，加入关键帧和对应的彩色图和深度图 insert Key Frame into point cloud viewer
    mpPointCloudMapping->insertKeyFrame( pKF, this->mImGray, this->mImDepth, this->mImRGB );

	    mnLastKeyFrameId = mCurrentFrame.mnId;
	    mpLastKeyFrame = pKF;
	}
	
/**
 * @brief 对Local MapPoints进行跟踪
 * 搜索 在对应当前帧的局部地图内搜寻和 当前帧地图点匹配点的 局部地图点
 * 局部地图点 搜寻和当前帧 关键点描述子 的匹配 有匹配的加入到 当前帧 特征点对应的地图点中
 * 
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
	void Tracking::SearchLocalPoints()
	{
	    // Do not search map points already matched
// 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
           // 因为当前的mvpMapPoints一定在当前帧的视野中
	    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit !=vend; vit++)
	    {
		MapPoint* pMP = *vit;// 当前帧的地图点
		if(pMP)
		{
		    if(pMP->isBad())
		    {
			*vit = static_cast<MapPoint*>(NULL);
		    }
		    else
		    {
			pMP->IncreaseVisible(); // 更新能观测到该点的帧数加1
			pMP->mnLastFrameSeen = mCurrentFrame.mnId;// 标记该点被当前帧观测到
			pMP->mbTrackInView = false;// 标记该点将来不被投影，因为已经匹配过
		    }
		}
	    }

	    int nToMatch=0;

	    // Project points in frame and check its visibility
// 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配    
	    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
	    {
		MapPoint* pMP = *vit;// 局部地图的 每一个地图点   
		// 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
		if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
		    continue;
		if(pMP->isBad())
		    continue;
	// 步骤2.1：判断LocalMapPoints中的点是否在在视野内
		// Project (this fills MapPoint variables for matching)
		if(mCurrentFrame.isInFrustum(pMP,0.5))
		{// 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
		    pMP->IncreaseVisible();
		    // 只有在视野范围内的MapPoints才参与之后的投影匹配
		    nToMatch++;
		}
	    }

	    if(nToMatch>0)
	    {
		ORBmatcher matcher(0.8);// 0.8  最短的距离 和 次短的距离 比值差异
		int th = 1;
		if(mSensor==System::RGBD)
		    th=3;
		// If the camera has been relocalised recently, perform a coarser search
		// 刚刚 进行过 重定位
		 // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
		if(mCurrentFrame.mnId < mnLastRelocFrameId+2)
		    th=5;
		// 步骤2.2：对视野范围内的MapPoints通过投影进行特征点匹配
		// 在局部地图点中搜寻 和 当前帧特征点描述子 匹配的地图点  加入到 当前帧 特征点对应的地图点中
		matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
	    }
	}
	


// 重定位
/*
 重定位Relocalization的过程大概是这样的：
1. 计算当前帧的BoW映射；
2. 在关键帧数据库中找到相似的候选关键帧；
3. 通过BoW匹配当前帧和每一个候选关键帧，如果匹配数足够 >15，进行EPnP求解；
    
4. 对求解结果使用BA优化，如果内点较少，则反投影候选关键帧的地图点 到当前帧 获取额外的匹配点
     根据特征所属格子和金字塔层级重新建立候选匹配，选取最优匹配；
     若这样依然不够，放弃该候选关键帧，若足够，则将通过反投影获取的额外地图点加入，再进行优化。
5. 如果内点满足要求(>50)则成功重定位，将最新重定位的id更新：mnLastRelocFrameId = mCurrentFrame.mnId;　　否则返回false。
 */
/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： \n
 * - K1个关键帧、K2个临近关键帧和参考关键帧
 * - 由这些关键帧观测到的MapPoints
 */
	bool Tracking::Relocalization()
	{
  // 1. 计算当前帧的BoW映射； Compute Bag of Words Vector
	    // 词典 N个M维的单词
	    // 一帧的描述子  n个M维的描述子
	    // 生成一个 N*1的向量 记录一帧的描述子 使用词典单词的情况
	    mCurrentFrame.ComputeBoW();

	    // Relocalization is performed when tracking is lost
	    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
 // 2. 在关键帧数据库中找到相似的候选关键帧；
           // 计算帧描述子 词典单词线性 表示的 词典单词向量
           // 和 关键帧数据库中 每个关键帧的线性表示向量 求距离 距离最近的一些帧 为 候选关键帧  
	    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
	    if(vpCandidateKFs.empty())
		return false;
	    const int nKFs = vpCandidateKFs.size();// 总的候选关键帧

	    // We perform first an ORB matching with each candidate
	    // If enough matches are found we setup a PnP solver
	    ORBmatcher matcher(0.75,true);// 描述子匹配器   最小距离 < 0.75*次短距离
	    vector<PnPsolver*> vpPnPsolvers;//两关键帧之间的匹配点  Rt 求解器
	    vpPnPsolvers.resize(nKFs);// 当前帧 和 每个候选关键帧 都有一个 求解器
	    vector<vector<MapPoint*> > vvpMapPointMatches;
	    // 当前帧 的关键点描述子 和 每个候选关键帧地图点 描述子的匹配点
	    
	    vvpMapPointMatches.resize(nKFs);//两个关键帧之间的 地图点匹配
	    vector<bool> vbDiscarded;// 候选关键帧与当前帧匹配 好坏 标志
	    vbDiscarded.resize(nKFs);

	    int nCandidates=0;

	    for(int i=0; i<nKFs; i++)// 关键帧数据库中 每一个候选 关键帧
	    {
	      
		KeyFrame* pKF = vpCandidateKFs[i];// 每一个候选 关键帧
		if(pKF->isBad())
		    vbDiscarded[i] = true;//  坏
		
		else
		{
// 3. 通过BoW匹配当前帧和每一个候选关键帧，如果匹配数足够 >15，进行EPnP求解；	  
		    int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
		    if(nmatches<15)
		    {
			vbDiscarded[i] = true;// 匹配效果不好
			continue;
		    }
		    else // 匹配数足够 >15   加入求解器
		    {
		      // 生成求解器
			PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
			pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);// 随机采样
			vpPnPsolvers[i] = pSolver;// 添加求解器
			nCandidates++;
		    }
		}
	    }
	    // Alternatively perform some iterations of P4P RANSAC
	    // Until we found a camera pose supported by enough inliers
      // 直到找到一个 候选匹配关键帧 和 符合 变换关系Rt 的足够的 内点数量
	    bool bMatch = false;
	    ORBmatcher matcher2(0.9,true);

	    while(nCandidates>0 && !bMatch)
	    {
		for(int i=0; i<nKFs; i++)// 
		{
		    if(vbDiscarded[i])// 跳过匹配效果差的 候选关键帧
			continue;

		    // Perform 5 Ransac Iterations   5次 随机采样序列 求解位姿  Tcw 
		    vector<bool> vbInliers;// 符合变换的 内点个数
		    int nInliers;
		    bool bNoMore;
           //求解器求解 进行EPnP求解
		    PnPsolver* pSolver = vpPnPsolvers[i];
		    cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);//迭代5次 得到变换矩阵

		    // If Ransac reachs max. iterations discard keyframe
		    if(bNoMore)//迭代5次效果还不好
		    {
			vbDiscarded[i]=true;// EPnP求解 不好   匹配效果差  放弃该  候选 关键帧
			nCandidates--;
		    }
// 4. 对求解结果使用BA优化，如果内点较少，则反投影当前帧的地图点到候选关键帧获取额外的匹配点；
// 若这样依然不够，放弃该候选关键帧，若足够，则将通过反投影获取的额外地图点加入，再进行优化。
		    // If a Camera Pose is computed, optimize
		    if(!Tcw.empty())
		    {
			Tcw.copyTo(mCurrentFrame.mTcw);

			set<MapPoint*> sFound;// 地图点

			const int np = vbInliers.size();// 符合 位姿  Tcw  的 内点数量

			for(int j=0; j<np; j++)
			{
			    if(vbInliers[j])// 每一个符和的 内点
			    {
				mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];// 对应的地图点 i帧  j帧下的地图点
				sFound.insert(vvpMapPointMatches[i][j]);
			    }
			    else
				mCurrentFrame.mvpMapPoints[j]=NULL;
			}
		      // 使用BA优化   位姿  返回优化较好效果较好的  3d-2d优化边
			int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

			if(nGood<10)
			    continue;// 这一候选帧 匹配优化后效果不好

			for(int io =0; io<mCurrentFrame.N; io++)
			    if(mCurrentFrame.mvbOutlier[io])// 优化后更新状态 外点
				mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);// 地图点为空指针

      // If few inliers, search by projection in a coarse window and optimize again
      // 如果内点较少，则反投影候选关键帧的地图点vpCandidateKFs[i] 到 当前帧像素坐标系下
      //  根据格子和金字塔层级信息 在 当前帧下 选择与地图点匹配的特征点
      // 获取额外的匹配点
			if(nGood<50)
			{
			    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

			    if(nadditional+nGood>=50)
			    {
			      // 当前帧 特征点对应的 地图点 数大于50 进行优化
				nGood = Optimizer::PoseOptimization(&mCurrentFrame);// 返回内点数量

				// If many inliers but still not enough, search by projection again in a narrower window
				// the camera has been already optimized with many points
				if(nGood>30 && nGood<50)
				{
				    sFound.clear();
				    for(int ip =0; ip<mCurrentFrame.N; ip++)
					if(mCurrentFrame.mvpMapPoints[ip])
					    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
				   // 缩小搜索窗口
				    nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

				    // Final optimization
				    if(nGood+nadditional>=50)
				    {
					nGood = Optimizer::PoseOptimization(&mCurrentFrame);

					for(int io =0; io<mCurrentFrame.N; io++)
					    if(mCurrentFrame.mvbOutlier[io])//外点
						mCurrentFrame.mvpMapPoints[io]=NULL;// 空指针
				    }
				}
			    }
			}


			// If the pose is supported by enough inliers stop ransacs and continue
			if(nGood>=50)
			{
			    bMatch = true;
			    break;
			}
		    }
		}
	    }

	    if(!bMatch)
	    {
		return false;
	    }
	    else
	    {
		mnLastRelocFrameId = mCurrentFrame.mnId;// 重定位 帧ID
		return true;
	    }

	}

// 跟踪重置
	void Tracking::Reset()
	{

	    cout << "系统重置 System Reseting" << endl;
	    if(mpViewer)
	    {
		mpViewer->RequestStop();
		while(!mpViewer->isStopped())
		    usleep(3000);
	    }

	    // Reset Local Mapping
	    cout << "重置局部建图 Reseting Local Mapper...";
	    mpLocalMapper->RequestReset();
	    cout << " done" << endl;

	    // Reset Loop Closing
	    cout << "重置回环检测 Reseting Loop Closing...";
	    mpLoopClosing->RequestReset();
	    cout << " done" << endl;

	    // Clear BoW Database
	    cout << "重置数据库 Reseting Database...";
	    mpKeyFrameDB->clear();
	    cout << " done" << endl;

	    // Clear Map (this erase MapPoints and KeyFrames)
	    mpMap->clear();

	    KeyFrame::nNextId = 0;
	    Frame::nNextId = 0;
	    mState = NO_IMAGES_YET;

	    if(mpInitializer)
	    {
		delete mpInitializer;
		mpInitializer = static_cast<Initializer*>(NULL);
	    }

	    mlRelativeFramePoses.clear();
	    mlpReferences.clear();
	    mlFrameTimes.clear();
	    mlbLost.clear();

	    if(mpViewer)
		mpViewer->Release();
	}

// 重新读取 配置文件
// 相机内参数
// 畸变校正参数
// 基线长度 × 焦距
	void Tracking::ChangeCalibration(const string &strSettingPath)
	{
	    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
	    float fx = fSettings["Camera.fx"];
	    float fy = fSettings["Camera.fy"];
	    float cx = fSettings["Camera.cx"];
	    float cy = fSettings["Camera.cy"];

	    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
	    K.at<float>(0,0) = fx;
	    K.at<float>(1,1) = fy;
	    K.at<float>(0,2) = cx;
	    K.at<float>(1,2) = cy;
	    K.copyTo(mK);

	    cv::Mat DistCoef(4,1,CV_32F);
	    DistCoef.at<float>(0) = fSettings["Camera.k1"];
	    DistCoef.at<float>(1) = fSettings["Camera.k2"];
	    DistCoef.at<float>(2) = fSettings["Camera.p1"];
	    DistCoef.at<float>(3) = fSettings["Camera.p2"];
	    const float k3 = fSettings["Camera.k3"];
	    if(k3!=0)
	    {
		DistCoef.resize(5);
		DistCoef.at<float>(4) = k3;
	    }
	    DistCoef.copyTo(mDistCoef);

	    mbf = fSettings["Camera.bf"];

	    Frame::mbInitialComputations = true;
	}
	
// 跟踪 + 建图 模式
	void Tracking::InformOnlyTracking(const bool &flag)
	{
	    mbOnlyTracking = flag;
	}



} //namespace ORB_SLAM
