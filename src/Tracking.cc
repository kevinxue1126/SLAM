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
                // B. More lost, relocation mode
		    else
		    {
			bOK = Relocalization();//Relocation BOW search, PnP 3d-2d matching Solve pose
		    }
		}
            // 2. If there is already a map, perform tracking + relocation (relocation after tracking is lost)================================================		
		else
		{
	        // A.track lost======================================================================================
		    if(mState==LOST)
		    {
			bOK = Relocalization();//Relocation BOW search, PnP 3d-2d matching Solve pose
		    }
                // B. normal tracking
		    else
		    {
		        // mbVO is a variable that only exists when mbOnlyTracking is true
                        // If mbVO is 0, it means that this frame matches a lot of MapPoints, and the tracking is normal.
                        // mbVO of 1 indicates that this frame matches very few MapPoints, less than 10, the rhythm to kneel     
			if(!mbVO)
			{
		   	   // a. There are enough points tracked in the previous frame
                           // 1. Move tracking mode, if that fails, try to use tracking reference frame mode
			    if(!mVelocity.empty())// on the move
			    {
				bOK = TrackWithMotionModel();// Track the previous frame at a constant speed Model
				 if(!bOK)// Newly added, if motion tracking mode fails, try to track with Tracking Reference Frame Mode
				    bOK = TrackReferenceKeyFrame();
			    }
                           // 2. Use Tracking Reference Frame Mode
			    else//not moved
			    {
				bOK = TrackReferenceKeyFrame();// track reference frame
			    }
			}
	         // b. There are few points tracked in the previous frame (to the textureless area, etc.), the rhythm to kneel, both tracking and positioning========   	
			else// If mbVO is 1, it means that this frame matches very few 3D map points, less than 10. The rhythm of kneeling is required to do both motion tracking and positioning
			{
                            // Use motion tracking and relocation mode to calculate two poses, if the relocation is successful, use the pose obtained from the relocation
			    bool bOKMM = false;
			    bool bOKReloc = false;
			    vector<MapPoint*> vpMPsMM;
			    vector<bool> vbOutMM;
			    cv::Mat TcwMM;// Pose results tracked by visual odometry
			    if(!mVelocity.empty())// With speed motion tracking mode
			    {
				bOKMM = TrackWithMotionModel();// Motion Tracking Mode Tracks Previous Frame Results
				vpMPsMM = mCurrentFrame.mvpMapPoints;// map point
				vbOutMM = mCurrentFrame.mvbOutlier;  // outer point
				TcwMM = mCurrentFrame.mTcw.clone();  // Save visual odometry pose results
			    }
			    
			    bOKReloc = Relocalization();// relocation mode
                         // 1.Relocation was unsuccessful, but motion tracking was successful, using tracking results===================================
			    if(bOKMM && !bOKReloc)
			    {
				mCurrentFrame.SetPose(TcwMM);// set frame position to visual odometry pose result
				mCurrentFrame.mvpMapPoints = vpMPsMM;// frame seen map point
				mCurrentFrame.mvbOutlier = vbOutMM;// outer point

				if(mbVO)
				{
				  // Update the observed degree of MapPoints of the current frame
				    for(int i =0; i<mCurrentFrame.N; i++)
				    {
					if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
					{
					    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
					}
				    }
				}
			    }
		         // 2. relocation mode success=================================================
			    else if(bOKReloc)// As long as the relocation is successful, the entire tracking process will proceed normally (location and tracking, more believe in relocation)
			    {
				mbVO = false;//Relocation succeeded
			    }

			    bOK = bOKReloc || bOKMM;// Motion Tracking/Relocation Success Sign
			}
		    }
		}

      	      // Step 2. Local map tracking	
	      // Through the previous calculations, an initial estimate of the pose has been obtained, 
              // and we can find more correspondences from the map points that have been generated through projection to get accurate results
	      // Following the initial tracking of the three modes, local map tracking is performed
	      // The descriptor of the local map point and the feature point of the current frame (the key point that has not yet been matched to the map point) perform descriptor matching
	      // Graph optimization to optimize using the pixel coordinates of the feature points of the current frame and the matching 3D map points to optimize on its original pose
	      // After matching optimization, the number of successful point pairs is generally greater than 30 and considered successful
	      // In the case of just relocation, it needs to be greater than 50 to be considered successful

		mCurrentFrame.mpReferenceKF = mpReferenceKF;// reference keyframe
	         // Step 2.1: After the initial pose is obtained by matching between frames, now track the local map to obtain more matches and optimize the current pose
		// local map: the current frame, the MapPoints of the current frame, the co-viewing relationship between the current keyframe and other keyframes
		// Tracking in the above two or two frames (the constant speed model tracks the previous frame, tracks the reference frame), 
		// here searches for the local key frame and collects all the local MapPoints, 
		// and then performs projection matching between the local MapPoints and the current frame to get more matching MapPoints. Pose optimization
                // There is a mapping thread
		if(!mbOnlyTracking)// Tracking + Mapping + Relocation
		{
		    if(bOK)
			bOK = TrackLocalMap(); // Local map tracking g20 optimization ------------------
		}
                // Local map tracking g20 optimization
		else// Track + Relocate
		{
		    // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
		    // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
		    // the camera we will use the local map again.
		    if(bOK && !mbVO)// Relocation succeeded
			bOK = TrackLocalMap();// local map tracking--------------------------
		}

		if(bOK)
		    mState = OK;
		else
		    mState=LOST;// lost

		// Update drawer
		mpFrameDrawer->Update(this);


                // Step 2.2 Local map tracking is successful, root motion model, clear outliers, etc., check whether new keyframes need to be created
		if(bOK)
		{
	       // a. If there is motion, update the motion model Update motion model The motion speed is the transformation matrix of the two needles before and after
		    if(!mLastFrame.mTcw.empty())
		    {      
			cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
			mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
			mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
			mVelocity = mCurrentFrame.mTcw*LastTwc;//The movement speed is the transformation matrix of the front and rear needles
		    }
		    else
			mVelocity = cv::Mat();//no speed
                    // Display the current camera pose
		    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

               // b. Clear the MapPoints temporarily added for the current frame in Update LastFrame    
                    // If the number of observation frames of the map point of the current frame is less than 1, the corresponding map point will be cleared.
		    for(int i=0; i< mCurrentFrame.N; i++)
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];//The current frame matches the map point
			if(pMP)//pointer exists
			    if(pMP->Observations()<1)// Its observation frame is less than 1
			    {// Exclude MapPoints added for tracking in UpdateLastFrame function
				mCurrentFrame.mvbOutlier[i] = false;// Outer Point Flag 0
				mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);//Clear the corresponding map point
			    }
		    }
               // c. Clear temporary MapPoints, delete temporary map points
                  // These MapPoints are generated in the UpdateLastFrame function of TrackWithMotionModel (only binocular and rgbd)
		  // In b, these MapPoints are only culled in the current frame, here they are deleted from the MapPoints database
		  // What is generated here is only to improve the frame-to-frame tracking effect of the binocular or rgbd camera. It will be thrown away after use and not added to the map.
		    //  list<MapPoint*>::iterator 
		    for(auto lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit != lend; lit++)
		    {
			MapPoint* pMP = *lit;
			delete pMP;// Delete the space corresponding to the map point
		    }
		    mlpTemporalPoints.clear();

	      // d. Determine whether a new keyframe needs to be created
		   // The last step is to determine whether to set the current frame as a keyframe. 
	          // Since redundant keyframes will be eliminated in Local Mapping, we need to insert new keyframes as soon as possible, so as to be more robust.
		    if(NeedNewKeyFrame())
			CreateNewKeyFrame();

	      	    // e. Outlier Clear Check Outlier Flags (updated when points that do not conform to transformation matrix are optimized) 
		    for(int i=0; i<mCurrentFrame.N;i++)
		    {
			if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
			    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
		    }
		}
      		// 3. Processing after tracking failure (two-by-two tracking failure or partial map tracking failure)
		if(mState == LOST)
		{
		    if(mpMap->KeyFramesInMap()<=5)// The number of keyframes is too small (just starting to build a map) Exit directly
		    {
			cout << "Track lost soon after initialisation, reseting..." << endl;
			mpSystem->Reset();
			return;
		    }
		}

		if(!mCurrentFrame.mpReferenceKF)
		    mCurrentFrame.mpReferenceKF = mpReferenceKF;

		mLastFrame = Frame(mCurrentFrame);//New keyframe
	    }

   
	    // Step 3: Return the tracked pose information=======================================================================
            // Calculate the transformation from the reference frame to the current frame Tcr = mTcw  * mTwr 
	    if(!mCurrentFrame.mTcw.empty())
	    {
	        // mTcw  * mTwr  = mTcr
		cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
		mlRelativeFramePoses.push_back(Tcr);
		mlpReferences.push_back(mpReferenceKF);
		mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
		mlbLost.push_back(mState==LOST);
	    }
	    else//Loss of tracking results in empty pose
	    {
		// This can happen if tracking is lost
		mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
		mlpReferences.push_back(mlpReferences.back());
		mlFrameTimes.push_back(mlFrameTimes.back());
		mlbLost.push_back(mState==LOST);
	    }

  }
	



	/**
	 * @brief Map initialization for binocular and rgbd
	 *
	 * Directly generate MapPoints due to depth information
	 */
	// First frame binocular/depth initialization
	void Tracking::StereoInitialization()
	{
	    if(mCurrentFrame.N>500)
  	    // [0] When the number of key points found is greater than 500, initialize and build the current frame as the first key frame
	    {
		// Set Frame pose to the origin
       		//【1】 Initialization The first frame is the origin of the world coordinate system Transformation matrix Diagonal unit matrix R = eye(3,3) t=zero(3,1)
		// Step 1: Set the initial pose
		mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

       		// [2] Create the first frame as a key frame Create KeyFrame normal frame map key frame database
		// Add map Add keyframe database
		// Step 2: Construct the current frame as the initial keyframe
		// The data type of mCurrentFrame is Frame
		// KeyFrame contains Frame, map 3D points, and BoW
		// There is an mpMap in KeyFrame, an mpMap in Tracking, and the mpMap in KeyFrame all point to this mpMap in Tracking
		// There is an mpKeyFrameDB in KeyFrame, an mpKeyFrameDB in Tracking, and mpMap in KeyFrame all point to this mpKeyFrameDB in Tracking
		KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
		// nsert KeyFrame in the map
		// The KeyFrame contains the map, and in turn the map also contains the KeyFrame, which contains each other
		// Step 3: Add this initial keyframe to the map
		mpMap->AddKeyFrame(pKFini);// Add keyframes to the map

		// Create MapPoints and asscoiate to KeyFrame
                // [3] Create a map point and associate it with the corresponding keyframe. The keyframe also adds a map point. The map adds a map point. The map point descriptor distance
		// Step 4: Construct MapPoint for each feature point	
		for(int i=0; i<mCurrentFrame.N;i++)// each key point of the frame
		{
		    float z = mCurrentFrame.mvDepth[i];// Depth values corresponding to keypoints Binocular and depth cameras have depth values
		    if(z>0)// effective depth 
		    {
		   	// Step 4.1: Obtain the 3D coordinates of the feature point through back projection
			cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);// Projected to 3D point coordinates in the world coordinate system
		   	// Step 4.2: Construct 3D Points as MapPoints	
			// Each 3d point corresponding to a keypoint with a valid depth is converted to a map point object
			MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
		  	// Step 4.3: Add properties to this MapPoint:
			// a. Observe the key frame of the MapPoint
			// b. Descriptor of the MapPoint
			// c. The average observation direction and depth range of the MapPoint
			
                         // a. Indicates which feature point of which KeyFrame the MapPoint can be observed by
			pNewMP->AddObservation(pKFini,i);// Add map point Observation Reference frame This map point can be observed on this frame
			 // b. Select the descriptor with the highest discriminating read from the feature points that observe the MapPoint
			pNewMP->ComputeDistinctiveDescriptors();// Map point to calculate the best descriptor
			// c. Update the average observation direction of the MapPoint and the range of the observation distance
			// The average observation direction and the range of the observation distance of the map point are all preparations for the subsequent descriptor fusion.
			pNewMP->UpdateNormalAndDepth();
			// update relative frame camera center normalized relative coordinates pyramid level distance from camera center
		   	// Step 4.4: Add the MapPoint to the map
			mpMap->AddMapPoint(pNewMP);// map add map point
                   	// Step 4.5: Indicate which feature point of the KeyFrame can observe which 3D point
			 pKFini->AddMapPoint(pNewMP,i);
		   	// Step 4.6: Add this MapPoint to mvpMapPoints of the current frame
                        // Create an index between the feature points of the current Frame and MapPoint
			mCurrentFrame.mvpMapPoints[i]=pNewMP;//current frame add map point
		    }
		}
		cout << "new map , : " << mpMap->MapPointsInMap() << " points" << endl;
 		// Step 5: Add this initial keyframe to the local map
		// [4] Add keyframes to local mapping Add keyframes to local keyframes Add all map points to local map points
		mpLocalMapper->InsertKeyFrame(pKFini);
               // Record
		mLastFrame = Frame(mCurrentFrame);// previous normal frame
		mnLastKeyFrameId=mCurrentFrame.mnId;// id
	 	mpLastKeyFrame = pKFini;// previous keyframe
               // local
		mvpLocalKeyFrames.push_back(pKFini);// local keyframes add keyframes
		mvpLocalMapPoints=mpMap->GetAllMapPoints();//Local map points Add all map points
		mpReferenceKF = pKFini;// reference frame
		mCurrentFrame.mpReferenceKF = pKFini;//current frame reference key frame
                // map
		mpMap->SetReferenceMapPoints(mvpLocalMapPoints);//map reference map point
		mpMap->mvpKeyFrameOrigins.push_back(pKFini);// map keyframe
                // visualization
		mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
		mState=OK;// tracking is normal
	    }
	}


	/**
	 * @brief Monocular Initialization First Frame Monocular Initialization
	 * There is a special initializer for single-purpose initialization, and the initializer can be successfully constructed only if there are more than 100 feature points in two consecutive frames.
	 * Calculate the fundamental matrix and the homography matrix in parallel, select one of the models, 
	 * recover the relative pose and point cloud between the first two frames, and obtain the matching, relative motion, and initial MapPoints of the initial two frames.
	 */
	void Tracking::MonocularInitialization()
	{
	    // 【1】Add the first frame and set the reference frame
	    if(!mpInitializer)// Not initialized successfully Initialize Initializer to get R t and 3D points
	    {
		// Set Reference Frame
		if(mCurrentFrame.mvKeys.size()>100)// Initialize only when the number of key points exceeds 100
		{
		    mInitialFrame = Frame(mCurrentFrame);// initial frame
		    mLastFrame = Frame(mCurrentFrame);// previous frame
		    mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());// are all feature points in the first frame
		    for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
			mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;//Match point abscissa
			
		   if(mpInitializer)
			delete mpInitializer;
		    
                    // initialize again
		    mpInitializer =  new Initializer(mCurrentFrame,1.0,200);// variance and iterations
		    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
		    return;
		}
	    }
	    // 【2】Add the second frame After the reference frame setting is completed, select whether to initialize according to the number of key points in the current frame
	    else// The first frame is initialized successfully. The current frame and the reference frame are matched to get R t
	    {
		// Try to initialize
     		//【3】Reinitialize Set the reference frame    
		if((int)mCurrentFrame.mvKeys.size()<=100)//The initializer can be successfully constructed only if there are more than 100 feature points in two consecutive frames
		{
		    delete mpInitializer;
		    mpInitializer = static_cast<Initializer*>(NULL);// reinitialize
		    fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
		    return;
		}
    		// [4] There are many feature points in the current frame, and the reference frame is used to find matching point pairs. Determine whether to initialize according to the number of matching point pairs.
		//  Find matching point pairs mvIniMatches
		ORBmatcher matcher(0.9,true);
		// mInitialFrame first frame  mCurrentFrame current frame second frame
		// mvbPreMatched is all feature points in the first frame;
		// mvIniMatches marks the matching state, and the unmatched mark is -1;
		// If nmatches<100 is returned, the initialization fails, and the initialization process is re-initialized
		// 100 block matches Search window size
		int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
     		//【5】 Check if there are enough correspondences
		if(nmatches<100)
		{
		    delete mpInitializer;
		    mpInitializer = static_cast<Initializer*>(NULL);
		    return;
		}
               
        	// [6] Initialize a large number of matching point pairs Calculate the moving pose of the camera Calculate the initial R t according to the fundamental matrix F or the homography matrix H
		cv::Mat Rcw; // Current Camera Rotation
		cv::Mat tcw; // Current Camera Translation
		vector<bool> vbTriangulated; // Points that conform to the interior points of the transformation matrix and that have normal 3D coordinates after triangulation
		// Triangulated Correspondences (mvIniMatches)	
 		// * Monocular camera initialization
		//* Homography matrix H for planar scenes (8 motion hypotheses) and fundamental matrix F for non-planar scenes (4 motion hypotheses)
		//* Then select a suitable model through a scoring rule, restore the camera's rotation matrix R and translation vector t and the corresponding 3D point (scale problem) Good or bad point mark
  		// Parallel computing decomposes the fundamental matrix and the homography matrix (the acquired points are exactly in the same plane) to obtain the motion (pose) between frames, and vbTriangulated marks whether a set of feature points can be triangulated.
  		// mvIniP3D is a container of type cv::Point3f, which is a temporary variable that stores the 3D points obtained by triangulation.
 
		if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
	     	// The most critical algorithm is to recover the camera pose and map points through the epipolar constraints of the initial two consecutive frames
		{
		    for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
		    {
			if(mvIniMatches[i]>=0 && !vbTriangulated[i])// is the matching point but the matching point is not on the transformation being calculated
			{
			    mvIniMatches[i]=-1;//This match point is not good
			    nmatches--;//Match point pairs - 1
			}
		    }

	 	    // [7] Set the world coordinate position and posture of the initial reference frame Diagonal matrix
		    mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
		    // 【8】Set the pose of the second frame (current frame)    
		    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
		    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
		    tcw.copyTo(Tcw.rowRange(0,3).col(3));		    
		    mCurrentFrame.SetPose(Tcw);
        	    // [9] Create a map and use Minimize reprojection error BA for map optimization to optimize pose and map points
		    CreateInitialMapMonocular();
		}
	    }
	}


	 /**
	 * @brief CreateInitialMapMonocular
	 * The initial frame is set as the origin of the world coordinate system. 
	 * After initialization, the solved current frame pose T Minimizes the reprojection error BA Globally optimizes the pose T generates MapPoints for the triangulation of the monocular camera
	 */
	void Tracking::CreateInitialMapMonocular()
	{
  	    //【1】Create KeyFrames
            // The construction of the initial map is to add these two key frames and the corresponding map points to the map (mpMap), and it is necessary to construct the key frames and map points respectively.
	    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);// Initial keyframe Add map Add keyframe database
	    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);// current keyframe second frame
  	    //[2] Calculate the linear representation vector of the frame descriptor in the descriptor dictionary
	    pKFini->ComputeBoW();
	    pKFcur->ComputeBoW();

   	    // 【3】Insert KFs in the map
	    mpMap->AddKeyFrame(pKFini);
	    mpMap->AddKeyFrame(pKFcur);

            // 【4】Create MapPoints and asscoiate to keyframes
	   // Some attributes need to be added to the map point:
	  //1. Observe the key frame of the map point (corresponding key point);
	  //2. The descriptor of the MapPoint;
	  //3. The average observation direction and observation distance of the MapPoint.
	    for(size_t i=0; i<mvIniMatches.size();i++)
	    {
		if(mvIniMatches[i]<0)// bad match don't
		    continue;

		// 【5】Create MapPoint.
		cv::Mat worldPos(mvIniP3D[i]);// Convert the 3D point vector 3d obtained by mvIniP3D triangulation into mat 3d
		MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

		pKFini->AddMapPoint(pMP,i);// Initial frame Add map point
		pKFcur->AddMapPoint(pMP,mvIniMatches[i]);// current frame add map point
		
        	// 【6】Add an observation frame to a map point The map point can be observed in both the reference frame and the current frame
		pMP->AddObservation(pKFini,i);
		pMP->AddObservation(pKFcur,mvIniMatches[i]);
		
        	// [7] Update some new parameters of map points, descriptor observation direction observation distance
		pMP->ComputeDistinctiveDescriptors();// The most representative descriptor of the map point on all observation frames
		pMP->UpdateNormalAndDepth();// The average viewing direction and viewing distance for this MapPoint.
		// Update map points relative to each observation frame camera center normalized coordinates
	       // Update the minimum and maximum distances of map points under each pyramid level under the reference frame
        	// 【8】The current frame is associated with the map point
		//Fill Current Frame structure
		mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
		mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;// is a good point outlier sign
		// 【9】Add map point
		mpMap->AddMapPoint(pMP);
	    }

	    // Update Connections
     	    // [10] The number of times the connection relationship with the new keyframe is observed
	   // It is also necessary to update the connection relationship between keyframes (with the number of common view map points as the weight):
	    pKFini->UpdateConnections();
	    pKFcur->UpdateConnections();

	    // Bundle Adjustment
	    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
            // [11] Globally optimize map BA to minimize reprojection error
	    Optimizer::GlobalBundleAdjustemnt(mpMap,20);// Globally optimize the reprojection error (LM) for these two frame poses:
	      // Note that the global optimization is used here, and the same function is used as the large loopback optimization after the loop closure detection adjustment.
	    
        // 【12】Set median depth to 1
	    // Need to normalize the median of map point depths in the first frame;
	    float medianDepth = pKFini->ComputeSceneMedianDepth(2);//  Monocular Environment Depth Median
	    float invMedianDepth = 1.0f/medianDepth;
        // [13] Detection reset If the depth is < 0 or the map points tracked in the second frame after optimization is found to be < 100, it also needs to be re-initialized.
	    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100 )
	    {
		cout << "Wrong initialization, reseting..." << endl;
		Reset();
		return;
	    }
       // [14] keyframe pose translation scale normalization
         // Otherwise, take the median depth as unit one, and normalize the pose of the second frame with all map points.
	    // Scale initial baseline
	    cv::Mat Tc2w = pKFcur->GetPose();// pose
	    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;// Shift normalized scale
	    pKFcur->SetPose(Tc2w);//set new pose

        // 【15】 Scale points
	    // map point normalized scale
	    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
	    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
	    {
		if(vpAllMapPoints[iMP])
		{
		    MapPoint* pMP = vpAllMapPoints[iMP];
		    pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);//map point scale normalization
		}
	    }
   	    // 【16】 object update
            // Insert keyframes for local maps
	    mpLocalMapper->InsertKeyFrame(pKFini);
	    mpLocalMapper->InsertKeyFrame(pKFcur);
           // current frame update pose
	    mCurrentFrame.SetPose(pKFcur->GetPose());
	    mnLastKeyFrameId=mCurrentFrame.mnId;// current frame iterate to previous frame prepare for next iteration
	    mpLastKeyFrame = pKFcur;// pointer
           // Local keyframes Local map point updates
	    mvpLocalKeyFrames.push_back(pKFcur);
	    mvpLocalKeyFrames.push_back(pKFini);
	    mvpLocalMapPoints=mpMap->GetAllMapPoints();
	    mpReferenceKF = pKFcur;// reference keyframe
	    mCurrentFrame.mpReferenceKF = pKFcur;// reference frame of the current frame

	    mLastFrame = Frame(mCurrentFrame);// current frame iterate to previous frame prepare for next iteration
           // reference map point
	    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
           // map display
	    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
           // Map keyframe sequence
	    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
	    mState=OK;// status ok
	}
	
	/**
	 * @brief Check if MapPoints in previous frame are replaced
	 * Check Replace Keyframe Map Point
	 * The last frame map point Whether there is a replacement point, if there is a replacement point, replace it
	 * The Local Mapping thread may replace some MapPoints in the key frame. Since mLastFrame is needed in tracking, check and update the MapPoints that were replaced in the previous frame.
	 * @see LocalMapping::SearchInNeighbors()
	 */
	void Tracking::CheckReplacedInLastFrame()
	{
	    for(int i =0; i<mLastFrame.N; i++)
	    {
		MapPoint* pMP = mLastFrame.mvpMapPoints[i];

		if(pMP)
		{
		    MapPoint* pRep = pMP->GetReplaced();// There are replacement points
		    if(pRep)
		    {
			mLastFrame.mvpMapPoints[i] = pRep;// make a replacement
		    }
		}
	    }
	}

// track reference frame  The robot doesn't move much
// The feature point descriptor of the current frame is matched with the descriptor of the map point in the reference key frame frame
 // Retain keypoint matching map point matching point pair in the highest three bins in the orientation histogram
// using word-band vector matching
// Both the key frame and the current frame are represented linearly by dictionary words
// The descriptors of the corresponding words must be relatively similar. Matching the descriptors of the corresponding words can speed up the matching.
// Match with the map points of the reference keyframe. The number of matching point pairs needs to be greater than 15
// Use graph optimization to optimize the pose based on the initial pose according to the map points and the corresponding pixels of the frame
// Remove outliers at the same time
// If there are more than 10 matching points in the end, return true and the tracking is successful.
	/**
	 * @brief Tracking MapPoints referenced to keyframes
	 * 
	 * 1. Calculate the word bag of the current frame, and assign the feature points of the current frame to the nodes of a specific layer
	 * 2. Match descriptors belonging to the same node
	 * 3. Estimate the pose of the current frame based on the matching pair
	 * 4. Eliminate false matches based on pose
	 * @return Returns true if the number of matches is greater than 10
	 */
	bool Tracking::TrackReferenceKeyFrame()
	{ 
	    // Compute Bag of Words vector
	    mCurrentFrame.ComputeBoW();// All feature point descriptors in the current frame are linearly represented by dictionary words

	    // We perform first an ORB matching with the reference keyframe
	    // If enough matches are found we setup a PnP solver
	    ORBmatcher matcher(0.7,true);// orb feature matcher 0.7 robust matching coefficient
	    vector<MapPoint*> vpMapPointMatches;
	    
            // Calculate the feature matching between the current frame and the reference key frame and return the number of matching point pairs
	    // Map points in the current frame and reference keyframes perform feature matching and match to existing map points
	    // The descriptor of each key point of the current frame matches the descriptor of each map point of the reference key frame
	    // Keep the closest matching map point and the shortest distance is not much different from the next shortest distance ( mfNNratio )
	    // If you need to consider the orientation information of the key points
	    // Count the direction histogram of the direction of the key point of the current frame to 30 steps
	    // Retain keypoint matching map point matching point pair in the highest three bins in the orientation histogram
	    // Both the key frame and the current frame are represented linearly by dictionary words
            // The descriptors of the corresponding words must be relatively similar. Matching the descriptors of the corresponding words can speed up the matching.
	    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

	    if(nmatches<15)// Match with the reference key frame Match the number of point pairs Need to be greater than 15
		return false;

	    mCurrentFrame.mvpMapPoints = vpMapPointMatches;// map point
	    mCurrentFrame.SetPose(mLastFrame.mTcw);// The pose is initially the pose of the previous frame
	    Optimizer::PoseOptimization(&mCurrentFrame);// Optimize the pose and mark whether it conforms to the transformation matrix Rt The outliers that do not conform
	    // Use graph optimization to optimize the pose based on the initial pose according to the map points and the corresponding pixels of the frame

	    // Discard outliers
	    // Remove the matching map points corresponding to the outliers
	    int nmatchesMap = 0;
	    for(int i =0; i<mCurrentFrame.N; i++)//every key point
	    {
		if(mCurrentFrame.mvpMapPoints[i])// If there is a corresponding matched map point
		{
		    if(mCurrentFrame.mvbOutlier[i])//It is an outlier that needs to be deleted Outlier Points that do not conform to the transformation relationship Update during optimization
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

			mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);//remove matching points
			mCurrentFrame.mvbOutlier[i]=false;//No matching map point Outer point flag Set to no
			pMP->mbTrackInView = false;
			pMP->mnLastFrameSeen = mCurrentFrame.mnId;
			nmatches--;
		    }
		    else if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)//Is the interior point colleague has the observation key frame
			nmatchesMap++;
		}
	    }

	    return nmatchesMap >= 10;
	}

	// update previous frame
	// update previous frame pose = world to previous frame's reference frame to previous frame
	// update previous frame map point
	/**
	 * @brief The binocular or rgbd camera generates new MapPoints according to the depth value of the previous frame
	 *
	 * In the case of binocular and rgbd, select some points with less depth (more reliable) \n
	 * Some new MapPoints can be generated from the depth value
	 */
	void Tracking::UpdateLastFrame()
	{
	    // Update pose according to reference keyframe
	    KeyFrame* pRef = mLastFrame.mpReferenceKF;// reference frame
	    cv::Mat Tlr = mlRelativeFramePoses.back();//The transform of the previous frame's reference frame to the previous frame Tlr
	    mLastFrame.SetPose(Tlr*pRef->GetPose());//last frame pose = world to the reference frame of the previous frame to the previous frame

	    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
		return;

	    // Create "visual odometry" MapPoints
	    // We sort points according to their measured depth by the stereo/RGB-D sensor
	    // The following binocular/depth cameras perform
	    vector<pair<float,int> > vDepthIdx;
	    vDepthIdx.reserve(mLastFrame.N);
	    for(int i=0; i<mLastFrame.N;i++)
	    {
		float z = mLastFrame.mvDepth[i];// The depth corresponding to the key point
		if(z>0)
		{
		    vDepthIdx.push_back(make_pair(z,i));
		}
	    }

	    if(vDepthIdx.empty())
		return;

	    sort(vDepthIdx.begin(),vDepthIdx.end());//deep sorting

	    // We insert all close points (depth < mThDepth)
	    // If less than 100 close points, we insert the 100 closest ones.
	    int nPoints = 0;
	    for(size_t j=0; j<vDepthIdx.size();j++)
	    {
		int i = vDepthIdx[j].second;

		bool bCreateNew = false;

		MapPoint* pMP = mLastFrame.mvpMapPoints[i];// The map point corresponding to the previous frame
		if(!pMP)
		    bCreateNew = true;// regenerate flag
		else if(pMP->Observations()<1)// The number of observation frames corresponding to the map point 1
		{
		    bCreateNew = true;
		}

		if(bCreateNew)//Regenerate 3D points
		{
		    cv::Mat x3D = mLastFrame.UnprojectStereo(i);// Generate 3D points
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

	// Movement mode tracking Two frames before and after the movement to get the transformation matrix
	// The map points of the previous frame are back-projected to the pixel coordinates of the current frame image and the key points of the current frame are in the same grid for descriptor matching. 
	// Search can speed up matching.
	/*
	Using the pose estimated by the uniform model, project the temporary map point in the LastFrame to the current pose, 
	and match according to the descriptor distance near the projected point (> 20 pairs of matching are required, otherwise the uniform model tracking will fail, and this will occur when the motion changes too much. 
	If the remaining matches are still >=10 pairs, the tracking is successful, otherwise the tracking fails, and Relocalization is required:

	Tracking with motion model (Tracking with motion model) The tracking rate is faster. 
	Assume that the object is moving at a uniform speed. 
	Use the pose and speed of the previous frame to estimate the pose of the current frame. The function used is TrackWithMotionModel().
	The matching here is to match the map points seen in the previous frame by projection, using 
	matcher.SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, ...)。
	*/
	/**
	* @brief Track the MapPoints of the previous frame according to the uniform velocity model
	* 
	* 1. For non-monocular cases, some new MapPoints (temporary) need to be generated for the previous frame
	* 2. Project the MapPoints of the previous frame onto the image plane of the current frame, and perform region matching at the projected position
	* 3. Estimate the pose of the current frame based on the matching pair
	* 4. Eliminate false matches based on pose
	* @return Returns true if the number of matches is greater than 10
	* @see V-B Initial Pose Estimation From Previous Frame
	*/
	bool Tracking::TrackWithMotionModel()
	{
	  
	    ORBmatcher matcher(0.9,true);// Matching point matcher Minimum distance < 0.9* short distance Matching is successful

	    // Update last frame pose according to its reference keyframe
	    // Create "visual odometry" points if in Localization Mode
	    // Update the previous frame pose = the world to the previous frame's reference frame and then to the previous frame
            // Update the map point of the previous frame
	    UpdateLastFrame();

	    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
	    // The current frame pose mVelocity is the pose transformation of the current frame and the previous frame
            // initialize null pointer
	    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

	    // Project points seen in previous frame
	    int th;
	    if(mSensor  != System::STEREO)
		th=15;// search window
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
		    if(mCurrentFrame.mvbOutlier[i])//outer point
		    {
			MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];// The current frame feature point matches the map point

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

	
	// Following the initial tracking of the three modes, local map tracking is performed
	// The descriptor of the local map point and the feature point of the current frame (the key point that has not yet been matched to the map point) perform descriptor matching
	// Graph optimization to optimize using the pixel coordinates of the feature points of the current frame and the matching 3D map points to optimize on its original pose
	// After matching optimization, the number of successful point pairs is generally greater than 30 and considered successful
	// In the case of just relocation, it needs to be greater than 50 to be considered successful
	/*
	 The above two only complete the inter-frame tracking in the visual odometer, 
	 and also need to track the local map to improve the accuracy: (This is actually what the Local Mapping thread does)
	 Local map tracking is required in TrackLocalMap()
	 First update the local map (UpdateLocalMap),
	 And search the local map point (SearchLocalPoint).
	 The update of the local map is divided into
	 Update of local map points (UpdateLocalPoints) and 
	 local keyframes (UpdateLocalKeyFrames).

	 In order to reduce the complexity, here is just a projection on the local graph. 
	 The key frame sequence in the local map that has the same point as the current frame is called K1, 
	 and the key frame sequence adjacent to K1 in the covisibility graph is called K2. The local map has a reference keyframe Kref ∈ K1 which has the most co-seen map cloud points with the current frame. 
	 For each map cloud point visible to K1 and K2, search in the current frame through the following steps:

	(1) Project the map point onto the current frame, and discard it if it exceeds the image range;
	(2) Calculate the angle between the current sight direction vector v and the average sight direction vector n of the map point cloud, and discard the point cloud with n v < cos(60°);
	(3) Calculate the distance d from the map point to the center of the camera, and consider [dmin, dmax] to be an area with constant scale. If d is not in this area, it will be discarded;
	(4) Calculate the scale factor of the image, which is d/dmin;
	(5) Compare the feature descriptor D of the map point with the ORB feature that has not yet been matched, and find the best match according to the previous scale factor.
	  In this way, the camera pose can be finally optimized by matching all map points.

	 */
	/**
	 * @brief Track the MapPoints of the Local Map
	 * 
	 * 1. Update the local map, including local keyframes and keypoints
	 * 2. Projection matching for local MapPoints
	 * 3. Estimate the pose of the current frame based on the matching pair
	 * 4. Eliminate false matches based on pose
	 * @return true if success
	 * @see V-D track Local Map
	 */
	bool Tracking::TrackLocalMap()
	{
	    // We have an estimation of the camera pose and some map points tracked in the frame.
	    // We retrieve the local map and try to find matches to points in the local map.
	    // [1] First, update the local map (UpdateLocalMap) to generate a local map corresponding to the current frame
	     // Update local map (frames and map points relative to current frame) for tracking of local map points Keyframes + map points
	     // Update local keyframes—part of the local map Keyframes with high co-visualization Subkeyframes Parent keyframes
	     // The update of local map points is relatively easy. It is completely based on the local key frame, and all the map points of the local key frame constitute the local map point.
	    UpdateLocalMap();
	    // [2] And search for local map points (SearchLocalPoint)
	    // The local map point is searched for matching with the current frame key point descriptor, and the matched ones are added to the map points corresponding to the current frame feature points
	    SearchLocalPoints();

 	    // 【3】Optimize Pose
	    Optimizer::PoseOptimization(&mCurrentFrame);
	    // During optimization, the pose transformation relationship of the current frame will be updated, and the interior/exterior markers of map points will be updated at the same time.
	    mnMatchesInliers = 0;

 	    // 【4】 Update MapPoints Statistics
	    for(int i=0; i<mCurrentFrame.N; i++)
	    {
		if(mCurrentFrame.mvpMapPoints[i])//Feature point find map point
		{
		    if(!mCurrentFrame.mvbOutlier[i])//is the interior point conforming to the transformation relation
		    {
			mCurrentFrame.mvpMapPoints[i]->IncreaseFound();// Feature point find map point sign
			if(!mbOnlyTracking)
			{
			    if(mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
				mnMatchesInliers++;//
			}
			else
			    mnMatchesInliers++;
		    }
		    else if(mSensor == System::STEREO)// Outer point clears matching map points under binocular
			mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
		}
	    }

	    // Decide if the tracking was succesful
	    // More restrictive if there was a relocalization recently
	    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers<50)
		return false;//If the relocation has just been performed, the number of matching point pairs is greater than 50 to be considered successful.

	    if(mnMatchesInliers<30)//Under normal circumstances, the number of matching point pairs found is greater than 30, which is considered successful
		return false;
	    else
		return true;
   }
	
	// Update local map (frames and map points relative to current frame) for tracking of local map points Keyframes + map points
	// Update local keyframes—part of the local map Keyframes with high co-visualization Subkeyframes Parent keyframes
	// The update of local map points is relatively easy. It is completely based on the local key frame, and all the map points of the local key frame constitute the local map point.
	/**
	 * @brief Whether the current frame is a keyframe
	 * @return true if needed
	 */
	void Tracking::UpdateLocalMap()
	{
	    // This is for visualization
	    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

	    // Update
	    UpdateLocalKeyFrames();//update keyframes
	    UpdateLocalPoints();//update map point
	}
	
	// Update local keyframes ------- part of the local map
	// How to select the local keyframe corresponding to the current frame
	 // Always limit the number of keys to no more than 80
	 // It can be modified here, it is time-consuming
	 // But the accuracy may decrease
	/*
	 When the number of keyframes is small (<=80), consider adding a second part of keyframes, 
	 which are closely related to the first part of keyframes, and always limit the number of keys to no more than 80.
	 The connections are closely reflected in three categories:
	 1. Keyframes with a high degree of co-visualization are the keyframes that have observed many map points in the current frame;
	 2. Sub keyframes;
	 3. Parent keyframe.


	transfer
	NeedNewKeyFrame() and 
	CreateNewKeyFrame()  two functions to complete.
	 */
	/**
	 * @brief Update local keyframes，called by UpdateLocalMap()
	 *
	 * Traverse the MapPoints of the current frame, take out the keyframes and adjacent keyframes where these MapPoints are observed, and update mvpLocalKeyFrames
	 */
     void Tracking::UpdateLocalKeyFrames()
	{
	    // Each map point vote for the keyframes in which it has been observed
	    map<KeyFrame*,int> keyframeCounter;
	    for(int i=0; i<mCurrentFrame.N; i++)
	    {
		if(mCurrentFrame.mvpMapPoints[i])//map point for the current frame
		{
		    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
		    if(!pMP->isBad())// observed
		    {
			const map<KeyFrame*,size_t> observations = pMP->GetObservations();
			for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
			    keyframeCounter[it->first]++;// Observation frame of the map point Number of times to observe the map point++
		    }
		    else
		    {
			mCurrentFrame.mvpMapPoints[i]=NULL;//Not observed Map point clear
		    }
		}
	    }

	    if(keyframeCounter.empty())
		return;

	    int max=0;
	    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

	    mvpLocalKeyFrames.clear();// Local keyframe clear
	    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

	    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
	    // map<KeyFrame*,int>::const_iterator
	    //  1. Keyframes with a high degree of co-visualization are the keyframes that have observed many map points in the current frame;	    
	    for( auto it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
	    {
		KeyFrame* pKF = it->first;//keyframes for map points

		if(pKF->isBad())
		    continue;
		if(it->second > max)// The keyframe with the most map points observed
		{
		    max = it->second;
		    pKFmax=pKF;
		}
		mvpLocalKeyFrames.push_back(it->first);// Save local keyframes
		pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
	    }

	    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
	    // vector<KeyFrame*>::const_iterator
	    // 
	    for(auto itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
	    {
		// Limit the number of keyframes
       		// Always limit the number of keys to no more than 80
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
	

	/*
	  The criteria for determining keyframes are as follows:
	 (1) After the last global relocation, another 20 frames have passed;
	 (2) The local mapping is idle, or 20 frames have passed after the last key frame was inserted;
	 (3) The current frame is tracked to more than 50 points;
	 (4) The current frame tracks 90% less than the reference key frame.
	*/
	/**
	 * @brief Whether the current frame is a keyframe
	 * @return true if needed
	 */
	bool Tracking::NeedNewKeyFrame()
	{
 	    // Step 1: If the user chooses to reposition on the interface, then no keyframes will be inserted
            // Since MapPoints are generated during the insertion of keyframes, the point cloud and keyframes on the map will not increase after the user chooses to relocate
	    if(mbOnlyTracking)// No mapping, no keyframes required
		return false;

	    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
	    // Mapping thread stopped
	    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
		return false;
	    // Number of keyframes in the map
	    const int nKFs = mpMap->KeyFramesInMap();

	    // Do not insert keyframes if not enough frames have passed from last relocalisation
	    // No need to insert keyframes just after repositioning The total number of keyframes exceeds the maximum value and no need to insert keyframes
    	    // Do not insert keyframes if not enough frames have passed from last relocalisation
            // Step 2: Determine whether the time since the last keyframe was inserted is too short
	    // mCurrentFrame.mnId is the ID of the current frame
	    // mnLastRelocFrameId is the ID of the last relocated frame
	    // mMaxFrames is equal to the frame rate of the image input
	    // If there are few keyframes, consider inserting keyframes or if it is more than 1s from the last reposition, consider inserting keyframes
	    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
		return false;
	    
	     // Step 3: Get the number of MapPoints tracked by the reference keyframe
	    // In the UpdateLocalKeyFrames function, the key frame with the highest degree of common view with the current key frame is set as the reference key frame of the current frame
	    // Tracked MapPoints in the reference keyframe
	    int nMinObs = 3;
	    if(nKFs <= 2)
		nMinObs = 2;
	    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
	    
	    // Step 4: Query whether the local map manager is busy
	    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
	    
	     // Step 5: For binocular or RGBD cameras, count the total number of MapPoints that can be added and the number of MapPoints tracked into the map
	    // Check how many "close" points are being tracked and how many could be potentially created.
	    int nNonTrackedClose = 0;
	    int nTrackedClose= 0;
	    if(mSensor != System::MONOCULAR)// binocular or rgbd
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
	    
	    // Step 6: Decide whether to insert keyframes
	    // Thresholds
	    // Set the inlier threshold, the inlier ratio that matches the feature points of the previous frame
	    // Thresholds sets the inlier threshold, the inlier ratio that matches the feature points of the previous frame
	    float thRefRatio = 0.75f;
	    if(nKFs<2)
		thRefRatio = 0.4f;// There is only one keyframe, so the threshold for inserting keyframes is very low

	    if(mSensor==System::MONOCULAR)
		thRefRatio = 0.9f;

	    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
	    // No keyframes inserted for a long time
	    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId+mMaxFrames;
	    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
	    // localMapper is idle
	    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
	    //Condition 1c: tracking is weak
	    // Track the rhythm of kneeling, 0.25 and 0.3 is a relatively low threshold
	    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
	    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
	   // The threshold is higher than c1c, and the repetition with the previous reference frame (the most recent key frame) is not too high
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
			// Too many keyframes cannot be blocked in the queue
			// The keyframes inserted by tracking are not inserted directly, 
			// but are inserted into mlNewKeyFrames first, and then localmapper pops them out one by one and inserts them into mspKeyFrames
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
	 * @brief Create new keyframes
	 *
	 * For non-single-purpose cases, create new MapPoints at the same time
	 */
	void Tracking::CreateNewKeyFrame()
	{
	    if(!mpLocalMapper->SetNotStop(true))
		return;
	    // keyframe add to map add to keyframe database
	    
	    // Step 1: Construct the current frame into a keyframe	    
	    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
	    
	    // Step 2: Set the current keyframe as the reference keyframe for the current frame
    	    // In the UpdateLocalKeyFrames function, the key frame with the highest degree of common view with the current key frame is set as the reference key frame of the current frame
	    mpReferenceKF = pKF;
	    mCurrentFrame.mpReferenceKF = pKF;
	    
    	    // This code has the same function as that part of the code in UpdateLastFrame
	    // Step 3: For binocular or rgbd cameras, generate new MapPoints for the current frame
	    if(mSensor != System::MONOCULAR)
	    {
	      // Calculate mRcw, mtcw and mRwc, mOw from Tcw
		mCurrentFrame.UpdatePoseMatrices();

		// We sort points by the measured depth by the stereo/RGBD sensor.
		// We create all those MapPoints whose depth < mThDepth.
		// If there are less than 100 close points we create the 100 closest.
		// binocular / depth
     		// Step 3.1: Get the feature points whose depth of the current frame is less than the threshold
               // Create new MapPoint, depth < mThDepth
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
	         // Step 3.2: Sort by depth from small to large
		    sort(vDepthIdx.begin(),vDepthIdx.end());
                 // Step 3.3: Pack the points that are relatively close into MapPoints
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
			    // These operations of adding properties are done every time a MapPoint is created
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
                	// This determines the density of the map point cloud for the binocular and rgbd cameras
                	// But it's not good to change these just to make the map dense, 
 			// because these MapPoints will participate in the entire slam process later
			if(vDepthIdx[j].first>mThDepth && nPoints>100)
			    break;
		    }
		}
	    }

	    mpLocalMapper->InsertKeyFrame(pKF);

	    mpLocalMapper->SetNotStop(false);


    // insert Key Frame into point cloud viewer
    mpPointCloudMapping->insertKeyFrame( pKF, this->mImGray, this->mImDepth, this->mImRGB );

	    mnLastKeyFrameId = mCurrentFrame.mnId;
	    mpLastKeyFrame = pKF;
	}
	
	/**
	 * @brief Tracking Local MapPoints
	 * Search Search the local map point corresponding to the current frame's local map point that matches the current frame's map point
	 * The local map point is searched for matching with the current frame key point descriptor, and the matched ones are added to the map points corresponding to the current frame feature points
	 * 
	 * Find the points within the field of view of the current frame in the local map, and perform projection matching between the points within the field of view and the feature points of the current frame
	 */
	void Tracking::SearchLocalPoints()
	{
	    // Do not search map points already matched
	    // Step 1: Traverse the mvpMapPoints of the current frame and mark these MapPoints not to participate in subsequent searches
           // Because the current mvpMapPoints must be in the field of view of the current frame
	    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit !=vend; vit++)
	    {
		MapPoint* pMP = *vit;// map point for the current frame
		if(pMP)
		{
		    if(pMP->isBad())
		    {
			*vit = static_cast<MapPoint*>(NULL);
		    }
		    else
		    {
			pMP->IncreaseVisible(); // Update the number of frames where the point can be observed plus 1
			pMP->mnLastFrameSeen = mCurrentFrame.mnId;// Mark the point as being observed by the current frame
			pMP->mbTrackInView = false;// Mark the point not to be projected in the future because it has already been matched
		    }
		}
	    }

	    int nToMatch=0;

	    // Project points in frame and check its visibility
	    // Step 2: Project all local MapPoints to the current frame, determine whether they are within the field of view, and then perform projection matching  
	    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
	    {
		MapPoint* pMP = *vit;// each map point of the local map
		// Has been observed by the current frame MapPoint no longer determines whether it can be observed by the current frame
		if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
		    continue;
		if(pMP->isBad())
		    continue;
		// Step 2.1: Determine whether the point in LocalMapPoints is in view
		// Project (this fills MapPoint variables for matching)
		if(mCurrentFrame.isInFrustum(pMP,0.5))
		{// Add 1 to the number of frames in which the point was observed, and the MapPoint is within the field of view of some frames
		    pMP->IncreaseVisible();
		    // Only MapPoints within the field of view participate in subsequent projection matching
		    nToMatch++;
		}
	    }

	    if(nToMatch>0)
	    {
		ORBmatcher matcher(0.8);// 0.8 Shortest distance and second shortest distance ratio difference
		int th = 1;
		if(mSensor==System::RGBD)
		    th=3;
		// If the camera has been relocalised recently, perform a coarser search
		 // If a relocation was performed not long ago, a broader search is performed and the threshold needs to be increased
		if(mCurrentFrame.mnId < mnLastRelocFrameId+2)
		    th=5;
		// Step 2.2: Match the feature points by projection to the MapPoints within the field of view
		// Search the local map points for map points that match the feature point descriptor of the current frame and add them to the map points corresponding to the feature points of the current frame
		matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
	    }
	}
	


	/*
	 * The process of relocation Relocalization is probably like this:
	 * 1. Calculate the BoW map of the current frame;
	 * 2. Find similar candidate keyframes in the keyframe database;
	 * 3. Match the current frame and each candidate key frame through BoW. If the number of matches is more than 15, perform EPnP solution;
	 * 4. Use BA optimization for the solution result. If there are few interior points, 
	 *    back-project the map points of the candidate key frame to the current frame to obtain additional matching points, 
	 *    re-establish candidate matching according to the grid and pyramid level to which the feature belongs, 
	 *    and select the optimal matching; If this is still not enough, discard the candidate key frame, 
	 *    if it is enough, add the additional map points obtained through back projection, and then optimize.
	 * 5. If the interior point meets the requirements (>50), the relocation is successful, and the id of the latest relocation is updated: mnLastRelocFrameId = mCurrentFrame.mnId; otherwise, it returns false.
	 */
	/**
	 * @brief update LocalMap
	 *
	 * Local maps include: \n
	 * - K1 keyframes, K2 adjacent keyframes and reference keyframes
	 * - MapPoints observed by these keyframes
	 */
	bool Tracking::Relocalization()
	{
  	    // 1. Calculate the BoW map of the current frame; Compute Bag of Words Vector
	    // Dictionary N M-dimensional words
	    // Descriptor of a frame n M-dimensional descriptors
	    // Generate a N*1 vector Record the descriptor of a frame Use dictionary words
	    mCurrentFrame.ComputeBoW();

	    // Relocalization is performed when tracking is lost
	    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
 	    // 2. Find similar candidate key frames in the key frame database; calculate the dictionary word vector represented by the linear representation of the frame descriptor dictionary word and the linear representation vector of each key frame in the key frame database. 
	    // Find some frames with the closest distance as candidate key frames
	    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
	    if(vpCandidateKFs.empty())
		return false;
	    const int nKFs = vpCandidateKFs.size();// total candidate keyframes

	    // We perform first an ORB matching with each candidate
	    // If enough matches are found we setup a PnP solver
	    ORBmatcher matcher(0.75,true);// Descriptor Matcher Minimum Distance < 0.75*Short Distance
	    vector<PnPsolver*> vpPnPsolvers;//Match point Rt solver between two keyframes
	    vpPnPsolvers.resize(nKFs);// There is a solver for the current frame and each candidate keyframe
	    vector<vector<MapPoint*> > vvpMapPointMatches;
	    // The keypoint descriptor of the current frame and the matching point of each candidate keyframe map point descriptor
	    
	    vvpMapPointMatches.resize(nKFs);//map point match between two keyframes
	    vector<bool> vbDiscarded;// The candidate keyframe matches the current frame Good or bad flag
	    vbDiscarded.resize(nKFs);

	    int nCandidates=0;

	    for(int i=0; i<nKFs; i++)// Each candidate keyframe in the keyframe database
	    {
	      
		KeyFrame* pKF = vpCandidateKFs[i];// each candidate keyframe
		if(pKF->isBad())
		    vbDiscarded[i] = true;//  bad
		
		else
		{
		    // 3. Match the current frame and each candidate key frame through BoW. If the number of matches is more than 15, perform EPnP solution;	  
		    int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
		    if(nmatches<15)
		    {
			vbDiscarded[i] = true;// bad match
			continue;
		    }
		    else //enough matches >15 to join the solver
		    {
		      // Generate solver
			PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
			pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);// random sampling
			vpPnPsolvers[i] = pSolver;// Add solver
			nCandidates++;
		    }
		}
	    }
	    // Alternatively perform some iterations of P4P RANSAC
	    // Until we found a camera pose supported by enough inliers
      	    // until a candidate matching keyframe is found and a sufficient number of inliers conforming to the transformation relation Rt
	    bool bMatch = false;
	    ORBmatcher matcher2(0.9,true);

	    while(nCandidates>0 && !bMatch)
	    {
		for(int i=0; i<nKFs; i++)// 
		{
		    if(vbDiscarded[i])// Skip candidate keyframes with poor matching performance
			continue;

		    // Perform 5 Ransac Iterations 
		    vector<bool> vbInliers;// The number of interior points that fit the transformation
		    int nInliers;
		    bool bNoMore;
           	    //Solver Solver Perform EPnP Solver
		    PnPsolver* pSolver = vpPnPsolvers[i];
		    cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);//Iterate 5 times to get the transformation matrix

		    // If Ransac reachs max. iterations discard keyframe
		    if(bNoMore)//迭代5次效果还不好
		    {
			vbDiscarded[i]=true;// EPnP solution is not good, the matching effect is poor, and the candidate key frame is discarded
			nCandidates--;
		    }
		    // 4. Use BA optimization for the solution results. If there are few interior points, back-project the map points of the current frame to the candidate key frames to obtain additional matching points;
		    // If this is still not enough, discard the candidate key frame, if it is enough, add the additional map points obtained through back projection, and then optimize.
		    // If a Camera Pose is computed, optimize
		    if(!Tcw.empty())
		    {
			Tcw.copyTo(mCurrentFrame.mTcw);

			set<MapPoint*> sFound;// map point

			const int np = vbInliers.size();// number of interior points that conform to pose Tcw

			for(int j=0; j<np; j++)
			{
			    if(vbInliers[j])// interior point of each sign
			    {
				mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];// Corresponding map point i frame map point under j frame
				sFound.insert(vvpMapPointMatches[i][j]);
			    }
			    else
				mCurrentFrame.mvpMapPoints[j]=NULL;
			}
		      // Use BA to optimize the pose and return the 3d-2d optimized edges with better optimization results
			int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

			if(nGood<10)
			    continue;// This candidate frame is not good after matching optimization

			for(int io =0; io<mCurrentFrame.N; io++)
			    if(mCurrentFrame.mvbOutlier[io])// Update state after optimization
				mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);// The map point is a null pointer

     			 // If few inliers, search by projection in a coarse window and optimize again
      			// If there are fewer interior points, backproject the map point vpCandidateKFs[i] of the candidate key frame to the pixel coordinate system of the current frame
      			// According to the grid and pyramid level information, select the feature point that matches the map point under the current frame
      			// Get extra match points
			if(nGood<50)
			{
			    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

			    if(nadditional+nGood>=50)
			    {
			      // The number of map points corresponding to the feature points of the current frame is greater than 50 for optimization
				nGood = Optimizer::PoseOptimization(&mCurrentFrame);// Returns the number of interior points

				// If many inliers but still not enough, search by projection again in a narrower window
				// the camera has been already optimized with many points
				if(nGood>30 && nGood<50)
				{
				    sFound.clear();
				    for(int ip =0; ip<mCurrentFrame.N; ip++)
					if(mCurrentFrame.mvpMapPoints[ip])
					    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
				   // Narrow the search window
				    nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

				    // Final optimization
				    if(nGood+nadditional>=50)
				    {
					nGood = Optimizer::PoseOptimization(&mCurrentFrame);

					for(int io =0; io<mCurrentFrame.N; io++)
					    if(mCurrentFrame.mvbOutlier[io])//Outer point
						mCurrentFrame.mvpMapPoints[io]=NULL;// null pointer
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
		mnLastRelocFrameId = mCurrentFrame.mnId;// Relocation Frame ID
		return true;
	    }

	}

	// Track reset
	void Tracking::Reset()
	{

	    cout << "System Reseting" << endl;
	    if(mpViewer)
	    {
		mpViewer->RequestStop();
		while(!mpViewer->isStopped())
		    usleep(3000);
	    }

	    // Reset Local Mapping
	    cout << "Reseting Local Mapper...";
	    mpLocalMapper->RequestReset();
	    cout << " done" << endl;

	    // Reset Loop Closing
	    cout << "Reseting Loop Closing...";
	    mpLoopClosing->RequestReset();
	    cout << " done" << endl;

	    // Clear BoW Database
	    cout << "Reseting Database...";
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

	// reread configuration file
	// In-camera parameters
	// Distortion Correction Parameters
	// Baseline length × focal length
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
	
	// Track + Mapping Mode
	void Tracking::InformOnlyTracking(const bool &flag)
	{
	    mbOnlyTracking = flag;
	}



} //namespace ORB_SLAM
