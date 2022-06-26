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
