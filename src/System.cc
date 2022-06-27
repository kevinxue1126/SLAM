/**
* This file is part of ORB-SLAM2.
* ORB mainly draws on the ideas of PTAM, and the work for reference mainly includes:
* Rubble's ORB feature points;
* DBow2's place recognition is used for closed-loop detection;
* Strasdat's closed-loop correction and covisibility graph ideas;
* and g2o by Kuemmerle and Grisetti for optimization.
* 
* 
* System entry:
* 1】Input image to get camera position
*       Monocular GrabImageMonocular(im);
*       Binocular GrabImageStereo(imRectLeft, imRectRight);
*       Depth GrabImageMonocular(imRectLeft, imRectRight);
* 
* 2】Convert to Grayscale
*       Monocular mImGray
*       Binocular mImGray, imGrayRight
*       Depth mImGray, imDepth
* 
* 3】Structure Frame
*       Monocular      Uninitialized        Frame(mImGray, mpIniORBextractor)
*       Monocular      initialized          Frame(mImGray, mpORBextractorLeft)
*       Binocular      Frame(mImGray,       imGrayRight, mpORBextractorLeft, mpORBextractorRight)
*       Depth          Frame(mImGray,       imDepth,        mpORBextractorLeft)
* 
* 4】Track
*   The data stream enters the Tracking thread   Tracking.cc
* 
* 
* 
* ORB-SLAM utilizes three threads for tracking, map building and loop closure detection respectively.

一、Track

    ORB feature extraction
    Initial pose estimation (velocity estimation)
    Attitude optimization (Track local map, use adjacent map points to find more feature matching, optimize attitude)
    select keyframes

二、Map building

    Add keyframes (update various graphs)
    Validate recently added map points (remove Outlier)
    Generate new map points (trigometry)
    Local Bundle adjustment (this keyframe and adjacent keyframes, remove Outlier)
    Validate keyframes (remove duplicate frames)

三、Closed loop detection

    Select similar frames（bag of words）
    Detect closed loop (calculate similarity transformation (3D<->3D, there is scale drift, so it is similarity transformation), RANSAC calculates the number of inner points)
    Fusion of 3D points to update various graphs
    Graph optimization (conductive transformation matrix), update all points of the map

* 
* 
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>


#include <time.h>
// String search
// Find the suffix, whether there is a suffix string in str
bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}

namespace ORB_SLAM2
{
	  // Default initialization function  word list file txt/bin file  configuration file     Sensors: Monocular, Binocular, Depth
	  System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,const bool bUseViewer):
			      mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
			      mbDeactivateLocalizationMode(false)//Initialize the variable directly
	  {
	      // Output welcome message
	      cout << endl <<
	      "ORB-SLAM2 SLAM" << endl <<
	      "under certain conditions. See LICENSE.txt." << endl << endl;

	      cout << "Enter camera as: ";
	      if(mSensor==MONOCULAR)
		  cout << "Monocular" << endl;
	      else if(mSensor==STEREO)
		  cout << "Stereo" << endl;
	      else if(mSensor==RGBD)
		  cout << "RGB-D" << endl;

	      //Check settings file opencv
	      cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
	      if(!fsSettings.isOpened())
	      {
		cerr << "Can't open settings file :  " << strSettingsFile << endl;
		exit(-1);
	      }
	      
	      // set for point cloud resolution
	      float resolution = fsSettings["PointCloudMapping.Resolution"];

	      //Load ORB Vocabulary
	      cout << endl << "Load ORB dictionary. This could take a while..." << endl;

	       
	       */
	  //  Open dictionary file
	  /////// ////////////////////////////////////
	      clock_t tStart = clock();// start time
	      // 1. create dictionary mpVocabulary = new ORBVocabulary()；and load the dictionary from the file =========================
	      mpVocabulary = new ORBVocabulary();// keyframe dictionary database
	      //bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
	      bool bVocLoad = false; //  bool  open dictionary flag
	      if (has_suffix(strVocFile, ".txt"))//
		    bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);//txt format open  
		    // -> Dereferencing and accessing member functions of pointer objects are equivalent to(*mpVocabulary).loadFromTextFile(strVocFile);
	      else
		    bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);//bin format open
	      if(!bVocLoad)
	      {
		  cerr << "dictionary path error " << endl;
		  cerr << "open file error: " << strVocFile << endl;
		  exit(-1);
	      }
	      printf("Database load time. Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);//Display file load time (seconds)
	  ///////////////////////////////////////////


	      // 2. Use the feature dictionary mpVocabulary to create a keyframe database KeyFrameDatabase
	      mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
	     // 3. Create a map object Map
	      mpMap = new Map();
	      // 4. Create Drawers. These are used by the Viewer
	      mpFrameDrawer = new FrameDrawer(mpMap);//keyframe display
	      mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);//map display

	      
	      // Initialize pointcloud mapping
	      mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

	      //Initialize the Tracking thread
	      //(it will live in the main thread of execution, the one that called this constructor)
	      // 5. Initializing trace thread object not started
	      //mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
	      //		      mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);
              mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                 mpMap, mpPointCloudMapping, mpKeyFrameDatabase, strSettingsFile, mSensor);


	      //Initialize the Local Mapping thread and launch
	      // 6. Initialize the local map construction thread and start it
	      mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
	      mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

	      //Initialize the Loop Closing thread and launch
	      // 7. Initialize the closed loop detection thread and start it
	      mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
	      mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

	      //Initialize the Viewer thread and launch

	      // 8. Initialize trace thread visualization and start
	      if(bUseViewer)
	      {
		  mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
		  mptViewer = new thread(&Viewer::Run, mpViewer);
		  mpTracker->SetViewer(mpViewer);
	      }

	      // 9. Set pointers between threads
	      mpTracker->SetLocalMapper(mpLocalMapper);   // Tracking thread associated with local mapping and loop closure detection thread
	      mpTracker->SetLoopClosing(mpLoopCloser);

	      mpLocalMapper->SetTracker(mpTracker);       // Local Mapping Thread Association Tracking and Loop Closure Detection Thread
	      mpLocalMapper->SetLoopCloser(mpLoopCloser);

	      mpLoopCloser->SetTracker(mpTracker);	  // Loop closure detection thread association tracking and local mapping thread
	      mpLoopCloser->SetLocalMapper(mpLocalMapper);
	  }


	  // Binocular Tracking Constant Mat Quantity Left, Right, Timestamp, Return Camera Pose 
	  cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
	  {
	      if(mSensor!=STEREO)
	      {
		  cerr << "The binocular tracking TrackStereo is called, but the sensor setting is not the binocular STEREO." << endl;
		  exit(-1);
	      }   

  	      // 1.  Detection of Mode Changes  Tracking + Mapping or Tracking + Positioning + Mapping Check mode change============   
	      {
		/*
		* The class unique_lock is a wrapper for a generic mutex owner,
		* Provides delayed locking（deferred locking），
		* Try for a limited time（time-constrained attempts），
		* Recursive locking（recursive locking），
		* Lock owner transitions, 
		* and use of condition variables.
		*/
		  unique_lock<mutex> lock(mMutexMode);//thread lock
		  //positioning mode  tracking + positioning   Mapping off
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();// Stop the mapping thread first
		      // Wait until Local Mapping has effectively stopped 
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);//rest 1s
		      }

		      mpTracker->InformOnlyTracking(true);//Enable tracking thread
		      mbActivateLocalizationMode = false;
		  }
		  //non-location-only mode   Tracking + Positioning + Mapping
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);//Enable tracking thread
		      mpLocalMapper->Release();// Release the mapping thread   
		      mbDeactivateLocalizationMode = false;
		  }
	      }

  	      // 2. Check the tracking thread restart
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();// thread reset
		      mbReset = false;
		  }
	      }
  	      // 3. binocular tracking
	      cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);//get camera pose
	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;//state
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;//tracked map points
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
	      return Tcw;
	  }

	  // depth camera tracking
	  cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
	  {
	      if(mSensor!=RGBD)
	      {
		  cerr << "The depth camera tracking TrackRGBD is called, but the sensor is not depth camera RGBD." << endl;
		  exit(-1);
	      }
    
     	      // 1.  Detection of Mode Changes (Tracking + Mapping or Tracking + Positioning + Mapping) Check mode change
	      {
		  unique_lock<mutex> lock(mMutexMode);
		  //  tracking + positioning
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();//stop mapping
		      // Wait until Local Mapping has effectively stopped
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);
		      }
		      mpTracker->InformOnlyTracking(true);// track only
		      mbActivateLocalizationMode = false;
		  }
		  // Tracking + Mapping
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);// Tracking + Mapping
		      mpLocalMapper->Release();// Release the mapping thread
		      mbDeactivateLocalizationMode = false;
		  }
	      }

  	      // 2. Check Tracking Tracking Thread Restart  
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();//thread reset
		      mbReset = false;
		  }
	      }
	      // initialization
	      // The number of feature points in the current frame is greater than 500 for initialization
	      // Set the first frame as the key frame pose as [I 0] 
	      // Calculate 3D points based on the depth obtained from the disparity of the first frame
	      // Generate map, add map point, map point observation frame, map point best descriptor, update map point direction and distance 
	      // map points for keyframes, add map points to the current frame, add map points to the map
	      // show map
	      //  ---- Calculate the transformation from the reference frame to the current frame  Tcr = mTcw  * mTwr
	   
	      // later frame
	      // If there is motion, track the previous frame, if the tracking fails, track the reference key frame
	      // No movement or recently performed, relocation is tracked, the most recent keyframe, the reference keyframe
	      // If the reference keyframe tracking fails, relocate and track all keyframes
	      // track local map
	      //  ---- Calculate the transformation from the reference frame to the current frame Tcr = mTcw  * mTwr

  	      // 3. RGBD camera tracking	        
	      cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);//get camera pose
	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
	      return Tcw;
	  }


	  // Monocular tracking
	  cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
	  {
	      if(mSensor!=MONOCULAR)
	      {
		  cerr << "The monocular trackMonocular is called, but the input sensor is not a monocular." << endl;
		  exit(-1);
	      }

    	      // 1.  Detection of mode change (tracking+mapping or tracking+localization+mapping)  Check mode change
	      {
		  unique_lock<mutex> lock(mMutexMode);// map locked
	       	  //  tracking + positioning 
		  if(mbActivateLocalizationMode)
		  {
		      mpLocalMapper->RequestStop();// stop mapping
		      // Waiting for the mapping thread to stop
		      while(!mpLocalMapper->isStopped())
		      {
			  usleep(1000);
		      }
		      mpTracker->InformOnlyTracking(true);//track only
		      mbActivateLocalizationMode = false;
		  }
	          // Tracking + Mapping + Positioning
		  if(mbDeactivateLocalizationMode)
		  {
		      mpTracker->InformOnlyTracking(false);
		      mpLocalMapper->Release();//
		      mbDeactivateLocalizationMode = false;
		  }
	      }

    	      // 2. Check Tracking Thread Restart
	      {
		  unique_lock<mutex> lock(mMutexReset);
		  if(mbReset)
		  {
		      mpTracker->Reset();
		      mbReset = false;
		  }
	      }
    	      // 3. Monocular tracking
	      // initialization
	      // The number of feature points in two consecutive frames is greater than 100 and the number of pairs of key points matching in two frames is greater than 100
	      // Initial frame [I 0] Second frame Fundamental matrix/homography recovery [R t] Global optimization and corresponding 3D points
	      // Create a map to optimize the map using the minimized reprojection error BA, optimize the pose and map points
	      // The inverse of the median depth distance, the translation vector of the normalized second frame pose and the three-axis coordinates of the map point
	      // show update	 
	      // later frame
	      cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);// Monocular tracking, get the camera pose
	      unique_lock<mutex> lock2(mMutexState);
	      mTrackingState = mpTracker->mState;
	      mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
	      mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

	      return Tcw;
	  }

	  //  Activate the mapping thread
	  void System::ActivateLocalizationMode()
	  {
	      unique_lock<mutex> lock(mMutexMode);
	      mbActivateLocalizationMode = true;
	  }

	  // It is the failure of the mapping thread
	  void System::DeactivateLocalizationMode()
	  {
	      unique_lock<mutex> lock(mMutexMode);
	      mbDeactivateLocalizationMode = true;
	  }

	  // map change sign
	  bool System::MapChanged()
	  {
	      static int n=0;
	      int curn = mpMap->GetLastBigChangeIdx();
	      if(n<curn)
	      {
		  n=curn;
		  return true;
	      }
	      else
		  return false;
	  }

	  // System reset reset
	  void System::Reset()
	  {
	      unique_lock<mutex> lock(mMutexReset);
	      mbReset = true;
	  }

 	  // system shut down, Close all threads completely
	  void System::Shutdown()
	  {
	      mpLocalMapper->RequestFinish();  // Finish and close the mapping thread
	      mpLoopCloser->RequestFinish();   // Finish and close the loop closure detection thread
              mpPointCloudMapping->shutdown(); // Turn off point cloud mapping
	      if(mpViewer)//explain visualization thread
	      {
		  mpViewer->RequestFinish();
		  while(!mpViewer->isFinished())
		      usleep(5000);
	      }
	      // Wait until all thread have effectively stopped
	      while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
	      {
		  usleep(5000);
	      }
	      if(mpViewer)
		  pangolin::BindToContext("ORB-SLAM2: Map Viewer");
	  }

	  // Save TUM dataset, camera pose trajectory
	  void System::SaveTrajectoryTUM(const string &filename)
	  {
	      cout << endl << "Save camera pose trajectory to file " << filename << " ..." << endl;
	      // 单目相机
	      if(mSensor==MONOCULAR)
	      {
		  cerr << "Monocular cameras are not suitable for TUM dataset" << endl;
		  return;
	      }

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// keyframe vector array container storage 
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);// Sorting ensures that the first keyframe is at the origin
	      cv::Mat Two = vpKFs[0]->GetPoseInverse();// world coordinate system pose

	      ofstream f;// File to save camera pose trajectory
	      f.open(filename.c_str());
	      f << fixed;
	      // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
	      // which is true when tracking failed (lbL).
	      list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();// The reference frame (previous frame) of the key frame
	      list<double>::iterator lT = mpTracker->mlFrameTimes.begin();// timestamp
	      list<bool>::iterator lbL = mpTracker->mlbLost.begin();// Sign Tracking Failed
	      for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
		  lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
	      {
		  if(*lbL)// Tracking failed continue
		      continue;

		  KeyFrame* pKF = *lRit;//pointer     the reference frame (previous frame) of the key frame

		  cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);// 4*4 identity matrix

		  // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
		  while(pKF->isBad())
		  {
		      Trw = Trw*pKF->mTcp;
		      pKF = pKF->GetParent();//reference frame reference frame
		  }

		  Trw = Trw*pKF->GetPose()*Two;// TWO is the starting camera pose
		  cv::Mat Tcw = (*lit)*Trw;// transformation matrix
		  cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();// rotation matrix
		  cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);// translation vector

		  vector<float> q = Converter::toQuaternion(Rwc);// Four elements corresponding to the rotation matrix
	  	  // 6-bit precision (timestamp + 9-bit precision) (translation vector + 9-bit precision) four-element pose
		  f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
	      }
	      f.close();
	      cout << endl << "trajectory saved!" << endl;
	  }

	 //Save keyframe tracks
	  void System::SaveKeyFrameTrajectoryTUM(const string &filename)
	  {
	      cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// keyframe vector array container storage 
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//Sorting ensures that the first keyframe is at the origin

	      // Transform all keyframes so that the first keyframe is at the origin.
	      // After a loop closure the first keyframe might not be at the origin.
	      //cv::Mat Two = vpKFs[0]->GetPoseInverse();

	      ofstream f;
	      f.open(filename.c_str());
	      f << fixed;

	      for(size_t i=0; i<vpKFs.size(); i++)
	      {
		  KeyFrame* pKF = vpKFs[i];// Keyframe

		// pKF->SetPose(pKF->GetPose()*Two);

		  if(pKF->isBad())
		      continue;
		  // The pose of the key frame has been converted to the first frame image coordinate system (world coordinate system)
		  cv::Mat R = pKF->GetRotation().t();// rotation matrix
		  vector<float> q = Converter::toQuaternion(R);// four elements
		  cv::Mat t = pKF->GetCameraCenter();// Translation matrix The position of the center point of the camera coordinate system of the current frame
		  f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
		    << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

	      }
	      f.close();
	      cout << endl << "trajectory saved!" << endl;
	  }
	  // KITTI========================================================================
	  void System::SaveTrajectoryKITTI(const string &filename)
	  {
	      cout << endl << "maintain camera pose  " << filename << " ..." << endl;
	      if(mSensor==MONOCULAR)
	      {
		  cerr << "Monocular is not suitable for KITTI dataset " << endl;
		  return;
	      }

	      vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();// keyframe vector array container storage
	      sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);//Sorting ensures that the first keyframe is at the origin
	      cv::Mat Two = vpKFs[0]->GetPoseInverse();// world coordinate system pose

	      ofstream f;
	      f.open(filename.c_str());
	      f << fixed;
	      // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
	      // which is true when tracking failed (lbL).
	      list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();// The reference value of the key frame, the key frame pose has been transformed into the world coordinate system
	      list<double>::iterator lT = mpTracker->mlFrameTimes.begin();//timestamp
	      for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
	      {
		  ORB_SLAM2::KeyFrame* pKF = *lRit;// reference frame

		  cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);// Transformation matrix Identity matrix

		  while(pKF->isBad())
		  {
		    //  cout << "bad parent" << endl;
		      Trw = Trw*pKF->mTcp;
		      pKF = pKF->GetParent();
		  }

		  Trw = Trw*pKF->GetPose()*Two;//Multiply by the reference value in turn

		  cv::Mat Tcw = (*lit)*Trw;
		  cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
		  cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

		  f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
		      Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
		      Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
	      }
	      f.close();
	      cout << endl << "trajectory saved!" << endl;
	  }
	  // ================================================
	  int System::GetTrackingState()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackingState;
	  }
	  //=================================================
	  vector<MapPoint*> System::GetTrackedMapPoints()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackedMapPoints;
	  }
	  //=================================================
	  vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
	  {
	      unique_lock<mutex> lock(mMutexState);
	      return mTrackedKeyPointsUn;
	  }

} //namespace ORB_SLAM
