/**
* This file is part of ORB-SLAM2.
*  Normal frame Each image will generate a frame
* 
************Binocular camera frame*************
* left and right
* orb feature extraction
* Calculate matching point pairs
* Calculate the depth distance of the corresponding keypoint from the disparity
* 
* binocular stereo matching
* 1】Establish a band area search table for each feature point of the left eye, and limit the search area, (the polar line correction has been performed before)
* 2】In the limited area, the feature points are matched by the descriptor, and the best matching point of each feature point is obtained (scaleduR0)
* 3】The matching correction amount bestincR is obtained through the SAD sliding window
* 4】(bestincR, dist)  (bestincR - 1, dist)  (bestincR +1, dist) Fitting a parabola with three points to get the sub-pixel correction deltaR
* 5】The final matching point position is : scaleduR0 + bestincR  + deltaR
* 
* Parallax Disparity and Depth
* z = bf /d      
* 
* Assign meshes to feature points
* 
******depth camera frame******************
* The depth value is determined by the uncorrected feature point coordinates for the value in the corresponding depth map
* The abscissa value of the matching point coordinate is the feature point coordinate
* Assign meshes to feature points
* 
********Monocular camera frame****************
* Depth value container initialization
* Match point coordinates and container initialization
* Assign meshes to feature points
* 
* 
* *************image pyramid********************************
* Layer 0 is the original image. After convolving the 0-layer Gaussian kernel, 
* downsampling (deleting all even-numbered rows and even-numbered columns) can obtain the first layer of the Gaussian pyramid; 
* insert 0 and restore it to the original size with Gaussian convolution, 
* subtract it from the 0-layer, and get the 0-layer pull The Plath pyramid corresponds to the information lost during the filtering and 
* downsampling process of the 0-layer Gaussian pyramid, on which features can be extracted. 
* Then continuously downsample to obtain Gaussian pyramids and Laplacian pyramids of different layers. 
* The scale corresponding to the extracted features is positively correlated with the number of layers of the pyramids. 
* The higher the number of layers, the larger the corresponding scale and the higher the scale uncertainty. 
* Through this processing of images, we can extract features that are invariant in size. 
* However, during feature matching, it is necessary to consider the corresponding number of layers (scales) when extracting feature points.
* 
* 
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

      long unsigned int Frame::nNextId=0;
      bool Frame::mbInitialComputations=true;
      float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
      float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
      float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

      Frame::Frame()
      {}

      /**
      * @brief Copy constructor    The default initialization is mainly to directly assign and write to variables in the class
      *
      * copy constructor, mLastFrame = Frame(mCurrentFrame)
      */
      Frame::Frame(const Frame &frame)
	  :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
	  mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
	  mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
	  mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
	  mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
	  mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
	  mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
	  mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
	  mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
	  mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
	  mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
      {
	  for(int i=0;i<FRAME_GRID_COLS;i++)//64 columns
	      for(int j=0; j<FRAME_GRID_ROWS; j++)//48 lines
		  mGrid[i][j]=frame.mGrid[i][j];//array of vectors

	  if(!frame.mTcw.empty())
	      SetPose(frame.mTcw);
      }
      
      /**
      * @brief  Initialize frame
      * @param imLeft       		 left image
      * @param imRight    		 right image
      * @param timeStamp		 timestamp
      * @param extractorLeft             left image orb feature extractor
      * @param extractorRight            right image orb feature extractor
      * @param voc                       orb dictionary
      * @param K                         in-camera parameters
      * @param distCoef                  distortion Correction Parameters
      * @param bf                        binocular camera baseline × focal length
      * @param thDepth                  
      */
      Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
	  :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
	  mpReferenceKF(static_cast<KeyFrame*>(NULL))
      {
	  // Frame ID
	  mnId=nNextId++;

	  // Scale Level Info
	  mnScaleLevels = mpORBextractorLeft->GetLevels();// feature extraction
	  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();//scale
	  mfLogScaleFactor = log(mfScaleFactor);
	  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

	  // ORB extraction
	  thread threadLeft(&Frame::ExtractORB,this,0,imLeft);// left camera 
	  thread threadRight(&Frame::ExtractORB,this,1,imRight);// right camera
	  threadLeft.join();//join the thread
	  threadRight.join();
		  
	  N = mvKeys.size();//The number of key points in the left image
	  if(mvKeys.empty())
	      return;
	  // Undistort feature points, no binocular correction is performed here, because the input image is required to be polar corrected
	  UndistortKeyPoints();
	  
      	// binocular matching
	  ComputeStereoMatches();

      // Initialize map points and their outer points;
	  mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
	  mvbOutlier = vector<bool>(N,false);// Whether the corresponding map point is an outer point


	  // This is done only for the first Frame (or after a change in the calibration)
	  if(mbInitialComputations)
	  {
	    // 对于未校正的图像for uncorrected images
	    ComputeImageBounds(imLeft);
	    // 640*480 image divided into 64*48 grids
	      mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);// The number of grids occupied by each pixel
	      mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

	      fx = K.at<float>(0,0);
	      fy = K.at<float>(1,1);
	      cx = K.at<float>(0,2);
	      cy = K.at<float>(1,2);
	      invfx = 1.0f/fx;
	      invfy = 1.0f/fy;

	      mbInitialComputations=false;
	  }

	  mb = mbf/fx;// Binocular camera baseline length
	
	// According to the pixel coordinates of the feature points are allocated to each grid
	  AssignFeaturesToGrid();
      }
      
      /**
      * @brief  initialization
      * @param mGray		 grayscale
      * @param imDepth           depth map
      */
      Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
	  :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
	  mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
      {
	  // Frame ID int type
	  mnId=nNextId++;

	  // Scale Level Info
	  mnScaleLevels = mpORBextractorLeft->GetLevels();
	  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
	  mfLogScaleFactor = log(mfScaleFactor);
	  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

	  // ORB extraction
	  ExtractORB(0,imGray);
	
          // number of feature points
	  N = mvKeys.size();

	  if(mvKeys.empty())
	      return;
	  // Correct keypoint coordinates 
	  UndistortKeyPoints();
	  
      	   // depth camera 
	  ComputeStereoFromRGBD(imDepth);

	  // Map points converted from key points
	  mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	  mvbOutlier = vector<bool>(N,false);

	  // This is done only for the first Frame (or after a change in the calibration)
	  if(mbInitialComputations)
	  {
	    // Calculated for uncorrected images
	      ComputeImageBounds(imGray);
	    // 640*480 image divided into 64*48 grids
	      mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
	      mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

	      fx = K.at<float>(0,0);
	      fy = K.at<float>(1,1);
	      cx = K.at<float>(0,2);
	      cy = K.at<float>(1,2);
	      invfx = 1.0f/fx;
	      invfy = 1.0f/fy;

	      mbInitialComputations=false;
	  }
		
		  
	// Binocular camera baseline length
	  mb = mbf/fx;
	
	// According to the pixel coordinates of the feature points are allocated to each grid
	  AssignFeaturesToGrid();
      }
      
      /**
      * @brief  initialization frame
      * @param mGray		 grayscale
      * 
      */
      Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
	  :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
	  mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
      {
	  // Frame ID
	  mnId=nNextId++;

	  // Scale Level Info
	  mnScaleLevels = mpORBextractorLeft->GetLevels();
	  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	  mfLogScaleFactor = log(mfScaleFactor);
	  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

	  // ORB extraction
	  ExtractORB(0,imGray);

	  N = mvKeys.size();// number of feature points

	  if(mvKeys.empty())
	      return;

	  // Correct keypoint coordinates
	  UndistortKeyPoints();

	  // Set no stereo information
	  mvuRight = vector<float>(N,-1);
	  mvDepth = vector<float>(N,-1);
	  
	  // map point
	  mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	  mvbOutlier = vector<bool>(N,false);

	  // This is done only for the first Frame (or after a change in the calibration)
	  if(mbInitialComputations)
	  {
	    // Calculated for uncorrected images
	      ComputeImageBounds(imGray);
	    // 640*480 image divided into 64*48 grids
	      mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
	      mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

	      fx = K.at<float>(0,0);
	      fy = K.at<float>(1,1);
	      cx = K.at<float>(0,2);
	      cy = K.at<float>(1,2);
	      invfx = 1.0f/fx;
	      invfy = 1.0f/fy;

	      mbInitialComputations=false;
	  }

	  mb = mbf/fx;// Binocular camera baseline length
	
	// According to the pixel coordinates of the feature points are allocated to each grid
	  AssignFeaturesToGrid();
      }
  
  
/**
 * @brief        Keypoints are assigned by mesh to speed up matching
 * 
 */    
      void Frame::AssignFeaturesToGrid()
      {
	  int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
	  for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
	      for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
		  mGrid[i][j].reserve(nReserve);

	  for(int i=0;i<N;i++)
	  {
	      const cv::KeyPoint &kp = mvKeysUn[i];

	      int nGridPosX, nGridPosY;
	      if(PosInGrid(kp,nGridPosX,nGridPosY))
		  mGrid[nGridPosX][nGridPosY].push_back(i);
	  }
      }

      // Keypoint Extraction + Descriptor
      void Frame::ExtractORB(int flag, const cv::Mat &im)
      {
	  if(flag==0)
	      (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);//Left image extractor
	  else
	      (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);// Right image extractor
      }

/**
 * @brief Set the camera pose.
 * 
 * 
 * @param Tcw Transformation from world to camera
 */
      void Frame::SetPose(cv::Mat Tcw)
      {
	  mTcw = Tcw.clone();// set pose
	  // Tcw_.copyTo(mTcw);// Copy to class variable w2c
	  UpdatePoseMatrices();// update rotation matrix
      }
 
/**
 * @brief Computes rotation, translation and camera center matrices from the camera pose.
 *
 * Calculate mRcw, mtcw and mRwc, mOw from Tcw
 */     
      void Frame::UpdatePoseMatrices()
      { 
	 // [x_camera 1] = [R|t]*[x_world 1]，Coordinates are in homogeneous form
         // x_camera = R*x_world + t
	  mRcw = mTcw.rowRange(0,3).colRange(0,3);
	  mRwc = mRcw.t();                      
	  mtcw = mTcw.rowRange(0,3).col(3);            
	  mOw = -mRwc*mtcw;
      }
      
/**
 * @brief Determine whether a point is in the field of view, 
 * check whether the map point is in the current field of view, 
 * the point depth under the camera coordinate system is less than 0, 
 * and the point is not in the field of view, 
 * and the horizontal and vertical coordinates of the point under the pixel coordinate system are within the corrected image size, 
 * and reprojection is calculated. Coordinates, the angle between the observation direction, the scale of the prediction in the current frame
 * 
 * @param  pMP             MapPoint
 * @param  viewingCosLimit Orientation thresholds for viewing angle and average viewing angle
 * @return                 true if is in view
 * @see SearchLocalPoints()
 */
      bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
      {
	  pMP->mbTrackInView = false;// initialization

	  // 3D in absolute coordinates
	  cv::Mat P = pMP->GetWorldPos();

	  // 3D in camera coordinates
	  const cv::Mat Pc = mRcw*P+mtcw;  
	  const float &PcX = Pc.at<float>(0);
	  const float &PcY= Pc.at<float>(1);
	  const float &PcZ = Pc.at<float>(2);

	  // Check positive depth
	  if(PcZ<0.0f)
	      return false;

	  // Project in image and check it is not outside
	  const float invz = 1.0f/PcZ;
	  const float u=fx*PcX*invz+cx;
	  const float v=fy*PcY*invz+cy;
	  if(u<mnMinX || u>mnMaxX)
	      return false;
	  if(v<mnMinY || v>mnMaxY)
	      return false;

	  // Check distance is in the scale invariance region of the MapPoint
	  const float maxDistance = pMP->GetMaxDistanceInvariance();
	  const float minDistance = pMP->GetMinDistanceInvariance();
	  // In the world coordinate system, the vector from the camera to the 3D point P, the vector direction is from the camera to the 3D point P
	  const cv::Mat PO = P-mOw;
	  const float dist = cv::norm(PO);
	  if(dist<minDistance || dist>maxDistance)
	      return false;

	// Check viewing angle
	  cv::Mat Pn = pMP->GetNormal();// |Pn|=1
	  const float viewCos = PO.dot(Pn)/dist;//  = P0.dot(Pn)/(|P0|*|Pn|); |P0|=dist 
	  if(viewCos < viewingCosLimit)
	      return false;

	  // Predict scale in the image
	  const int nPredictedLevel = pMP->PredictScale(dist,this);

	  // Data used by the tracking
	  pMP->mbTrackInView = true;
	  pMP->mTrackProjX = u;
	  pMP->mTrackProjXR = u - mbf*invz;
	  pMP->mTrackProjY = v;
	  pMP->mnTrackScaleLevel= nPredictedLevel;
	  pMP->mTrackViewCos = viewCos;

	  return true;
      }
      
/**
 * @brief Find the feature points in the square with x, y as the center and the side length is 2r and in [minLevel, maxLevel], 
 * and obtain the feature frame points in a certain area, 
 * among which, minLevel and maxLevel inspect the feature points from the image pyramid which layer was extracted.
 * 
 * @param x        image coordinates x
 * @param y        image coordinates y
 * @param r        side length
 * @param minLevel minimum scale
 * @param maxLevel maximum scale
 * @return         The sequence number of the feature point that satisfies the condition
 */

      vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
      {
	  vector<size_t> vIndices;
	  vIndices.reserve(N);

	  const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
	  if(nMinCellX>=FRAME_GRID_COLS)
	      return vIndices;

	  const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
	  if(nMaxCellX<0)
	      return vIndices;

	  const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
	  if(nMinCellY>=FRAME_GRID_ROWS)
	      return vIndices;

	  const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
	  if(nMaxCellY<0)
	      return vIndices;

	  const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

	  for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
	  {
	      for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
	      {
		  const vector<size_t> vCell = mGrid[ix][iy];
		  if(vCell.empty())
		      continue;

		  for(size_t j=0, jend=vCell.size(); j<jend; j++)
		  {
		      const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
		      if(bCheckLevels)
		      {
			  if(kpUn.octave<minLevel)
			      continue;
			  if(maxLevel>=0)
			      if(kpUn.octave>maxLevel)
				  continue;
		      }

		      const float distx = kpUn.pt.x-x;
		      const float disty = kpUn.pt.y-y;

		      if(fabs(distx)<r && fabs(disty)<r)
			  vIndices.push_back(vCell[j]);
		  }
	      }
	  }

	  return vIndices;
      }


      // Check if a point is in a partitioned grid
      bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
      {
	  posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
	  posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);
	  //Keypoint's coordinates are undistorted, which could cause to go out of the image
	  if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
	      return false;
	  return true;
      }
      

/**
 * @brief Bag of Words Representation
 *
 * Using words (ORB word dictionary) to linearly represent all descriptors in a frame is equivalent to a sentence represented by several words, 
 * a dictionary of N M-dimensional words
 * 
 * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
 */
      void Frame::ComputeBoW()
      {
	  if(mBowVec.empty())//Dictionary representation vector is empty
	  {
	      vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
	      // Feature vector associate features with nodes in the 4th level (from leaves up)
	      // We assume the vocabulary tree has 6 levels, change the 4 otherwise
	      mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
	  }
      }

      // Correct the points in the element image using the distortion parameters to obtain the corrected coordinates
      void Frame::UndistortKeyPoints()
      {
	  if(mDistCoef.at<float>(0)==0.0)// distortion Correction Parameters No
	  {
	      mvKeysUn=mvKeys;//Corrected and uncorrected point coordinates
	      return;
	  }
	  // Fill matrix with points
	  cv::Mat mat(N,2,CV_32F);
	  for(int i=0; i<N; i++)
	  {
	      mat.at<float>(i,0)=mvKeys[i].pt.x;
	      mat.at<float>(i,1)=mvKeys[i].pt.y;
	  }

	  // Correct key point coordinates Undistort points
	  mat=mat.reshape(2);
	  cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
	  mat=mat.reshape(1);

	  // Fill undistorted keypoint vector
	  mvKeysUn.resize(N);
	  for(int i=0; i<N; i++)
	  {
	      cv::KeyPoint kp = mvKeys[i];
	      kp.pt.x=mat.at<float>(i,0);
	      kp.pt.y=mat.at<float>(i,1);
	      mvKeysUn[i]=kp;
	  }
      }

      // For uncorrected images Calculate the size of the corrected image
      void Frame::ComputeImageBounds(const cv::Mat &imLeft)
      {
	  if(mDistCoef.at<float>(0)!=0.0)
	  {
	      cv::Mat mat(4,2,CV_32F);
	      // The four vertex positions of the image
	      mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
	      mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
	      mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
	      mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

	      // Undistort corners
	      mat=mat.reshape(2);
	      cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
	      mat=mat.reshape(1);

	      mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
	      mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
	      mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
	      mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

	  }
	  else // No Distortion Correction Parameters
	  {
	      mnMinX = 0.0f;
	      mnMaxX = imLeft.cols;
	      mnMinY = 0.0f;
	      mnMaxY = imLeft.rows;
	  }
      }
      
/**
 * @brief binocular matching
 *
 * Find matching points in the right image for each feature point in the left image
 * Find a match according to the descriptor distance on the baseline (with redundant range), and then perform SAD precise positioning
 * Finally, sort all SAD values, eliminate matching pairs with larger SAD values, and then use parabolic fitting to obtain sub-pixel precision matching
 * After the match is successful, mvuRight(ur) and mvDepth(Z) will be updated
 */
      /*
      * 1】Establish a band area search table for each feature point of the left eye, and limit the search area, (the polar line correction has been performed before)
      * 2】Match feature points through descriptors in a limited area to obtain the best matching point for each feature point（scaleduR0）   bestIdxR  uR0 = mvKeysRight[bestIdxR].pt.x;   scaleduR0 = round(uR0*scaleFactor);
      * 3】The matching correction amount is obtained through the SAD sliding window bestincR
      * 4】(bestincR, dist)  (bestincR - 1, dist)  (bestincR +1, dist) Three points are fitted to a parabola, and the sub-pixel correction amount deltaR is obtained to eliminate the matching feature points with large SAD matching deviation.
      * 5】The final matching point position is : scaleduR0 + bestincR  + deltaR
      */
      void Frame::ComputeStereoMatches()
      {
	  mvuRight = vector<float>(N,-1.0f);
	  mvDepth = vector<float>(N,-1.0f);

	  const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

	  const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
	  vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

	  for(int i=0; i<nRows; i++)
	      vRowIndices[i].reserve(200);

	  const int Nr = mvKeysRight.size();

	  for(int iR=0; iR<Nr; iR++)
	  {
	      const cv::KeyPoint &kp = mvKeysRight[iR];
	      const float &kpY = kp.pt.y;	      
	      const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
	      const int maxr = ceil(kpY+r);
	      const int minr = floor(kpY-r);

	      for(int yi=minr;yi<=maxr;yi++)
		  vRowIndices[yi].push_back(iR);
	  }

	  // Set limits for search
	  const float minZ = mb;       
	  const float minD = 0;       
	  const float maxD = mbf/minZ;  


	  // For each left keypoint search a match in the right image
	  vector<pair<int, int> > vDistIdx;
	  vDistIdx.reserve(N);
	  
	  for(int iL=0; iL<N; iL++)
	  {
	      const cv::KeyPoint &kpL = mvKeys[iL];
	      const int &levelL = kpL.octave;
	      const float &vL = kpL.pt.y;
	      const float &uL = kpL.pt.x;

	      const vector<size_t> &vCandidates = vRowIndices[vL];

	      if(vCandidates.empty())
		  continue;

	      const float minU = uL-maxD;
	      const float maxU = uL-minD;

	      if(maxU<0)
		  continue;

	      int bestDist = ORBmatcher::TH_HIGH;
	      size_t bestIdxR = 0;

	      const cv::Mat &dL = mDescriptors.row(iL);）
	      // Compare descriptor to right keypoints  
	      for(size_t iC=0; iC<vCandidates.size(); iC++)
	      {
		  const size_t iR = vCandidates[iC];
		  const cv::KeyPoint &kpR = mvKeysRight[iR];

		  if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
		      continue;

		  const float &uR = kpR.pt.x;

		  if(uR>=minU && uR<=maxU)
		  {
		      const cv::Mat &dR = mDescriptorsRight.row(iR);
		      const int dist = ORBmatcher::DescriptorDistance(dL,dR);

		      if(dist<bestDist)
		      {
			  bestDist = dist;
			  bestIdxR = iR;
		      }
		  }
	      }
	      
	      // Subpixel match by correlation
	      if(bestDist<thOrbDist)
	      {
		  // coordinates in image pyramid at keypoint scale
		  const float uR0 = mvKeysRight[bestIdxR].pt.x;
		  const float scaleFactor = mvInvScaleFactors[kpL.octave];
		  const float scaleduL = round(kpL.pt.x*scaleFactor);
		  const float scaledvL = round(kpL.pt.y*scaleFactor);
		  const float scaleduR0 = round(uR0*scaleFactor);

		  // sliding window search
		  const int w = 5;
		  cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
		  IL.convertTo(IL,CV_32F);
		  IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

		  int bestDist = INT_MAX;
		  int bestincR = 0;
		  const int L = 5;
		  vector<float> vDists;
		  vDists.resize(2*L+1);
		  const float iniu = scaleduR0+L-w;
		  const float endu = scaleduR0+L+w+1;
		  if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
		      continue;

		  for(int incR=-L; incR<=+L; incR++)
		  {
		      cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
		      IR.convertTo(IR,CV_32F);
		      IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

		      float dist = cv::norm(IL,IR,cv::NORM_L1);
		      if(dist<bestDist)
		      {
                         bestDist =  dist;
                         bestincR = incR; 
		      }

		      vDists[L+incR] = dist;
		  }

		  if(bestincR==-L || bestincR==L)
		      continue;
		  // Sub-pixel match (Parabola fitting)
		  const float dist1 = vDists[L+bestincR-1];
		  const float dist2 = vDists[L+bestincR];
		  const float dist3 = vDists[L+bestincR+1];

		  const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
		  if(deltaR<-1 || deltaR>1)
		      continue;

		  // Re-scaled coordinate
		  float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

		  float disparity = (uL-bestuR);

		  if(disparity>=minD && disparity<maxD)
		  {
		      if(disparity<=0)
		      {
			  disparity=0.01;
			  bestuR = uL-0.01;
		      }
		      mvDepth[iL]=mbf/disparity; 
		      mvuRight[iL] = bestuR;
		      vDistIdx.push_back(pair<int,int>(bestDist,iL));
		  }
	      }
	  }
     
	  sort(vDistIdx.begin(),vDistIdx.end());
	  const float median = vDistIdx[vDistIdx.size()/2].first;
	  const float thDist = 1.5f*1.4f*median; 

	  for(int i=vDistIdx.size()-1;i>=0;i--)
	  {
	      if(vDistIdx[i].first<thDist)
		  break;
	      else 
	      {
		  mvuRight[vDistIdx[i].second]=-1;
		  mvDepth[vDistIdx[i].second]=-1;
	      }
	  }
      }

      void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
      {
	  mvuRight = vector<float>(N,-1);
	  mvDepth = vector<float>(N,-1);

	  for(int i=0; i<N; i++)
	  {
	      const cv::KeyPoint &kp = mvKeys[i];
	      const cv::KeyPoint &kpU = mvKeysUn[i];

	      const float &v = kp.pt.y;
	      const float &u = kp.pt.x;

	      const float d = imDepth.at<float>(v,u);

	      if(d>0)
	      {
		  mvDepth[i] = d;
		  mvuRight[i] = kpU.pt.x-mbf/d;
	      }
	  }
      }
      
/**
 * @brief Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
 * @param  i  i-th keypoint
 * @return   3D point (relative to the world coordinate system)
 */
      // get the world coordinate system
      cv::Mat Frame::UnprojectStereo(const int &i)
      {
	  const float z = mvDepth[i];
	  if(z>0)
	  {
	    // pixel coordinate value
	      const float u = mvKeysUn[i].pt.x;
	      const float v = mvKeysUn[i].pt.y;
	    // like polar coordinates
	      const float x = (u-cx)*z*invfx;
	      const float y = (v-cy)*z*invfy;
	      cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
	      return mRwc*x3Dc+mOw;
	  }
	  else
	      return cv::Mat();
      }

} //namespace ORB_SLAM
