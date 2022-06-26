/**
* This file is part of ORB-SLAM2.
* loopback detection
* For newly added key frames, perform loop closure detection (WOB binary dictionary matching detection, calculate similarity transformation through Sim3 algorithm) ---------> Closed loop correction (closed loop fusion and graph optimization)
* 
* Closed loop detection
* mlpLoopKeyFrameQueue  Key frame queue (joined by localMapping thread)
* 1】Take a frame parameter from the queue minCommons minScoreToRetain >>> mpCurrentKF
* 2】Determine if it is more than 10 frames away from the last closed-loop detection
* 3】Calculate the word band BOW match between the current frame and the connected key frame, the lowest score minScore mpCurrentKF
* 4】Detect closed loop candidate frames vpLoopConnectedKFs
* 5】Detect the continuity of candidate frames, group adjacent ones into a group, and group adjacent groups into groups (pKF, minScore)
* 6】Find non-connected and keyframes that have words in common with the current frame lKFsharingwords
* 7】Count the number of words with the most common words in the candidate frame and pKF maxcommonwords
* 8】get the threshold minCommons = 0.8 × maxcommonwords  
* 9】Filter common words greater than minCommons and words with BOW matches, and the minimum score is greater than minScore lscoreAndMatch
* a】Divide the existing ones into one group, calculate the highest score bestAccScore in the group, and get the key frame with the highest score in each group lsAccScoreAndMatch
* b】get the threshold minScoreToRetain = 0.75  ×  bestAccScore  
* c】Get the closed loop detection candidate frame
* e】Calculate the similarity transformation [sR|t] of the two stitches at the closed loop
* f】Loop closure, fusion, pose and graph optimization
*/

#include "LoopClosing.h"
#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

      LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
	  mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
	  mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
	  mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
      {
	  mnCovisibilityConsistencyTh = 3;
      }

      void LoopClosing::SetTracker(Tracking *pTracker)
      {
	  mpTracker=pTracker;
      }

      void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
      {
	  mpLocalMapper=pLocalMapper;
      }


      void LoopClosing::Run()
      {
	  mbFinished =false;

	  while(1)
	  {
	      // Check if there are keyframes in the queue
	      // The key frames in Loopclosing are sent from LocalMapping, and LocalMapping is sent from Tracking
	      // Insert keyframes into closed-loop detection queue mlpLoopKeyFrameQueue through InsertKeyFrame in LocalMapping
	      // Step 1: The key frame in the closed-loop detection queue mlpLoopKeyFrameQueue is not empty and will be processed all the time
	      if(CheckNewKeyFrames())
	      {
		  // Detect loop candidates and check covisibility consistency
		  // Step 2: Detect whether loop closure occurs and similar keyframes appear 		
		  if(DetectLoop())
		  {
		    // Compute similarity transformation [sR|t]
		    // In the stereo/RGBD case s=1
		    // Step 3: Loop closure occurs, computing similarity transformation [sR|t]
		    if(ComputeSim3())
		    {
			// Perform loop fusion and pose graph optimization
			// Step 4: Loop Closure, Fusion, Pose and Graph Optimization      
			CorrectLoop();
		    }
		  }
	      }       
	      //Step 5: Initial Reset Request
	      ResetIfRequested();

	      if(CheckFinish())
		  break;

	      usleep(5000);
	  }

	  SetFinish();
      }
      
      /**
	* @brief   Insert keyframes into the keyframe loop closure detection queue
	* @return  None
	*/
      void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
      {
	  unique_lock<mutex> lock(mMutexLoopQueue);
	  if(pKF->mnId != 0)
	      mlpLoopKeyFrameQueue.push_back(pKF);
      }
      
      /**
	* @brief Check if there are any pending loop detection frame keyframes in the list
	* @return Returns true if it exists
	*/
      bool LoopClosing::CheckNewKeyFrames()
      {
	  unique_lock<mutex> lock(mMutexLoopQueue);
	  return(!mlpLoopKeyFrameQueue.empty());
      }
      
	/**
	 * @brief    Detect whether a closed loop occurs, and a familiar key frame appears
	 * @return   Returns true if it exists
	 */
      bool LoopClosing::DetectLoop()
      {
	
	  // Step 1: Take a keyframe from the queue	
	  {
	      unique_lock<mutex> lock(mMutexLoopQueue);
	      mpCurrentKF = mlpLoopKeyFrameQueue.front();
	      mlpLoopKeyFrameQueue.pop_front();//出队
	      // Avoid that a keyframe can be erased while it is being process by this thread
	      mpCurrentKF->SetNotErase();
	  }

	  //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
	  // Step 2: If it is not long since the last closed loop (less than 10 frames), or if there are not 10 key frames in the map in total, the closed loop detection will not be performed.	  
	  if(mpCurrentKF->mnId < mLastLoopKFid+10)
	  {
	      mpKeyFrameDB->add(mpCurrentKF);//Add keyframe database
	      mpCurrentKF->SetErase();
	      return false;
	  }

	  // Compute reference BoW similarity score
	  // This is the lowest score to a connected keyframe in the covisibility graph
	  // We will impose loop candidates to have a higher similarity than this
 	  // Step 3: Traverse all co-view key frames, calculate the bow similarity score between the current key frame and each co-view key frame, and get the minimum score minScore
          // All common view keyframes for the current frame 
	  const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
	  const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;// BoW dictionary word description vector for the current frame
	  float minScore = 1;
	  // Iterate over each common view keyframe
	  for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
	  {
	      KeyFrame* pKF = vpConnectedKeyFrames[i];// each common view keyframe
	      if(pKF->isBad())
		  continue;
	      const DBoW2::BowVector &BowVec = pKF->mBowVec;// BoW dictionary word description vector for each co-view keyframe

	      float score = mpORBVocabulary->score(CurrentBowVec, BowVec);// The smaller the matching score, the more similar, the larger the greater the difference.

	      if(score < minScore)
		  minScore = score;// Minimum score minScore
	  }
	  
 	  // Step 4: Find the closed-loop candidate frame that matches the current frame with the lowest score minScore in all keyframe databases
	  // Query the database imposing the minimum score
	  vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

	  // If there are no loop candidates, just add new keyframe and return false
	  if(vpCandidateKFs.empty())// no closed loop candidate frame
	  {
	      mpKeyFrameDB->add(mpCurrentKF);//Add a new keyframe to the keyframe database
	      mvConsistentGroups.clear();// Continuity of candidate frame groups
	      mpCurrentKF->SetErase();
	      return false;
	  }

	  // For each loop candidate check consistency with previous loop candidates
	  // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
	  // A group is consistent with a previous group if they share at least a keyframe
	  // We must detect a consistent loop in several consecutive keyframes to accept it
	  // Step 5: Detect in candidate frames, candidate frames with continuity
	  // 1. Each candidate frame forms a "sub-candidate group spCandidateGroup" with the key frames connected to itself, vpCandidateKFs --> spCandidateGroup
	  // 2. Detect whether each key frame in the "sub-candidate group" exists in the "continuous group", if there is nCurrentConsistency++, put the "sub-candidate group" into the "current continuous group vCurrentConsistentGroups"
	  // 3. If nCurrentConsistency is greater than or equal to 3, then the candidate frame represented by the "sub-candidate group" passes the test and enters mvpEnoughConsistentCandidates 
	  mvpEnoughConsistentCandidates.clear();// Closed-loop frame obtained after final screening
	  // ConsistentGroup data type is pair<set<KeyFrame*>,int>
	   // ConsistentGroup.firs corresponds to the key frame in each "continuous group", and ConsistentGroup.second is the serial number of each "continuous group"
	  vector<ConsistentGroup> vCurrentConsistentGroups;//Continuity of candidate frame groups
	  vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);// Whether the sub-continuous group is continuous
       	  // Step 5.1: Traverse each closed-loop candidate frame
	  for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
	  {
	      KeyFrame* pCandidateKF = vpCandidateKFs[i];// Each closed-loop candidate frame
	      
       	      // Step 5.2: Form a "sub-candidate group" of itself and the keyframes connected to it
	      set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();// keyframe connected to self
	      spCandidateGroup.insert(pCandidateKF);// count yourself

	      bool bEnoughConsistent = false;
	      bool bConsistentForSomeGroup = false;
	      
              // Step 5.3: Traverse the previous "sub-continuous group"
	      for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
	      {
	     	   // Take a previous subconsecutive group
		  set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;
		  // Traverse each "sub-candidate group" and detect whether each key frame in the candidate group exists in the "sub-continuous group"
		  // If a frame coexists in the "sub-candidate group" and the previous "sub-consecutive group", then the "sub-candidate group" is contiguous with the "sub-consecutive group"
		  bool bConsistent = false;
		  // set<KeyFrame*>::iterator
	 	  // Step 5.4: Traverse each sub-candidate group to detect whether each key frame in the candidate group exists in the "sub-continuous group"
		  for(auto sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit != send; sit++)
		  {
		      if(sPreviousGroup.count(*sit))// There is a frame that co-exists in the "sub-candidate group" and the previous "sub-consecutive group"
		      {
			  bConsistent=true;// Then the "sub-candidate group" is contiguous with this "sub-continuous group"
			  bConsistentForSomeGroup=true;
			  break;
		      }
		  }
         	  // Step 5.5: Continuous
		  if(bConsistent)
		  {
		      int nPreviousConsistency = mvConsistentGroups[iG].second;//the previous sub-contiguous group that is contiguous with the sub-candidate group
		      int nCurrentConsistency = nPreviousConsistency + 1;//current subcontiguous group
		      if(!vbConsistentGroup[iG])// Subcontiguous group, not contiguous
		      {
			  ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);//Consecutive group number corresponding to the sub-candidate frame
			  vCurrentConsistentGroups.push_back(cg);
			  vbConsistentGroup[iG]=true; // Set the continuation flag for a continuation group
			  //this avoid to include the same group more than once
		      }
		      if(nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
		      {
			  mvpEnoughConsistentCandidates.push_back(pCandidateKF);// Closed-loop candidate frame
			  bEnoughConsistent=true; //this avoid to insert the same candidate more than once
		      }      
		      break;
		  }
	      }

	       // If the group is not consistent with any previous group insert with consistency counter set to zero
	       // Step 6: If all the key frames of the "sub-candidate group" do not exist in the "sub-consistent group", 
	       // then vCurrentConsistentGroups will be empty, so all the "sub-candidate groups" are copied to vCurrentConsistentGroups, 
	       // and finally used to update mvConsistentGroups, The counter is set to 0 and starts over
	      if(!bConsistentForSomeGroup)
	      {
		  ConsistentGroup cg = make_pair(spCandidateGroup,0);
		  vCurrentConsistentGroups.push_back(cg);
	      }
	  }

	  // Update Covisibility Consistent Groups
	  // Step 7: Update Continuous Groups 
	  mvConsistentGroups = vCurrentConsistentGroups;
	  // Add Current Keyframe to database
	  
	  // Step 8: Add the current keyframe to the keyframe database	  
	  mpKeyFrameDB->add(mpCurrentKF);

	  if(mvpEnoughConsistentCandidates.empty())
	  {
	      mpCurrentKF->SetErase();
	      return false;
	  }
	  else
	  {
	      return true;
	  }

	  mpCurrentKF->SetErase();
	  return false;
      }
      

/**
 * @brief Calculate the Sim3 transform of the current frame and the closed-loop frame, etc.
 *
 * 1. Accelerate the matching of descriptors through Bow, and use RANSAC to roughly calculate the Sim3 of the current frame and the closed-loop frame (current frame---closed-loop frame)
 * 2. According to the estimated Sim3, project the 3D points to find more matches, and calculate the more accurate Sim3 by the optimized method (current frame---closed loop frame)
 * 3. Match the MapPoints of the closed-loop frame and the keyframes connected to the closed-loop frame with the points of the current frame (current frame---closed-loop frame + connected keyframe)
 * 
 * Note that the above matching results are all stored in the member variable mvpCurrentMatchedPoints. 
 * For the actual update steps, see CorrectLoop() Step 3: Start Loop Fusion
 * 
 * 
 * Step 1: Traverse each closed-loop candidate keyframe and construct a sim3 solver
 * Step 2: Take a key frame pKF from the screened closed-loop candidate frames
 * Step 3: Match the current frame mpCurrentKF with the closed-loop candidate key frame pKF to obtain matching point pairs
 *    Step 3.1 Skip candidate closed-loop frames with few matching point pairs
 *    Step 3.2: Construct Sim3 solver from matching point pairs
 * Step 4: Iterate each candidate closed-loop key frame Sim3, use the similarity transformation solver to solve, and the similarity transformation from the candidate closed-loop key frame to the schedule frame
 * Step 5: Through the Sim3 transformation obtained in Step 4, use the sim3 transformation to match to obtain more matching points to make up for the missing matching in Step 3
 * Step 6: G2O Sim3 optimization, as long as there is a candidate frame through the solution and optimization of Sim3, it will jump out and stop the judgment of other candidate frames
 * Step 7: If there is no closed-loop matching candidate frame through the solution and optimization of Sim3, clear the candidate closed-loop key frame
 * Step 8: Take out the connected keyframes that match the keyframes in the closed loop, get their map points MapPoints and put them into mvpLoopMapPoints
 * Step 9: Match the closed loop on the key frame and the map points MapPoints of the connected key frame, project it to the current key frame for projection matching, and find more matches for the current frame
 * Step 10: Determine whether the current frame matches all the detected closed-loop keyframes with enough MapPoints	
 * Step 11: Satisfy the number of matching points > 40, find success, clear mvpEnoughConsistentCandidates
 * @return  The calculation is successful and returns true
 */
      bool LoopClosing::ComputeSim3()
      {
	  // For each consistent loop candidate we try to compute a Sim3
          // Number of closed-loop candidate frames
	  const int nInitialCandidates = mvpEnoughConsistentCandidates.size();
	  // We compute first ORB matches for each candidate
	  // If enough matches are found, we setup a Sim3Solver
	  ORBmatcher matcher(0.75,true);

	  vector<Sim3Solver*> vpSim3Solvers;//Similarity Transform Solver
	  vpSim3Solvers.resize(nInitialCandidates);// Each candidate frame has a Sim3Solver

	  vector<vector<MapPoint*> > vvpMapPointMatches;//Each candidate closed-loop key frame and the current frame will be matched and calculated
	  vvpMapPointMatches.resize(nInitialCandidates);

	  vector<bool> vbDiscarded;// Candidate closed-loop keyframes
	  vbDiscarded.resize(nInitialCandidates);

	  int nCandidates=0; //candidates with enough matches
	  // Step 1: Traverse each closed-loop candidate keyframe, construct sim3, and solver
	  for(int i=0; i<nInitialCandidates; i++)
	  {
	      // Step 2: Take a key frame pKF from the screened closed-loop candidate frames
	      KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

	      // avoid that local mapping erase it while it is being processed in this thread
	      // Prevent the KeyFrameCulling function from culling this key frame as a redundant frame in LocalMapping
	      pKF->SetNotErase();

	      if(pKF->isBad())
	      {
		  vbDiscarded[i] = true;// the candidate closed-loop frame
		  continue;
	      }
 		  // Step 3: Match the current frame mpCurrentKF with the closed-loop candidate key frame pKF to obtain matching point pairs
                 // The matching feature points between mpCurrentKF and pKF are obtained by bow acceleration, vvpMapPointMatches is the MapPoints corresponding to the matching feature points
                 // Accelerate matching through word bag to track map points of reference keyframes
	      int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);
	      
    	      // Step 3.1 Skip candidate closed-loop frames with few matching point pairs
	      if(nmatches<20)//The number of points tracked is too small, and the matching effect is not good
	      {
		  vbDiscarded[i] = true;// the candidate closed-loop frame
		  continue;
	      }
	      else
	      {
    		//Step 3.2: Construct Sim3 solver from matching point pairs 
            	// If mbFixScale is true, it is 6DoFf optimization (binocular RGBD), if it is false, it is 7DoF optimization (monocular one more scale scaling)
		 Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
		  pSolver->SetRansacParameters(0.99,20,300);// at least 20 inliers, max 300 iterations
		  vpSim3Solvers[i] = pSolver;
	      }
              // Add 1 to the number of candidate keyframes involved in Sim3 calculation
	      nCandidates++;
	  }

	  bool bMatch = false;// Used to mark whether there is a candidate frame to solve and optimize through Sim3
	  // Perform alternatively RANSAC iterations for each candidate
	  // until one is succesful or all fail
	  // Keep looping through all candidate frames, each candidate frame iterates 5 times, if no result is obtained after 5 iterations, change to the next candidate frame
	  // Until a candidate frame is successfully iterated for the first time, bMatch is true, or the total number of iterations of a candidate frame exceeds the limit, and it is directly eliminated
 	  // Step 4: Iterate over each candidate closed-loop keyframe, and the similarity transform solver solves
	  while(nCandidates > 0 && !bMatch)
	  {
	      // Iterate over each candidate closed-loop keyframe 
	      for(int i=0; i<nInitialCandidates; i++)
	      {
		  if(vbDiscarded[i])
		      continue;

		  KeyFrame* pKF = mvpEnoughConsistentCandidates[i];//Candidate closed-loop keyframe pKF

		  // Perform 5 Ransac Iterations
		  vector<bool> vbInliers;
		  int nInliers;// Number of interior points
		  bool bNoMore;// This is a local variable, initialized inside pSolver->iterate(...)

		  Sim3Solver* pSolver = vpSim3Solvers[i];//Corresponding sim3 solver
		  // Iterates up to 5 times, the returned Scm is the candidate frame pKF, to the Sim3 transformation of the current frame mpCurrentKF (T12)
		  cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

		  // If Ransac reachs max. iterations discard keyframe
                  // After n loops, each iteration is 5 times, for a total of n*5 iterations
              	  // The total number of iterations reaches the maximum limit of 300 times, and the qualified Sim3 transformation has not been obtained, and the candidate frame is eliminated.
		  if(bNoMore)
		  {
		      vbDiscarded[i]=true;
		      nCandidates--;
		  }

		  // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
		 // similar transformation
		  if(!Scm.empty())
		  {
		      vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
		      for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
		      {
			  // Save the interior point that conforms to the sim3 transformation, the map point MapPoint of the inlier
			  if(vbInliers[j])
			    vpMapPointMatches[j]=vvpMapPointMatches[i][j];
		      }
		      
		      // Step 5: Through the Sim3 transformation obtained in Step 4, use the sim3 transformation to match to obtain more matching points to make up for the missing matching in Step 3
                      // [sR t;0 1]
		      cv::Mat R = pSolver->GetEstimatedRotation();// R (R12) from candidate frame pKF to current frame mpCurrentKF
		      cv::Mat t = pSolver->GetEstimatedTranslation();// From the candidate frame pKF to t (t12) of the current frame mpCurrentKF, in the current frame coordinate system, the direction is from pKF to the current frame
		      const float s = pSolver->GetEstimatedScale();// Candidate frame pKF, the transformation scale s (s12) scaling ratio to the current frame mpCurrentKF
                
		      // Find more matches (successful closed-loop matching needs to meet enough matching feature points, there will be missing matches when using SearchByBoW for feature point matching before)
                      // Through Sim3 transformation, determine the approximate area of pKF1's feature points in pKF2, and similarly, determine the approximate area of ​​pKF2's feature points in pKF1
                     // Matching through descriptors in this area captures the missing matching feature points before pKF1 and pKF2, and updates the matching vpMapPointMatches		     
		      matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);// 7.5 Search radius parameter
		      
		      // Step 6: G2O Sim3 optimization, as long as there is a candidate frame through the solution and optimization of Sim3, it will jump out and stop the judgment of other candidate frames
                      // Convert OpenCV's Mat matrix to Eigen's Matrix type
		      g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);// Optimize initial value
		      
                	// If mbFixScale is true, it is a 6DoF optimization (binocular RGBD), if it is false, it is a 7DoF optimization (monocular multi-dimensional spatial scale)
                	// Optimize the Sim3 between the MapPoints corresponding to mpCurrentKF and pKF to obtain the optimized quantity gScm		      
		      const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

		      // If optimization is succesful stop ransacs and continue
	 	      // Finally, the number of interior points that G2O conforms to is greater than 20, and the solution is successful.
		      if(nInliers>=20)
		      {
			  bMatch = true;
			   // mpMatchedKF is the key frame detected by the final closed loop to form a closed loop with the current frame
			  mpMatchedKF = pKF;// closed-loop frame
			  // Get the Sim3 transformation from the world coordinate system to the candidate frame, Scale=1
			  g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);// T2W     
			  // Get the Sim3 transformation from the world coordinate system to the current frame after g2o optimization
			  mg2oScw = gScm*gSmw;// T1W =  T12 * T2W
			  mScw = Converter::toCvMat(mg2oScw);

			  mvpCurrentMatchedPoints = vpMapPointMatches;
			  break;// As long as one candidate frame passes the solution and optimization of Sim3, it will jump out and stop the judgment of other candidate frames.
		      }
		  }
	      }
	  }
	  
 	  // Step 7: If there is no closed-loop matching candidate frame through the solution and optimization of Sim3, clear the candidate closed-loop key frame
	  if(!bMatch)
	  {
	      for(int i=0; i<nInitialCandidates; i++)
		  mvpEnoughConsistentCandidates[i]->SetErase();
	      mpCurrentKF->SetErase();
	      return false;
	  }
	  
	  // Step 8: Take out the connected keyframes of the keyframes on the closed loop matching, get their MapPoints and put them into mvpLoopMapPoints
	  // Note the key frame on the match: mpMatchedKF
	  // Take out the MapPoints of vpLoopConnectedKFs and put them into mvpLoopMapPoints
	  // Retrieve MapPoints seen in Loop Keyframe and neighbors
	  vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();// On closed-loop matching, the connected keyframes of the keyframes
	  vpLoopConnectedKFs.push_back(mpMatchedKF);
	  mvpLoopMapPoints.clear();
	  // vector<KeyFrame*>::iterator
	  // Iterate over each closed loop keyframe and its neighbors
	  for(auto  vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
	  {
	      KeyFrame* pKF = *vit;//Each closed-loop keyframe and its neighbors
	      vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
	      for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
	      {
		  MapPoint* pMP = vpMapPoints[i];
		  if(pMP)
		  {
		      if(!pMP->isBad() && pMP->mnLoopPointForKF != mpCurrentKF->mnId)
		      {
			  mvpLoopMapPoints.push_back(pMP);// 闭环地图点 加入该点
			  // 标记该MapPoint被mpCurrentKF闭环时观测到并添加，避免重复添加
			  pMP->mnLoopPointForKF = mpCurrentKF->mnId;
		      }
		  }
	      }
	  }
	  
// 步骤9：将闭环匹配上关键帧以及相连关键帧的 地图点 MapPoints 投影到当前关键帧进行投影匹配 为当前帧查找更多的匹配
	  // 根据投影为当前帧查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
	  // 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，并根据尺度确定一个搜索区域，
	  // 根据该MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差小于TH_LOW即匹配成功，更新mvpCurrentMatchedPoints
	  // mvpCurrentMatchedPoints将用于SearchAndFuse中检测当前帧MapPoints与匹配的MapPoints是否存在冲突
	  // Find more matches projecting with the computed Sim3
	  matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);// 10匹配距离 阈值

	  // If enough matches accept Loop
	  
// 步骤10：判断当前帧 与检测出的所有闭环关键帧是否有足够多的MapPoints匹配	  
	  int nTotalMatches = 0;
	  for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
	  {
	      if(mvpCurrentMatchedPoints[i])//当前帧 关键点 找到匹配的地图点
		  nTotalMatches++;//匹配点数 ++
	  }
// 步骤11：满足匹配点对数>40 寻找成功 清空mvpEnoughConsistentCandidates
	  if(nTotalMatches >= 40)
	  {
	      for(int i=0; i<nInitialCandidates; i++)
		  if(mvpEnoughConsistentCandidates[i] != mpMatchedKF)
		      mvpEnoughConsistentCandidates[i]->SetErase();
	      return true;
	  }
	  // 没找到
	  else
	  {
	      for(int i=0; i<nInitialCandidates; i++)
		  mvpEnoughConsistentCandidates[i]->SetErase();
	      mpCurrentKF->SetErase();
	      return false;
	  }

      }

      
 
/**
 * @brief 闭环融合 全局优化
 *
 * 1. 通过求解的Sim3以及相对姿态关系，调整与 当前帧相连的关键帧 mvpCurrentConnectedKFs 位姿 
 *      以及这些 关键帧观测到的MapPoints的位置（相连关键帧---当前帧）
 * 2. 用当前帧在闭环地图点 mvpLoopMapPoints 中匹配的 当前帧闭环匹配地图点 mvpCurrentMatchedPoints  
 *     更新当前帧 之前的 匹配地图点 mpCurrentKF->GetMapPoint(i)
 * 2. 将闭环帧以及闭环帧相连的关键帧的 所有地图点 mvpLoopMapPoints 和  当前帧相连的关键帧的点进行匹配 
 * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 * 
 * mvpCurrentConnectedKFs    当前帧相关联的关键帧
 * vpLoopConnectedKFs            闭环帧相关联的关键帧        这些关键帧的地图点 闭环地图点 mvpLoopMapPoints
 * mpMatchedKF                        与当前帧匹配的  闭环帧
 * 
 * mpCurrentKF 当前关键帧    优化的位姿 mg2oScw     原先的地图点 mpCurrentKF->GetMapPoint(i)
 * mvpCurrentMatchedPoints  当前帧在闭环地图点中匹配的地图点  当前帧闭环匹配地图点 
 * 
 */ 
      void LoopClosing::CorrectLoop()
      {
	  cout << "检测到闭环 Loop detected!" << endl;

	  // Send a stop signal to Local Mapping
	  // Avoid new keyframes are inserted while correcting the loop
// 步骤0：请求局部地图停止，防止局部地图线程中InsertKeyFrame函数插入新的关键帧	  
	  mpLocalMapper->RequestStop();

	  // If a Global Bundle Adjustment is running, abort it
// 步骤1：停止全局优化  
	  if(isRunningGBA())
	  {
	      unique_lock<mutex> lock(mMutexGBA);
	      // 这个标志位仅用于控制输出提示，可忽略
	      mbStopGBA = true;//停止全局优化 

	      mnFullBAIdx++;

	      if(mpThreadGBA)
	      {
		  mpThreadGBA->detach();
		  delete mpThreadGBA;
	      }
	  }
	  
// 步骤2：等待 局部建图线程 完全停止
	  // Wait until Local Mapping has effectively stopped
	  while(!mpLocalMapper->isStopped())
	  {
	      usleep(1000);
	  }
	  
 // 步骤3：根据共视关系更新当前帧与其它关键帧之间的连接
	  // Ensure current keyframe is updated
	  mpCurrentKF->UpdateConnections();
	  
// 步骤4：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
	  // 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
	  // 通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换
	  // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
	  mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();//当前帧 的 相连 关键帧
	  mvpCurrentConnectedKFs.push_back(mpCurrentKF);//也加入自己
        // 先将 当前帧 mpCurrentKF 的Sim3变换存入，固定不动
	  KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;// 得到闭环g2o优化后各个关键帧的位姿 没有优化的位姿 
	  CorrectedSim3[mpCurrentKF] = mg2oScw;// 当前帧 对应的 sim3位姿
	  cv::Mat Twc = mpCurrentKF->GetPoseInverse();//当前帧---> 世界 


	  {
	      // Get Map Mutex
	      unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	      
    // 步骤4.1：通过位姿传播，得到Sim3调整后其它与当前帧相连关键帧的位姿（只是得到，还没有修正）
	      // vector<KeyFrame*>::iterator
	      //  遍历与当前帧相连的关键帧
	      for(auto vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
	      {
		  KeyFrame* pKFi = *vit;//与当前帧相连的关键帧

		  cv::Mat Tiw = pKFi->GetPose();// 帧位姿
                  // currentKF在前面已经添加
		  if(pKFi != mpCurrentKF)
		  {
		     // 得到当前帧到pKFi帧的相对变换
		      cv::Mat Tic = Tiw*Twc;
		      cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
		      cv::Mat tic = Tic.rowRange(0,3).col(3);
		      g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
		    // 当前帧的位姿mg2oScw 固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
		      g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;// 
		      //Pose corrected with the Sim3 of the loop closure
	   // 得到闭环g2o优化后各个关键帧的位姿
		      CorrectedSim3[pKFi]=g2oCorrectedSiw;
		  }

		  cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
		  cv::Mat tiw = Tiw.rowRange(0,3).col(3);
		  g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
		  //Pose without correction
	   // 当前帧相连关键帧，没有进行闭环g2o优化的位姿
		  NonCorrectedSim3[pKFi]=g2oSiw;
	      }
     // 步骤4.2：步骤4.1得到调整相连帧位姿后，修正这些关键帧的MapPoints
	      // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
	    // KeyFrameAndPose::iterator
         // 遍历 每一个相连帧 用优化的位姿 修正帧相关的地图点
	    for(auto mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
	      {
		  KeyFrame* pKFi = mit->first;//帧
		  g2o::Sim3 g2oCorrectedSiw = mit->second;//优化后的位姿  世界到 帧
		  g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();//帧 到 世界

		  g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];//未优化的位姿
		  
         // 遍历 帧的 每一个地图点
		  vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();//所有的地图点
		  for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
		  {
		      MapPoint* pMPi = vpMPsi[iMP];// 每一个地图点
		      
		      if(!pMPi)
			  continue;
		      if(pMPi->isBad())
			  continue;
		      if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)// 标记 防止重复修正
			  continue;

		      // Project with non-corrected pose and project back with corrected pose
	     // 将该未校正的 eigP3Dw 先从 世界坐标系映射 到 未校正的pKFi相机坐标系，然后再反映射到 校正后的世界坐标系下      
		      cv::Mat P3Dw = pMPi->GetWorldPos();
		      Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
		      Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map( g2oSiw.map(eigP3Dw) );

		      cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);//opencv mat格式
		      pMPi->SetWorldPos(cvCorrectedP3Dw);//更新地图点坐标值
		      pMPi->mnCorrectedByKF = mpCurrentKF->mnId;// 标记 防止重复修正
		      pMPi->mnCorrectedReference = pKFi->mnId;//
		      pMPi->UpdateNormalAndDepth();//更新 地图点 观测 方向等信息
		  }

		  // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
     // 步骤4.3：将Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
		  Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();//旋转矩阵 转到 旋转向量 roll pitch yaw
		  Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
		  double s = g2oCorrectedSiw.scale();//相似变换尺度

		  eigt *=(1./s); //[R t/s;0 1]

		  cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

		  pKFi->SetPose(correctedTiw);//更新关键帧 位姿 

		  // Make sure connections are updated
     // 步骤4.4：根据共视关系更新当前帧与其它关键帧之间的连接	  
		  pKFi->UpdateConnections();
	      }
	      
 // 步骤5：检查当前帧的地图点MapPoints 与闭环检测时匹配的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
	      // Start Loop Fusion
	      // Update matched map points and replace if duplicated
	      // 遍历每一个当前帧在闭环匹配时的地图点
	      for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
	      {
		  if(mvpCurrentMatchedPoints[i])
		  {
		      MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];//当前帧 在闭环检测匹配的地图点
		      MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);//当前帧之前的地图点
		  // 如果有重复的MapPoint（当前帧和匹配帧各有一个），则用闭环匹配得到的代替现有的
		      if(pCurMP)
			  pCurMP->Replace(pLoopMP);
		  // 如果当前帧没有该MapPoint，则直接添加    
		      else
		      {
			  mpCurrentKF->AddMapPoint(pLoopMP,i);//帧添加 关键点对应的 地图点
			  pLoopMP->AddObservation(mpCurrentKF,i);// 地图点 添加帧 和 其上对应的 关键点id
			  pLoopMP->ComputeDistinctiveDescriptors();//更新地图点描述子
		      }
		  }
	      }

	  }

	  // Project MapPoints observed in the neighborhood of the loop keyframe
	  // into the current keyframe and neighbors using corrected poses.
	  // Fuse duplications.
	  
// 步骤6：通过将 闭环时相连关键帧的地图点 mvpLoopMapPoints 投影到这些 当前帧相邻关键帧中，进行MapPoints检查与替换	  
	  SearchAndFuse(CorrectedSim3);

	  
// 步骤7：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
	  // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
	  map<KeyFrame*, set<KeyFrame*> > LoopConnections;//新 一级二级相关联关系
	  // vector<KeyFrame*>::iterator
   // 步骤7.1：遍历当前帧相连关键帧（一级相连）	  
	  for(auto vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
	  {
	      KeyFrame* pKFi = *vit;
   // 步骤7.2：得到与当前帧相连关键帧的相连关键帧（二级相连） 之前二级相邻关系      
	      vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

	      // Update connections. Detect new links.
   // 步骤7.3：更新一级相连关键帧的连接关系	      
	      pKFi->UpdateConnections();
   // 步骤7.4：取出该帧更新后的连接关系	 新二级相邻关系     
	      LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
	      // vector<KeyFrame*>::iterator
   // 步骤7.5：从新连接关系中 去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系      
	      for(auto vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
	      {
		  LoopConnections[pKFi].erase(*vit_prev);// 新二级相邻关系 中删除旧 二级相连关系
	      }
   // 步骤7.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
	      // vector<KeyFrame*>::iterator
	      for(auto vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
	      {
		  LoopConnections[pKFi].erase(*vit2);
	      }
	  }

	  // Optimize graph
// 步骤8：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系	  
	  Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

	  mpMap->InformNewBigChange();

	  // Add loop edge
// 步骤9：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
         // 这两句话应该放在OptimizeEssentialGraph之前，因为 OptimizeEssentialGraph 的步骤4.2中有优化，（wubo???）  
	  mpMatchedKF->AddLoopEdge(mpCurrentKF);
	  mpCurrentKF->AddLoopEdge(mpMatchedKF);

	  // Launch a new thread to perform Global Bundle Adjustment
// 步骤10：新建一个线程用于全局BA优化
       // OptimizeEssentialGraph只是优化了一些主要关键帧的位姿，这里进行全局BA可以全局优化所有位姿和MapPoints  
	  mbRunningGBA = true;
	  mbFinishedGBA = false;
	  mbStopGBA = false;
	  mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

	  // Loop closed. Release Local Mapping.
	  mpLocalMapper->Release();    

	  mLastLoopKFid = mpCurrentKF->mnId;   
      }

/**
 * @brief  通过将闭环时相连关键帧上所有的MapPoints投影到这些 关键帧 中，进行MapPoints检查与替换   
 * @param CorrectedPosesMap  闭环相邻帧 及其 对应的 sim3位姿
 */      
      void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
      {
	  ORBmatcher matcher(0.8);
// 遍历每一个 帧sim3位姿
	  for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
	  {
	    
	      KeyFrame* pKF = mit->first;//每一个相邻帧
	      g2o::Sim3 g2oScw = mit->second;//位姿
	      cv::Mat cvScw = Converter::toCvMat(g2oScw);//opencv格式
              // mvpLoopMapPoints 为闭环时 相邻关键帧上的 多有地图点
	      vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
	      
	  // 将闭环相连帧的MapPoints坐标变换到pKF帧坐标系，然后投影，检查冲突并融    
	      matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);
	      //对 相邻帧 匹配的地图点 融合更新  vpReplacePoints 是地图点的融合 

	      // Get Map Mutex
	      unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	      const int nLP = mvpLoopMapPoints.size();
	      for(int i=0; i<nLP;i++)
	      {
		  MapPoint* pRep = vpReplacePoints[i];//地图点被关键帧上的点代替的点  关键帧上的点
		  if(pRep)
		  {
		      pRep->Replace(mvpLoopMapPoints[i]);// 用mvpLoopMapPoints替换掉之前的 再替换回来？？
		  }
	      }
	  }
      }

 /**
 * @brief    请求闭环检测线程重启
 * @param nLoopKF 闭环当前帧 id
 */          
      void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
      {
	  cout << "Starting Global Bundle Adjustment" << endl;

	  int idx =  mnFullBAIdx;
	  Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

	  // Update all MapPoints and KeyFrames
	  // Local Mapping was active during BA, that means that there might be new keyframes
	  // not included in the Global BA and they are not consistent with the updated map.
	  // We need to propagate the correction through the spanning tree
	  // 更新地图点 和 关键帧
	  {
	      unique_lock<mutex> lock(mMutexGBA);
	      if(idx!=mnFullBAIdx)
		  return;

	      if(!mbStopGBA)
	      {
		  cout << "全局优化完成 Global Bundle Adjustment finished" << endl;
		  cout << "更新地图 Updating map ..." << endl;
		  mpLocalMapper->RequestStop();// 停止建图
		  // Wait until Local Mapping has effectively stopped
		  while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
		  {
		      usleep(1000);
		  }

		  // Get Map Mutex
		  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
		  
               
// 步骤1：更新关键帧  地图中所有的关键帧
		  // Correct keyframes starting at map first keyframe
		  list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
		  while(!lpKFtoCheck.empty())
		  {
		      KeyFrame* pKF = lpKFtoCheck.front();// 地图中的 关键帧
		      const set<KeyFrame*> sChilds = pKF->GetChilds();// 孩子 帧
		      cv::Mat Twc = pKF->GetPoseInverse();
		      // 遍历每一个孩子帧
		      for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
		      {
			  KeyFrame* pChild = *sit;// 每一个孩子帧
			  if(pChild->mnBAGlobalForKF != nLoopKF)//跳过 闭环发生事时的当前帧 避免重复
			  {
			      cv::Mat Tchildc = pChild->GetPose()*Twc;// 父亲帧到孩子帧
			      pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
			      pChild->mnBAGlobalForKF = nLoopKF;// 标记

			  }
			  lpKFtoCheck.push_back(pChild);
		      }

		      pKF->mTcwBefGBA = pKF->GetPose();
		      pKF->SetPose(pKF->mTcwGBA);
		      lpKFtoCheck.pop_front();
		  }
		  
// 步骤2：更新 地图点 
		  // Correct MapPoints
		  const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

		  for(size_t i=0; i<vpMPs.size(); i++)
		  {
		      MapPoint* pMP = vpMPs[i];

		      if(pMP->isBad())
			  continue;

		      if(pMP->mnBAGlobalForKF == nLoopKF)//  关键帧优化过 更新地图点
		      {
			  // If optimized by Global BA, just update
			  pMP->SetWorldPos(pMP->mPosGBA);
		      }
		      else
		      {
			  // Update according to the correction of its reference keyframe
			  KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();//地图点参考帧

			  if(pRefKF->mnBAGlobalForKF != nLoopKF)
			      continue;

			  // Map to non-corrected camera
			  cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
			  cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
			  cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

			  // Backproject using corrected camera
			  cv::Mat Twc = pRefKF->GetPoseInverse();
			  cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
			  cv::Mat twc = Twc.rowRange(0,3).col(3);

			  pMP->SetWorldPos(Rwc*Xc+twc);// 参考帧 地图点
		      }
		  }            

		  mpMap->InformNewBigChange();

		  mpLocalMapper->Release();

		  cout << "Map updated!" << endl;
	      }

	      mbFinishedGBA = true;
	      mbRunningGBA = false;
	  }
      }

     
      
/**
 * @brief    请求闭环检测线程重启
 * @param 无
 */  
      void LoopClosing::RequestReset()
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
	      usleep(5000);
	  }
      }
      
/**
 * @brief    请求闭环检测线程重启 成功 进行重启
 * @param 无
 */ 
      void LoopClosing::ResetIfRequested()
      {
	  unique_lock<mutex> lock(mMutexReset);
	  if(mbResetRequested)
	  {
	      mlpLoopKeyFrameQueue.clear();
	      mLastLoopKFid=0;
	      mbResetRequested=false;
	  }
      }

      void LoopClosing::RequestFinish()
      {
	  unique_lock<mutex> lock(mMutexFinish);
	  mbFinishRequested = true;
      }

      bool LoopClosing::CheckFinish()
      {
	  unique_lock<mutex> lock(mMutexFinish);
	  return mbFinishRequested;
      }

      void LoopClosing::SetFinish()
      {
	  unique_lock<mutex> lock(mMutexFinish);
	  mbFinished = true;
      }

      bool LoopClosing::isFinished()
      {
	  unique_lock<mutex> lock(mMutexFinish);
	  return mbFinished;
      }


} //namespace ORB_SLAM
