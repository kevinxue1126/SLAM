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
			  mvpLoopMapPoints.push_back(pMP);// Closed-loop map point, join the point
			  // Mark the MapPoint to be observed and added when it is closed by mpCurrentKF to avoid repeated additions
			  pMP->mnLoopPointForKF = mpCurrentKF->mnId;
		      }
		  }
	      }
	  }
	  
	  // Step 9: Match the closed loop on the key frame and the map points MapPoints of the connected key frame, project it to the current key frame for projection matching, and find more matches for the current frame
	  // Find more matches for the current frame based on the projection (successful closed-loop matching needs to meet enough matching feature points)
	  // According to the Sim3 transformation, each mvpLoopMapPoints is projected onto mpCurrentKF, and a search area is determined according to the scale,
	  // According to the descriptor of the MapPoint and the feature points in the area, if the matching error is less than TH_LOW, the matching is successful, and mvpCurrentMatchedPoints is updated.
	  // mvpCurrentMatchedPoints will be used in SearchAndFuse to detect whether there is a conflict between the current frame MapPoints and the matching MapPoints
	  // Find more matches projecting with the computed Sim3
	  matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

	  // If enough matches accept Loop
	  
	  // Step 10: Determine whether the current frame matches all the detected closed-loop keyframes with enough MapPoints	  
	  int nTotalMatches = 0;
	  for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
	  {
	      if(mvpCurrentMatchedPoints[i])//The key point of the current frame, find the matching map point
		  nTotalMatches++;//Match points ++
	  }
	  // Step 11: Satisfy the number of matching points > 40, find success, clear mvpEnoughConsistentCandidates
	  if(nTotalMatches >= 40)
	  {
	      for(int i=0; i<nInitialCandidates; i++)
		  if(mvpEnoughConsistentCandidates[i] != mpMatchedKF)
		      mvpEnoughConsistentCandidates[i]->SetErase();
	      return true;
	  }
	  // did not find
	  else
	  {
	      for(int i=0; i<nInitialCandidates; i++)
		  mvpEnoughConsistentCandidates[i]->SetErase();
	      mpCurrentKF->SetErase();
	      return false;
	  }

      }

      
 
	/**
	 * @brief closed loop fusion， global optimization
	 *
	 * Through the solved Sim3, and the relative attitude relationship. 
	 * Adjust the pose of the keyframe mvpCurrentConnectedKFs connected to the current frame, 
	 * and the position of the MapPoints observed by these keyframes (connected keyframe---current frame)
	 *      
	 * Use the current frame to match in the closed-loop map point mvpLoopMapPoints, 
	 *  the current frame closed-loop matching map point mvpCurrentMatchedPoints, 
	 *  update the current frame, the previous matching map point mpCurrentKF->GetMapPoint(i)
	 *     
	 * 2. Match the closed-loop frame and the keyframes connected to the closed-loop frame, all map points mvpLoopMapPoints and the points of the keyframe connected to the current frame.
	 * 3. Update the connection relationship between these frames through the matching relationship of MapPoints, that is, update the covisibility graph
	 * 4. Optimize the Essential Graph (Pose Graph), and adjust the position of MapPoints according to the optimized pose
	 * 5. Create a thread for global Bundle Adjustment
	 * 
	 * mvpCurrentConnectedKFs    keyframe associated with the current frame
	 * vpLoopConnectedKFs        Keyframes associated with closed loop frames
	 * mpMatchedKF               Closed-loop frame that matches the current frame
	 * 
	 * mpCurrentKF current keyframe, Optimized pose mg2oScw, the original map point mpCurrentKF->GetMapPoint(i)
	 * mvpCurrentMatchedPoints  The map point that the current frame matches in the closed-loop map point, the current frame's closed-loop matching map point
	 * 
	 */ 
      void LoopClosing::CorrectLoop()
      {
	  cout << "检测到闭环 Loop detected!" << endl;

	  // Send a stop signal to Local Mapping
	  // Avoid new keyframes are inserted while correcting the loop
	  // Step 0: Request the local map to stop to prevent the InsertKeyFrame function from inserting a new keyframe in the local map thread	  
	  mpLocalMapper->RequestStop();

	  // If a Global Bundle Adjustment is running, abort it
	  // Step 1: Stop Global Optimization 
	  if(isRunningGBA())
	  {
	      unique_lock<mutex> lock(mMutexGBA);
	      // This flag is only used to control the output prompt and can be ignored
	      mbStopGBA = true; 

	      mnFullBAIdx++;

	      if(mpThreadGBA)
	      {
		  mpThreadGBA->detach();
		  delete mpThreadGBA;
	      }
	  }
	  
	  // Step 2: Wait until Local Mapping has effectively stopped
	  while(!mpLocalMapper->isStopped())
	  {
	      usleep(1000);
	  }
	  
	  // Step 3: Ensure current keyframe is updated
	  mpCurrentKF->UpdateConnections();
	  
	  // Step 4: Through the pose propagation, get the poses of the keyframes connected to the current frame after Sim3 optimization, and their MapPoints
	  // The Sim transformation between the current frame and the world coordinate system has been determined and optimized in the ComputeSim3 function. 
	  // Through the relative pose relationship, the Sim3 transformation between these connected keyframes and the world coordinate system can be determined
	  // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
	  mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
	  mvpCurrentConnectedKFs.push_back(mpCurrentKF);
          // First save the Sim3 transformation of the current frame mpCurrentKF, fixed
	  KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;// Get the pose of each key frame after closed-loop g2o optimization, no optimized pose
	  CorrectedSim3[mpCurrentKF] = mg2oScw;// The sim3 pose corresponding to the current frame
	  cv::Mat Twc = mpCurrentKF->GetPoseInverse();//current frame ---> world


	  {
	      // Get Map Mutex
	      unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	      
    	      // Step 4.1: Through pose propagation, get the pose of other keyframes connected to the current frame after adjustment by Sim3 (just obtained, not yet corrected)
	      // vector<KeyFrame*>::iterator
	      //  Iterate over keyframes connected to the current frame
	      for(auto vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
	      {
		  KeyFrame* pKFi = *vit;//keyframe connected to the current frame

		  cv::Mat Tiw = pKFi->GetPose();// frame pose
                  // currentKF has been added before
		  if(pKFi != mpCurrentKF)
		  {
		      // Get the relative transformation from the current frame to the pKFi frame
		      cv::Mat Tic = Tiw*Twc;
		      cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
		      cv::Mat tic = Tic.rowRange(0,3).col(3);
		      g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
		      // The pose mg2oScw of the current frame is fixed, and other key frames get the pose adjusted by Sim3 according to the relative relationship
		      g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
		      //Pose corrected with the Sim3 of the loop closure
	   	      // Get the pose of each key frame after closed-loop g2o optimization
		      CorrectedSim3[pKFi]=g2oCorrectedSiw;
		  }

		  cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
		  cv::Mat tiw = Tiw.rowRange(0,3).col(3);
		  g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
		  //Pose without correction
	   	  // The current frame is connected to keyframes, and there is no pose for closed-loop g2o optimization
		  NonCorrectedSim3[pKFi]=g2oSiw;
	      }
     	      // Step 4.2: After getting the pose of the connected frames adjusted in Step 4.1, correct the MapPoints of these keyframes
	      // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
	     // KeyFrameAndPose::iterator
            // Traverse each connected frame and use the optimized pose to correct frame-related map points
	    for(auto mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
	      {
		  KeyFrame* pKFi = mit->first;//frame
		  g2o::Sim3 g2oCorrectedSiw = mit->second;//optimized pose
		  g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();//frame

		  g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];//unoptimized pose
		  
        	  //Iterate over each map point of the frame
		  vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();//all map points
		  for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
		  {
		      MapPoint* pMPi = vpMPsi[iMP];// every map point
		      
		      if(!pMPi)
			  continue;
		      if(pMPi->isBad())
			  continue;
		      if(pMPi->mnCorrectedByKF == mpCurrentKF->mnId)// Prevent duplicate corrections
			  continue;

		      // Project with non-corrected pose and project back with corrected pose 
		      cv::Mat P3Dw = pMPi->GetWorldPos();
		      Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
		      Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map( g2oSiw.map(eigP3Dw) );

		      cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);//opencv mat Format
		      pMPi->SetWorldPos(cvCorrectedP3Dw);//Update map point coordinate values
		      pMPi->mnCorrectedByKF = mpCurrentKF->mnId;// Prevent duplicate corrections
		      pMPi->mnCorrectedReference = pKFi->mnId;
		      pMPi->UpdateNormalAndDepth();//update map point, observation and directions, etc.
		  }

		  // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
     		  // Step 4.3: Convert Sim3 to SE3, and update the pose of keyframes according to the updated Sim3
		  Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();//Rotation Matrix to Rotation Vector 
		  Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
		  double s = g2oCorrectedSiw.scale();//Similarity Transformation Scale

		  eigt *=(1./s); //[R t/s;0 1]

		  cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

		  pKFi->SetPose(correctedTiw);//update keyframe and pose 

		  // Make sure connections are updated
     		  // Step 4.4: Update the connection between the current frame and other key frames according to the common view relationship	  
		  pKFi->UpdateConnections();
	      }
	      
 	      // Step 5: Check whether there is a conflict between the map points MapPoints of the current frame and the MapPoints matched during loop closure detection, and replace or fill the conflicting MapPoints
	      // Start Loop Fusion
	      // Update matched map points and replace if duplicated
	      // Traverse the map points of each current frame when the loop is closed
	      for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
	      {
		  if(mvpCurrentMatchedPoints[i])
		  {
		      MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];//Detects matching map points in the current frame in a closed loop
		      MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);//map point before the current frame
		      // If there are duplicate MapPoints (one for the current frame and one for the matching frame), replace the existing one with the one obtained from the closed loop match
		      if(pCurMP)
			  pCurMP->Replace(pLoopMP);
		      // If the current frame does not have the MapPoint, add it directly    
		      else
		      {
			  mpCurrentKF->AddMapPoint(pLoopMP,i);//The frame is added to the map point corresponding to the keypoint
			  pLoopMP->AddObservation(mpCurrentKF,i);// Map point, add frame and its corresponding keypoint id
			  pLoopMP->ComputeDistinctiveDescriptors();//Update map point descriptor
		      }
		  }
	      }

	  }

	  // Project MapPoints observed in the neighborhood of the loop keyframe
	  // into the current keyframe and neighbors using corrected poses.
	  // Fuse duplications.
	  
	  // Step 6: Check and replace MapPoints by projecting the map points mvpLoopMapPoints of the connected keyframes when the loop is closed to these adjacent keyframes of the current frame	  
	  SearchAndFuse(CorrectedSim3);

	  
	  // Step 7: Update the common-view connection relationship between the current keyframes, and obtain the newly obtained connection relationship due to the fusion of MapPoints when the loop is closed
	  // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
	  map<KeyFrame*, set<KeyFrame*> > LoopConnections;//New first-level, second-level relationship
	  // vector<KeyFrame*>::iterator
   	  // Step 7.1: Traverse the current frame connected keyframes (one-level connected)	  
	  for(auto vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
	  {
	      KeyFrame* pKFi = *vit;
   	      // Step 7.2: Obtain the connected key frame (secondary connection) of the keyframe connected to the current frame. The previous second-level adjacent relationship    
	      vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

	      // Update connections. Detect new links.
   	      // Step 7.3: Update the connection relationship of the first-level connected keyframes	      
	      pKFi->UpdateConnections();
  	      // Step 7.4: Take out the updated connection relationship of the frame, the new second-level adjacent relationship
	      LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
	      // vector<KeyFrame*>::iterator
   	      // Step 7.5: From the new connection relationship, remove the secondary connection relationship before the closed loop, and the remaining connection is the connection relationship obtained by the closed loop     
	      for(auto vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
	      {
		  LoopConnections[pKFi].erase(*vit_prev);// Delete the old, second-level connected relationship in the new second-level adjacent relationship
	      }
   	       // Step 7.6: Remove the first-level connection relationship before the closed loop from the connection relationship, and the remaining connection is the connection relationship obtained by the closed loop
	      // vector<KeyFrame*>::iterator
	      for(auto vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
	      {
		  LoopConnections[pKFi].erase(*vit2);
	      }
	  }

	  // Optimize graph
	  // Step 8: Perform EssentialGraph optimization, LoopConnections is the newly generated connection relationship after the closed loop is formed, excluding the connection relationship between the current frame and the closed-loop matching frame in Step 7	  
	  Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

	  mpMap->InformNewBigChange();

	  // Add loop edge
	  // Step 9: Add the edge between the current frame and the closed-loop matching frame (this connection is not optimized) 
	  mpMatchedKF->AddLoopEdge(mpCurrentKF);
	  mpCurrentKF->AddLoopEdge(mpMatchedKF);

	  // Launch a new thread to perform Global Bundle Adjustment
	  // Step 10: Create a new thread for global BA optimization
          // OptimizeEssentialGraph just optimizes the poses of some main keyframes. Performing global BA here can optimize all poses and MapPoints globally.
	  mbRunningGBA = true;
	  mbFinishedGBA = false;
	  mbStopGBA = false;
	  mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

	  // Loop closed. Release Local Mapping.
	  mpLocalMapper->Release();    

	  mLastLoopKFid = mpCurrentKF->mnId;   
      }

      /**
	* @brief  Check and replace MapPoints by projecting all MapPoints on keyframes connected to loop closure into these keyframes   
	* @param CorrectedPosesMap  Closed-loop adjacent frames and their corresponding sim3 poses
	*/      
      void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
      {
	  ORBmatcher matcher(0.8);
	  // Traverse each frame sim3 pose
	  for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
	  {
	    
	      KeyFrame* pKF = mit->first;//every adjacent frame
	      g2o::Sim3 g2oScw = mit->second;//pose
	      cv::Mat cvScw = Converter::toCvMat(g2oScw);//opencv format
              // When mvpLoopMapPoints is closed loop, there are many map points on adjacent keyframes
	      vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
	      
	      // Transform the MapPoints coordinates of the closed-loop connected frames to the pKF frame coordinate system, then project, check for conflicts and merge  
	      matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);
	      //For adjacent frames, matching map points, fusion update vpReplacePoints is the fusion of map points 
	      // Get Map Mutex
	      unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
	      const int nLP = mvpLoopMapPoints.size();
	      for(int i=0; i<nLP;i++)
	      {
		  MapPoint* pRep = vpReplacePoints[i];//The map point is replaced by the point on the keyframe, the point on the keyframe
		  if(pRep)
		  {
		      pRep->Replace(mvpLoopMapPoints[i]);// Replace the previous ones with mvpLoopMapPoints
		  }
	      }
	  }
      }

      /**
	* @brief    Request closed loop detection thread restart
	* @param nLoopKF closed loop current frame id
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
	  {
	      unique_lock<mutex> lock(mMutexGBA);
	      if(idx!=mnFullBAIdx)
		  return;

	      if(!mbStopGBA)
	      {
		  cout << "Global Bundle Adjustment finished" << endl;
		  cout << "Updating map ..." << endl;
		  mpLocalMapper->RequestStop();// stop mapping
		  // Wait until Local Mapping has effectively stopped
		  while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
		  {
		      usleep(1000);
		  }

		  // Get Map Mutex
		  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
		  
               
		  // Step 1: Update keyframes, all keyframes in the map
		  // Correct keyframes starting at map first keyframe
		  list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());
		  while(!lpKFtoCheck.empty())
		  {
		      KeyFrame* pKF = lpKFtoCheck.front();// Keyframes in the map
		      const set<KeyFrame*> sChilds = pKF->GetChilds();// kids frame
		      cv::Mat Twc = pKF->GetPoseInverse();
		      // iterate over each child frame
		      for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
		      {
			  KeyFrame* pChild = *sit;// every child frame
			  if(pChild->mnBAGlobalForKF != nLoopKF)//skip, the current frame when the closed loop happens, avoid repetition
			  {
			      cv::Mat Tchildc = pChild->GetPose()*Twc;// father frame to child frame
			      pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
			      pChild->mnBAGlobalForKF = nLoopKF;// mark

			  }
			  lpKFtoCheck.push_back(pChild);
		      }

		      pKF->mTcwBefGBA = pKF->GetPose();
		      pKF->SetPose(pKF->mTcwGBA);
		      lpKFtoCheck.pop_front();
		  }
		  
		  // Step 2: Update the map point
		  // Correct MapPoints
		  const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

		  for(size_t i=0; i<vpMPs.size(); i++)
		  {
		      MapPoint* pMP = vpMPs[i];

		      if(pMP->isBad())
			  continue;

		      if(pMP->mnBAGlobalForKF == nLoopKF)//  Keyframes have been optimized, and map points have been updated
		      {
			  // If optimized by Global BA, just update
			  pMP->SetWorldPos(pMP->mPosGBA);
		      }
		      else
		      {
			  // Update according to the correction of its reference keyframe
			  KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();//Map point reference frame

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

			  pMP->SetWorldPos(Rwc*Xc+twc);// reference frame
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
	 * @brief    Request closed loop detection thread restart
	 * @param    None
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
