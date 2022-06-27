/**
* This file is part of ORB-SLAM2.
* Global/local optimization Using G2O graph optimization
* 
* http://www.cnblogs.com/luyb/p/5447497.html
* 
* Optimized objective function In the SLAM problem, several common constraints are:
* 1. The mapping relationship between 3D points and 2D features (through the projection matrix);
* 2. The transformation relationship between pose and pose (through three-dimensional rigid body transformation);
* 3. The matching relationship between two-dimensional features and two-dimensional features (through the F matrix);
* 4. Other relationships (such as similar transformation relationships in monocular).
* If we can know that some of these relationships are accurate, we can define such relationships and their corresponding residuals in g2o, 
* and gradually reduce the residual sum by iteratively optimizing the pose, so as to achieve the goal of optimizing the pose .
* 
*
* 1 Local optimization
* When a new keyframe is added to the visibility graph, the author performs a local optimization near the keyframe, as shown in the following figure.
* Pos3 is a newly added keyframe whose initial estimated pose has been obtained. At this point, 
* Pos2 is the key frame connected to Pos3, X2 is the 3D point seen by Pos3, 
* and X1 is the 3D point seen by Pos2. 
* These are all local information and participate in Bundle Adjustment together. 
* At the same time, Pos1 can also see X1, but it has no direct connection with Pos3. 
* It belongs to the local information associated with Pos3 and participates in Bundle Adjustment, 
* but the value of Pos1 remains unchanged (the pose is fixed). Pos0 and X0 do not participate in Bundle Adjustment.
*
* 2 Global optimization
* In the global optimization, all keyframes (except the first frame Pos0 (fixed pose)) and 3D points are involved in the optimization
*
*
* 3 Sim3 pose optimization at closed loop
* When loop closures are detected, the poses of the two keyframes connected by loop closures need to be optimized by Sim3 (to make their scales consistent).
* The similarity transformation matrix S12 between the two frames is optimized to minimize the projection error of the two-dimensional corresponding point (feature).
*
*/

#include "Optimizer.h"
// function optimization method
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
//Optimization methods The Levenberg–Marquardt algorithm provides numerical solutions for numerical nonlinear minimization (local minima).
// matrix decomposition solver
#include "Thirdparty/g2o/g2o/core/block_solver.h"//An implementation of a fast matrix factorization solver. Mainly from holdmod, csparse. Choose one of these first when using g2o.
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"// Matrix Linear Optimization Solver
// #include <g2o/solvers/csparse/linear_solver_csparse.h>  // csparse solver
// #include <g2o/solvers/dense/linear_solver_cholmod.h //
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"// dense matrix linear solver
// graph edge vertex type
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"// Defined vertex types 6-dimensional optimization variables such as camera pose
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"// Defined vertex types 7-dimensional optimization variables such as camera pose + depth information
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include<Eigen/StdVector>
#include "Converter.h"
#include<mutex>

namespace ORB_SLAM2
{

    // Global Optimization Map Iterations
	/**
	 * @brief    All MapPoints and keyframes in pMap are globally optimized by bundle adjustment
	 *  This global BA optimization is used in two places in this program:
	 * a.Monocular initialization： CreateInitialMapMonocular    function
	 * b.closed loop optimization： RunGlobalBundleAdjustment    function
	 * @param pMap            global map
	 * @param nIterations     number of optimization iterations
	 * @param pbStopFlag      set whether to force pause, force termination when the iteration needs to be terminated 
	 * @param nLoopKF         number of keyframes
	 * @param bRobust         whether to use kernel function robust optimization (slightly longer)
	 * 
	 */  
    void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
	vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();// keyframes of the map
	vector<MapPoint*> vpMP = pMap->GetAllMapPoints();// map point
	BundleAdjustment(vpKFs,vpMP,nIterations,pbStopFlag, nLoopKF, bRobust);
    }

    // BA Minimize Reprojection Error Keyframes Map Points Optimization Iterations
    // Optimize the pose and map point coordinates of keyframes
	/**
	 * @brief bundle adjustment Optimization
	 * 3D-2D Minimize Reprojection Error e = (u,v) - K * project(Tcw*Pw) \n
	 * 
	 * 1. Vertex: g2o::VertexSE3Expmap()，Tcw pose of the current frame
	 *            g2o::VertexSBAPointXYZ()，MapPoint is mWorldPos map point coordinates
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge binary edge
	 *         + Vertex Connected Vertex 1: Pose Tcw of the current frame to be optimized
	 *         + Vertex Connected vertex 2: mWorldPos map point coordinates of the MapPoint to be optimized
	 *         + measurement Measured value (true value)：The two-dimensional position (u, v) pixel coordinates of the MapPoint map point in the current frame
	 *         + InfoMatrix Information matrix (error weight matrix): invSigma2(It is related to the scale at which the feature points are located)
	 *         
	 * @param   vpKFs          Keyframe
	 * @param   vpMP           MapPoints
	 * @param   nIterations    Number of iterations (20 times)
	 * @param   pbStopFlag     Whether to force a suspension
	 * @param   nLoopKF        Number of keyframes
	 * @param   bRobust        Whether to use a kernel function
	 */   
    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
				    int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
	vector<bool> vbNotIncludedMP;
	vbNotIncludedMP.resize(vpMP.size());//Number of map points
	// Step 1: Initialize the g2o optimizer	
        // Step 1.1: Set the solver type frame pose pose dimension to 6 (optimization variable dimension), map point landmark dimension to 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;// pose dimension is 6 (optimization variable dimension), landmark dimension is 3
        // typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // The pose dimension is 6 (optimization variable dimension), and the landmark dimension is 3
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();// matrix solver pointer
	// linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
      	// Step 1.2: Set up the solver
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        // Step 1.2: Set up the solver
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// LM Lymar algorithm
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// Gauss Newton
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//dogleg algorithm
       // Step 1.4: Setting up the sparse optimization solver
	g2o::SparseOptimizer optimizer;// Sparse optimization model
	optimizer.setAlgorithm(solver);// Set up the solver
        //   Set whether to force termination, force termination when the iteration needs to be terminated
	if(pbStopFlag)
	    optimizer.setForceStopFlag(pbStopFlag);// Optimization stop sign

	long unsigned int maxKFid = 0;// Maximum keyframe ID
	
	// Step 2: Add Vertices to the Optimizer
	// Set KeyFrame vertices 
     	// Step 2.1: Add Keyframe Pose Vertices to the Optimizer Add Pose Vertices Set 6DOF Pose Vertices for Each Frame
	for(size_t i=0; i<vpKFs.size(); i++)
	{
	    KeyFrame* pKF = vpKFs[i];//Each keyframe in the graph
	    if(pKF->isBad())//bad frame not optimized wild frame
		continue;
	    // vertex vertex optimization variable
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//camera pose 
	    vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose())); // Optimize the initial value of the variable mat form pose Convert to SE3Quat Lie algebra form
	    vSE3->setId(pKF->mnId);// vertex id
	    vSE3->setFixed(pKF->mnId == 0);// The initial frame pose is fixed as a unit diagonal matrix.  world coordinate origin
	    optimizer.addVertex(vSE3);//add vertex
	    if(pKF->mnId > maxKFid)
		maxKFid = pKF->mnId;// Maximum keyframe ID
	}

	const float thHuber2D = sqrt(5.99);	 //  g2o is optimized to two values pixel coordinates            Time Robust Optimization Kernel Function Coefficients
	const float thHuber3D = sqrt(7.815);   //  g2o is optimized to 3 values, and the kernel function coefficients are robustly optimized when pixel coordinates + parallax

     	// Step 2.2: Add MapPoints vertices to the optimizer, add 3 DOF map points vertices
	// Set MapPoint vertices
	for(size_t i=0; i<vpMP.size(); i++)// every map point
	{
	    MapPoint* pMP = vpMP[i];// map point
	    if(pMP->isBad())//wild spot continue
		continue;
	    // g2o 3d map point type
	    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();// 3D point  landmarks
	    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));//Optimize variable initial value
	    const int id = pMP->mnId + maxKFid+1;//Set the 3d map point, the vertex in the g2o map is followed by the pose vertex id
	    vPoint->setId(id);// vertex id 
	    vPoint->setMarginalized(true);// During the optimization process, this node should be marginalized, and mar must be set in g2o
	    optimizer.addVertex(vPoint);// add vertex

	  //  const map<KeyFrame*,size_t> observations = pMP->GetObservations();
	  const auto observations = pMP->GetObservations();// Observation keyframes that can observe this map point should be connected to this map vertex
	    // The connection between map points and map points is a constraint relationship 
	    int nEdges = 0;
	    
	    // Step 3: Add to the optimizer the relationship between projected edge map points and their respective observation frames 
	    // map<KeyFrame*,size_t>::const_iterator mit 
	    for( map<KeyFrame*,size_t>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
	    {

		KeyFrame* pKF = mit->first;//A keyframe at which the point was observed
		if(pKF->isBad() || pKF->mnId > maxKFid)//This keyframe is wild or not within the optimized vertex range continue
		    continue;

		nEdges++;// edge count
		// The map point is on the corresponding observation frame, the corresponding key point pixel, coordinate
		const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];// The coordinates of the pixel point on the image corresponding to the change point of the observation frame

      		// [7] For a monocular camera, if the coordinates of the matching point in the right image are less than 0, it is a monocular
		if( pKF->mvuRight[mit->second] < 0 )
		{
		    Eigen::Matrix<double,2,1> obs;//pixel coordinates
		    obs << kpUn.pt.x, kpUn.pt.y;

		    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();// side
	      	    // Set the map point corresponding to the vertex and the observation keyframe for   camera pose
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// corresponding map point
		    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));// corresponding key frame
		    e->setMeasurement(obs);// Observations are pixel coordinates on the frame for
		    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];// The number of layers on the image pyramid
		    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//information matrix  Error weight matrix
		    // The observed value is two values, pixel coordinates, so the error weight matrix is 2*2
		    // 
    		    // 2D - 3D point pair optimization method  si * pi = K * T * Pi = K * [R t]* Pi  =  K * exp(f) * Pi  =  K * Pi'  
		    //   Pi' (map point) is the coordinate in the camera coordinate system exp(f) * Pi Front three-dimensional (Xi', Yi', Zi') exp(f) is the Lie algebra form of T
		    /*  s*u       [fx 0 cx         X'
		    *   s*v  =     0 fy cy  *     Y'
		    *   s             0 0  1]         Z'
		    *  Use the third line to eliminate s (actually the depth of P') u pixel coordinates
		    *  u = fx * X'/Z' + cx    Abscissa
		    *  v = fy * Y'/Z'  + cy   Y-axis
		    * 
		    * The p observation is the pixel coordinate on the frame, and u is the converted map point
		    *  * error e  = p - u = p -K *P' 
		    * e vs ∇f = e vs u derivative * u vs ∇f derivative = u vs ∇f derivative = u vs P'derivative * P'vs ∇f derivative         chain derivation rule
		    *
		    *  * u vs P' Partial derivative = - [ u vs X' Partial derivative u vs Y' Partial derivative u vs Z' Partial derivative;
		    *                                   v vs X' Partial derivative v vs Y' Partial derivative  v vs Z' Partial derivative]  = - [ fx/Z'   0        -fx * X'/Z' ^2 
		    *                                                                                                                        0       fy/Z'    -fy* Y'/Z' ^2]
		    *  *  P' vs ∇f Partial derivative = [ I  -P' cross product matrix] 3*6 size   pan forward  rotate behind
		    *  = [ 1 0  0   0   Z'   -Y' 
		    *       0 1  0  -Z'  0    X'
		    *       0 0  1  Y'   -X   0]
		    * There are vectors t = [ a1 a2 a3] such
		    * cross product matrix = [0  -a3  a2;
		    *                     a3  0  -a1; 
		    *                    -a2 a1  0 ]  
		    * 
		    * Multiply the two to get
		    * J = - [fx/Z'   0      -fx * X'/Z' ^2   -fx * X'*Y'/Z' ^2      fx + fx * X'^2/Z' ^2    -fx*Y'/Z'
		    *           0     fy/Z'   -fy* Y'/Z' ^2    -fy -fy* Y'^2/Z' ^2   fy * X'*Y'/Z' ^2          fy*X'/Z'    ] 
		    * If the rotation is in the front, the translation is in the back, and the first three columns are swapped and the last three columns are exchanged
		    * 
		    * [2]  Optimize the coordinate value of point P
		    * The partial derivative of e with respect to P   = e  vs P' Partial derivative *  P' vs P Partial derivative = e  vs P' Partial derivative * R
		    * P' = R * P + t
		    * P' vs P Partial derivative  = R
		    * 
		    */		
		    if(bRobust)// Robust optimization kernel function
		    {
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);// Set the robust optimization kernel function
			rk->setDelta(thHuber2D);
		    }
		    e->fx = pKF->fx;// Iteratively find the parameters required for Jacobian
		    e->fy = pKF->fy;
		    e->cx = pKF->cx;
		    e->cy = pKF->cy;
		    optimizer.addEdge(e);
		}
      		// 【9】For binocular and depth cameras      
		else
		{
		    Eigen::Matrix<double,3,1> obs;// Observation pixel coordinates and parallax
		    const float kp_ur = pKF->mvuRight[mit->second];//depth-derived parallax   disparity obtained by stereo matching
		    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

		    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();// binocular   edge type

		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// corresponding map point
		    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));// corresponding key frame
		    e->setMeasurement(obs);// observations
		    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];// error weight
		    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;// error weight
		    // The observed value is 3 values, pixel coordinates and parallax, so the error weight matrix is 3*3
		    e->setInformation(Info);// information matrix  error weight matrix

		    if(bRobust)
		    {
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);// Robust optimization kernel function
			rk->setDelta(thHuber3D);// coefficient
		    }

		    e->fx = pKF->fx;
		    e->fy = pKF->fy;
		    e->cx = pKF->cx;
		    e->cy = pKF->cy;
		    e->bf = pKF->mbf;// Parallax optimization is a required parameter

		    optimizer.addEdge(e);
		}
	    }

	    if(nEdges==0)// The number of edges is 0, and there are no map points
	    {
		optimizer.removeVertex(vPoint);
		vbNotIncludedMP[i]=true;
	    }
	    else
	    {
		vbNotIncludedMP[i]=false;
	    }
	}
	// Step 4: Start iterative optimization
	// Optimize!
	optimizer.initializeOptimization();//initialization
	optimizer.optimize(nIterations);// optimization iteration

    	// Recover optimized data
 	// Step 5: Get the optimized results From the optimized results, update the data
	//Keyframes
     	// Step 5.1: Update Frame Pose
	for(size_t i=0; i<vpKFs.size(); i++)
	{
	    KeyFrame* pKF = vpKFs[i];
	    if(pKF->isBad())
		continue;
	    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));// 位姿
	    g2o::SE3Quat SE3quat = vSE3->estimate();
	    if(nLoopKF==0)
	    {
		pKF->SetPose(Converter::toCvMat(SE3quat));// update frame pose
	    }
	    else
	    {
		pKF->mTcwGBA.create(4,4,CV_32F);
		Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
		pKF->mnBAGlobalForKF = nLoopKF;
	    }
	}
     	// Step 5.2: Update map points
	for(size_t i=0; i<vpMP.size(); i++)
	{
	    if(vbNotIncludedMP[i])
		continue;

	    MapPoint* pMP = vpMP[i];// map point

	    if(pMP->isBad())
		continue;
	    // Optimized map points
	    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));

	    if(nLoopKF==0)
	    {
		pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));// set 3d coordinates
		pMP->UpdateNormalAndDepth();// Update information such as distance from camera center
	    }
	    else
	    {
		pMP->mPosGBA.create(3,1,CV_32F);
		Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
		pMP->mnBAGlobalForKF = nLoopKF;
	    }
	}

    }


  
	/**
	 * @brief  Only optimize the pose of a single normal frame, map points are not optimized  Pose Only Optimization
	 * Track the reference key frame in the current frame, get the matching point and set the pose of the previous frame as the initial pose for optimization 
	 * The current frame tracks the local map to get the matched map points
	 * 3D-2D Minimize Reprojection Error e = (u,v) - project(Tcw*Pw) \n
	 * Only optimize the Tcw of Frame, not the coordinates of MapPoints
	 * 
	 * 1. Vertex: g2o::VertexSE3Expmap()，Tcw of the current frame
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZOnlyPose()，BaseUnaryEdge
	 *         + Vertex：Tcw of the current frame to be optimized
	 *         + measurement：The two-dimensional position of the MapPoint in the current frame (u, v)
	 *         + InfoMatrix: invSigma2(It is related to the scale where the feature points are located)
	 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()，BaseUnaryEdge
	 *         + Vertex：Tcw of the current frame to be optimized
	 *         + measurement：The 2D position of the MapPoint in the current frame(ul,v,ur)
	 *         + InfoMatrix: invSigma2(It is related to the scale where the feature points are located)
	 * @param   pFrame Frame
	 * @return  inliers number   Returns the optimal number of edges map points and the corresponding two-pixel coordinate feature points on the frame
	 */    
    int Optimizer::PoseOptimization(Frame *pFrame)
    {
      
        // Step 1: Construct the g2o optimizer      
        // Step 1.1: Set the solver type frame pose pose dimension to 6 (optimization variable dimension), and map point landmark dimension to 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
	// linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
    	// Step 1.2 Set up the solver
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    	// Step 1.3 Set the function optimization method
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// Gauss Newton
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );// dogleg algorithm	
    	// Step 1.4 Set up the sparse optimization solver
	g2o::SparseOptimizer optimizer;// Sparse optimization model
	optimizer.setAlgorithm(solver);
	int nInitialCorrespondences=0;

	// Set Frame vertex
	// Step 2: Add pose vertices Set the 6DOF pose vertices for each frame
	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));//Initial value Camera pose vertex
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);// Optimizer adds pose vertices

	// Monocular Edge Type
	const int N = pFrame->N;//  The number of map points in the frame
	vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;// Monocular Edge Container Save Edge
	vector<size_t> vnIndexEdgeMono;
	vpEdgesMono.reserve(N);
	vnIndexEdgeMono.reserve(N);

	// Binocular/Depth Edge Type
	vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;// binocular/depth edge container save edge
	vector<size_t> vnIndexEdgeStereo;
	vpEdgesStereo.reserve(N);
	vnIndexEdgeStereo.reserve(N);

	const float deltaMono = sqrt(5.991);
	const float deltaStereo = sqrt(7.815);

	// Step 3: Adding Unary Edges: Camera Projection Model
	{
	unique_lock<mutex> lock(MapPoint::mGlobalMutex);

	for(int i=0; i<N; i++)// map points per frame   
	{
	    MapPoint* pMP = pFrame->mvpMapPoints[i];//each map point on the frame
	    if(pMP)
	    {
		// Monocular observation
       		// In the case of monocular, it is also possible that under binocular, the left interest point of the current frame cannot find a matching right interest point
	        // Add only edges and corresponding map points (parameters) that optimize the pose
		if(pFrame->mvuRight[i]<0)// Right POI coordinates
		{
		    nInitialCorrespondences++;// number of sides
		    pFrame->mvbOutlier[i] = false;
	   	    // data observation pixel coordinate
		    Eigen::Matrix<double,2,1> obs;
		    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];//current frame key
		    obs << kpUn.pt.x, kpUn.pt.y;//
	   	    // Edges optimize pose only, edges are based on unary, edges connect a vertex
		    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));//   connect a vertex
		    e->setMeasurement(obs);//measurements
		    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
		    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);//error information matrix  weights

		    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		    e->setRobustKernel(rk);// Robust optimization kernel function
		    rk->setDelta(deltaMono);

		    // parameter
		    e->fx = pFrame->fx;
		    e->fy = pFrame->fy;
		    e->cx = pFrame->cx;
		    e->cy = pFrame->cy;
		    cv::Mat Xw = pMP->GetWorldPos();// true coordinate points, provided as parameters not as optimization variables
		    e->Xw[0] = Xw.at<float>(0);
		    e->Xw[1] = Xw.at<float>(1);
		    e->Xw[2] = Xw.at<float>(2);

		    optimizer.addEdge(e);

		    vpEdgesMono.push_back(e);// container preservation   side
		    vnIndexEdgeMono.push_back(i);
		}
		
       		// Binocular addition only optimizes the edges of the pose and the corresponding map points (parameters)         
		else  // Stereo observation
		{
		    nInitialCorrespondences++;// number of sides
		    pFrame->mvbOutlier[i] = false;

		    //SET EDGE
	    	    // Observations  pixel coordinates + parallax
		    Eigen::Matrix<double,3,1> obs;
		    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
		    const float &kp_ur = pFrame->mvuRight[i];
		    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;
	    	    //  edge base unary     edge base unary
		    g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
		    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 1 vertex
		    e->setMeasurement(obs);
		    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
		    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
		    // The observed value is 3 values, pixel coordinates and parallax, so the error weight matrix is 3*3
		    e->setInformation(Info);

		    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		    e->setRobustKernel(rk);// Robust optimization kernel function
		    rk->setDelta(deltaStereo);

		    // parameter
		    e->fx = pFrame->fx;
		    e->fy = pFrame->fy;
		    e->cx = pFrame->cx;
		    e->cy = pFrame->cy;
		    e->bf = pFrame->mbf;// Parameters required for parallax
		    cv::Mat Xw = pMP->GetWorldPos();// Map points as parameters not as optimization variables
		    e->Xw[0] = Xw.at<float>(0);
		    e->Xw[1] = Xw.at<float>(1);
		    e->Xw[2] = Xw.at<float>(2);

		    optimizer.addEdge(e);// add edge

		    vpEdgesStereo.push_back(e);
		    vnIndexEdgeStereo.push_back(i);
		}
	    }

	}
	}


	if(nInitialCorrespondences<3)// The number of vertex edges is less than 3 not optimized
	    return 0;

	// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
	// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
	// Step 4: Start the optimization, and optimize four times in total. After each optimization, the observations are divided into outliers and inliers, and the outliers do not participate in the next optimization.
        // Since outlier and inlier discrimination is performed on all observations after each optimization, it is possible that outliers previously judged to become inliers, and vice versa
        // Threshold calculated based on the chi-square test (assuming the measurement has a one-pixel deviation)
	const float chi2Mono[4]={5.991,5.991,5.991,5.991};
	const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
	const int its[4]={10,10,10,10};    // Optimize 10 times
	int nBad=0;
	for(size_t it=0; it<4; it++)
	{
	    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));// initial value
	    optimizer.initializeOptimization(0);// Optimize edges with level 0
	    optimizer.optimize(its[it]);

	    nBad=0;	
	    // Monocular Update Outer Point Flag
	//    if(pFrame->mvuRight[1]<0)// If the matching point coordinate is less than 0, it is monocular
	 //   {	
		    for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)// monocular each side
		    {
			g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

			const size_t idx = vnIndexEdgeMono[i];

			if(pFrame->mvbOutlier[idx])// outer point
			{
			    e->computeError();// note g2o will only calculate the error of the active edge 
			}
			const float chi2 = e->chi2();
			if(chi2 > chi2Mono[it])
			{                
			    pFrame->mvbOutlier[idx] = true;// bad point
			    e->setLevel(1);//  set outlier
			    nBad++;
			}
			else
			{
			    pFrame->mvbOutlier[idx] = false;// It turned out to be an outer point. After optimization, the error became smaller and became an inner point.
			    e->setLevel(0);// set inlier
			}

			if(it==2)
			    e->setRobustKernel(0);// Except for the first two optimizations that require RobustKernel, the rest of the optimizations do not require
		    }
	  //}
	  //  else{
		      for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
		      {
			  g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

			  const size_t idx = vnIndexEdgeStereo[i];

			  if(pFrame->mvbOutlier[idx])
			  {
			      e->computeError();
			  }

			  const float chi2 = e->chi2();

			  if(chi2>chi2Stereo[it])
			  {
			      pFrame->mvbOutlier[idx]=true;
			      e->setLevel(1);
			      nBad++;
			  }
			  else
			  {                
			      e->setLevel(0);
			      pFrame->mvbOutlier[idx]=false;
			  }

			  if(it==2)// Except for the first two optimizations that require RobustKernel, the rest of the optimizations do not require
			      e->setRobustKernel(0);
		      }
	  //  }
	    if(optimizer.edges().size()<10)
		break;
      }    

	// Recover optimized pose and return number of inliers
	// Step 5: Update frame pose after optimization
	g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
	g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
	cv::Mat pose = Converter::toCvMat(SE3quat_recov);
	pFrame->SetPose(pose);

	return nInitialCorrespondences-nBad;
    }

    
	/**
	 * @brief Local Bundle Adjustment  Local map optimization   Remove some frame/map point observation pairs with large optimization errors
	 *
	 * 1. Vertex:
	 *     - g2o::VertexSE3Expmap()，   A collection of local keyframes LocalKeyFrames , that is, the pose of the current keyframe and the pose of the keyframes connected to the current keyframe at the first level
	 *     - g2o::VertexSE3Expmap()，   Local fixed frame FixedCameras, that is, the key frame that can observe the local map point LocalMapPoints
	 *                                                  But the keyframes that do not belong to the local keyframes LocalKeyFrames, the pose of these keyframes remains unchanged in the optimization
	 *     - g2o::VertexSBAPointXYZ()， Local map point set LocalMapPoints, that is, the positions of all MapPoints that can be observed by LocalKeyFrames
	 * 
	 * 2. Edge:
	 *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge   
	 *         + Vertex：Tcw of keyframe, Pw of MapPoint
	 *         + measurement：The 2D position of the MapPoint in the keyframe (u, v)
	 *         + InfoMatrix: invSigma2(It is related to the scale at which the feature points are located)
	 *     - g2o::EdgeStereoSE3ProjectXYZ()，BaseBinaryEdge
	 *         + Vertex：Tcw of keyframe, Pw of MapPoint
	 *         + measurement：2D position of MapPoint in keyframes(ul,v,ur)
	 *         + InfoMatrix: invSigma2(It is related to the scale at which the feature points are located)
	 *         
	 * @param pKF              KeyFrame
	 * @param pbStopFlag       flag to stop optimization
	 * @param pMap             After optimization, the mutex mMutexMapUpdate of Map needs to be used when updating the state
	 */
    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
    {    
	// Local KeyFrames: First Breath Search from Current Keyframe
	list<KeyFrame*> lLocalKeyFrames;//  The first-level adjacent frame set of the local key frame set key frame
	
	// Step 1: Add the current keyframe to the local keyframe set lLocalKeyFrames
	lLocalKeyFrames.push_back(pKF);
	pKF->mnBALocalForKF = pKF->mnId;
	// Step 2: Find the keyframes connected by keyframes (one-level connected) and add them to lLocalKeyFrames	
       // Find first-level neighbors of keyframes
	const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();// first-level neighbors of keyframes
	for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
	{
	    KeyFrame* pKFi = vNeighKFs[i];// first-level neighbors of keyframes
	    pKFi->mnBALocalForKF = pKF->mnId;
	    if(!pKFi->isBad())
		lLocalKeyFrames.push_back(pKFi);// The current frame and its first-level neighbors
	}
	
	// Step 3: Traverse the keyframes in lLocalKeyFrames and add their observed MapPoints to the local map point set lLocalMapPoints
	// Local MapPoints seen in Local KeyFrames
	list<MapPoint*> lLocalMapPoints;// local map point set
	// Iterate over each local keyframe
        // list<KeyFrame*>::iterator  
	for(auto lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();// map points per local keyframe
	    // Iterate over each map point of each local keyframe
	    // vector<MapPoint*>::iterator 
	    for(auto vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
	    {
		MapPoint* pMP = *vit;// every map point at every local keyframe
		if(pMP)
		    if(!pMP->isBad())
			if(pMP->mnBALocalForKF != pKF->mnId)// avoid duplication
			{
			    lLocalMapPoints.push_back(pMP);// add to local map point set 
			    pMP->mnBALocalForKF=pKF->mnId;// mark has been added to point set
			}
	    }
	}

	//Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
	// Step 4: The key frames of the map points of the local map point set can be observed, but they are not key frames of the local key frames. These key frames are not optimized during the local BA optimization.
	list<KeyFrame*> lFixedCameras;
	// Traverse each map point in the local map point set to view its observation frame
	// list<MapPoint*>::iterator
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    map<KeyFrame*,size_t> observations = (*lit)->GetObservations();//Observation frame of local map point
	    // Traverse the observation frame of each local map point to see if it is in the local keyframe
	    // map<KeyFrame*,size_t>::iterator
	    for(auto mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKFi = mit->first;// Observation frame for each local map point
                // The local key frame does not contain the observation frame of the local map point and the local key frame is not added, to avoid repeated addition
		if(pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
		{                
		    pKFi->mnBAFixedForKF = pKF->mnId;// mark  local fixed frame
		    if(!pKFi->isBad())
			lFixedCameras.push_back(pKFi);// add  local fixed frame
		}
	    }
	}
	// Step 5: Construct the g2o optimizer
	// Setup optimizer
     	// Step 5.1: solver type frame pose pose dimension is 6 (optimization variable dimension), map point landmark dimension is 3
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
	linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
     	// Step 5.2: Iterative Optimization Algorithm  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// Gauss Newton
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );// dogleg algorithm	
     	// Step 5.3: Set up the optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
     	// Set whether to force termination, force termination when the iteration needs to be terminated
	if(pbStopFlag)
	    optimizer.setForceStopFlag(pbStopFlag);

	unsigned long maxKFid = 0;

	// Step 6: Add Vertex Local Keyframe Pose Vertices Set Local KeyFrame vertices
	//list<KeyFrame*>::iterator lit
	for(auto lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKFi = *lit;//  local keyframe
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();// vertex type
	    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));// initial value
	    vSE3->setId(pKFi->mnId);//id
	    vSE3->setFixed(pKFi->mnId == 0);// first frame keyframe fixed
	    optimizer.addVertex(vSE3);// add vertex
	    if(pKFi->mnId > maxKFid)
		maxKFid = pKFi->mnId;// The largest local keyframe
	}

	// Step 7: Add Vertices: Set Fixed Keyframe Vertices  Set Fixed KeyFrame vertices
        // list<KeyFrame*>::iterator lit
	for(auto lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKFi = *lit;// local fixed keyframe
	    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();//vertex type
	    vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
	    vSE3->setId(pKFi->mnId);
	    vSE3->setFixed(true);//the pose is fixed
	    optimizer.addVertex(vSE3);
	    if(pKFi->mnId > maxKFid)
		maxKFid = pKFi->mnId;
	}

	// Step 8: Set the map point vertex frame and each map point may be connected to form an edge
	const int nExpectedSize = ( lLocalKeyFrames.size() + lFixedCameras.size() ) * lLocalMapPoints.size();

	vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;// monocular map point    edge type 
	vpEdgesMono.reserve(nExpectedSize);
	vector<KeyFrame*> vpEdgeKFMono;// monocular keyframe
	vpEdgeKFMono.reserve(nExpectedSize);

	vector<MapPoint*> vpMapPointEdgeMono;// binocular map
	vpMapPointEdgeMono.reserve(nExpectedSize);// 
	vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;// Binocular map point edge
	vpEdgesStereo.reserve(nExpectedSize);

	vector<KeyFrame*> vpEdgeKFStereo;// binocular    keyframe
	vpEdgeKFStereo.reserve(nExpectedSize);
	vector<MapPoint*> vpMapPointEdgeStereo;// binocular map point 
	vpMapPointEdgeStereo.reserve(nExpectedSize);

	const float thHuberMono = sqrt(5.991);
	const float thHuberStereo = sqrt(7.815);
	// list<MapPoint*>::iterator lit
        // Iterate over each local map point
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    // add vertex：MapPoint
	    MapPoint* pMP = *lit;// each local map point
	    g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();// map point   vertex type
	    vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));// initial value
	    int id = pMP->mnId + maxKFid + 1;// after vertex id
	    vPoint->setId(id);
	    vPoint->setMarginalized(true);
	    optimizer.addVertex(vPoint);// add vertex

	    const map<KeyFrame*,size_t> observations = pMP->GetObservations();// the observation frame corresponding to the map point

	    // Step 9: Build edges for each pair of associated MapPoint and KeyFrame 
	    // map<KeyFrame*,size_t>::const_iterator mit
	    for(auto  mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
	    {
		KeyFrame* pKFi = mit->first;// observed keyframes for each vertex 

		if(!pKFi->isBad())
		{                
		    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];// the pixel coordinates of the observation frame corresponding to the map point

	 	    // Step 9.1：Add edge under monocular observation  Under monocular observation Observation value Pixel coordinates
		    if(pKFi->mvuRight[mit->second] < 0)
		    {
		        // observations  two-dimensional  pixel coordinates
			Eigen::Matrix<double,2,1> obs;
			obs << kpUn.pt.x, kpUn.pt.y;
		     	// binary edge
			g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();// two vertices of a binary edge
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));// map point
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));// frame pose
			e->setMeasurement(obs);// observations
			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);// error information matrix
		    	 // kernel function
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberMono);
		        // parameter information for Jacobian matrices    iterative optimization
			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;

			optimizer.addEdge(e);// add edge
			vpEdgesMono.push_back(e);// monocular side
			vpEdgeKFMono.push_back(pKFi);// keyframe
			vpMapPointEdgeMono.push_back(pMP);// map point
		    }		    
	   		// Step 9.2：under the eyes  add edge
		    else 
		    {
		    	// Observations Pixel coordinates and parallax
			Eigen::Matrix<double,3,1> obs;
			const float kp_ur = pKFi->mvuRight[mit->second];
			obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// Pixel Coordinates and Parallax
		      	// binary edge
			g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
			e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
			e->setMeasurement(obs);
			const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
			Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
			e->setInformation(Info);
		      	//  kernel function
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
			rk->setDelta(thHuberStereo);
		    	// parameter information for Jacobian matrices       iterative optimization
			e->fx = pKFi->fx;
			e->fy = pKFi->fy;
			e->cx = pKFi->cx;
			e->cy = pKFi->cy;
			e->bf = pKFi->mbf;
		    	// add edge
			optimizer.addEdge(e);
			vpEdgesStereo.push_back(e);// save edge information
			vpEdgeKFStereo.push_back(pKFi);// keyframe
			vpMapPointEdgeStereo.push_back(pMP);// map point 
		    }
		}
	    }
	}

	if(pbStopFlag)
	    if(*pbStopFlag)
		return;
	// Step 10: Start optimizing 
	optimizer.initializeOptimization();
	optimizer.optimize(5);// Optimize 5 times

	bool bDoMore= true;

	if(pbStopFlag)
	    if(*pbStopFlag)
		bDoMore = false;

	if(bDoMore) //It is necessary to eliminate the points with large errors and then optimize 10 times
	{
	    // Step 11: Detect outlier (excessive error outside point), and set the next time not to optimize
	    // Check inlier observations
	    // 更新内点 标志
	    //  if(pKFi->mvuRight[1]<0)
	    //  {  
	  
	 // 步骤11.1：遍历每一个单目 优化边
	      for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	      {
		  g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
		  MapPoint* pMP = vpMapPointEdgeMono[i];// 边对应的地图点

		  if(pMP->isBad())
		      continue;
		  // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		  if(e->chi2() > 5.991 || !e->isDepthPositive())
		  {
		      e->setLevel(1);// 不优化
		  }
		  e->setRobustKernel(0);// 不使用核函数
	      }
	      
         // 步骤11.2 : 遍历每一个双目 优化边
	      for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	      {
		  g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
		  MapPoint* pMP = vpMapPointEdgeStereo[i];

		  if(pMP->isBad())
		      continue;
                 // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
		  if(e->chi2() > 7.815 || !e->isDepthPositive())
		  {
		      e->setLevel(1);// 不优化
		  }

		  e->setRobustKernel(0);// 不使用核函数
	      }

	// Optimize again without the outliers
// 步骤12：排除误差较大的outlier后再次优化 10次
	optimizer.initializeOptimization(0);
	optimizer.optimize(10);

      }
      
 // 步骤13：在优化后重新计算误差，剔除连接误差比较大的关键帧和MapPoint
	vector<pair<KeyFrame*,MapPoint*> > vToErase;// 连接误差较大 需要 剔除的 关键帧和MapPoint
	vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());//单目边 双目边
	    // Check inlier observations 
	 // 每一个单目边  误差 两维 
	    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
	    {
		g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];//单目优化边
		MapPoint* pMP = vpMapPointEdgeMono[i];// 边对应的 地图点

		if(pMP->isBad())
		    continue;
               // 基于卡方检验计算出的阈值（误差过大，删除 假设测量有一个像素的偏差）
		if(e->chi2()>5.991 || !e->isDepthPositive())
		{
		    KeyFrame* pKFi = vpEdgeKFMono[i];//边对应的 帧
       // 步骤13.1：标记需要删除的边	    
		    vToErase.push_back(make_pair(pKFi,pMP));//删除 这个边 
		}
	    }
          // 每一个双目边  误差 三维 
	    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
	    {
		g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];//双目优化边
		MapPoint* pMP = vpMapPointEdgeStereo[i];// 边对应的 地图点

		if(pMP->isBad())
		    continue;

		if(e->chi2()>7.815 || !e->isDepthPositive())
		{
		    KeyFrame* pKFi = vpEdgeKFStereo[i];//边对应的 帧
       // 步骤13.1：标记需要删除的边	    
		    vToErase.push_back(make_pair(pKFi,pMP));//删除 这个边 
		}
	    }
	    
       //步骤13.2 删除 误差较大的 边
         // 连接偏差比较大，在关键帧中剔除对该MapPoint的观测
         // 连接偏差比较大，在MapPoint中剔除对该关键帧的观测
	 // Get Map Mutex
	unique_lock<mutex> lock(pMap->mMutexMapUpdate);
	if(!vToErase.empty())
	{
	    for(size_t i=0;i<vToErase.size();i++)
	    {
		KeyFrame* pKFi = vToErase[i].first;// 边对应的 帧
		MapPoint* pMPi = vToErase[i].second;// 边对应的 地图点
		pKFi->EraseMapPointMatch(pMPi);//帧 删除 地图点    观测
		pMPi->EraseObservation(pKFi);// 地图点 删除 观测帧 观测
	    }
	}

	// Recover optimized data
// 步骤14：优化后更新关键帧位姿以及MapPoints的位置、平均观测方向等属性
     // 步骤14.1：优化后更新 关键帧 Keyframes
	// list<KeyFrame*>::iterator
	for(auto  lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
	{
	    KeyFrame* pKF = *lit;//关键帧
	    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
	    g2o::SE3Quat SE3quat = vSE3->estimate();
	    pKF->SetPose(Converter::toCvMat(SE3quat));
	}

     //步骤14.2：优化后 更新地图点 Points
        // list<MapPoint*>::iterator
	for(auto lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
	{
	    MapPoint* pMP = *lit;//地图点
	    g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid+1));
	    pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));//更新位置
	    pMP->UpdateNormalAndDepth();// 更新 平均观测方向 深度
	}
    }

    
/**
 * @brief 闭环检测后，关键帧连接图 EssentialGraph 优化
 *
 * 1. Vertex:
 *     - g2o::VertexSim3Expmap ， Essential graph 中关键帧的位姿
 * 2. Edge:
 *     - g2o::EdgeSim3()，BaseBinaryEdge      基础二元边
 *         + Vertex： 关键帧的 位姿 Tcw， 地图点MapPoint的 位置 Pw
 *         + measurement：经过CorrectLoop函数步骤2，Sim3传播校正后的位姿
 *         + InfoMatrix: 单位矩阵     
 *
 * @param pMap                 全局地图
 * @param pLoopKF            闭环匹配上的关键帧
 * @param pCurKF              当前关键帧
 * @param NonCorrectedSim3    未经过Sim3传播调整过的关键帧 位姿 对
 * @param CorrectedSim3           经过Sim3传播调整过的关键帧 位姿 对
 * @param LoopConnections       因闭环时 MapPoints 调整而新生成的边
 * @param bFixScale                     固定尺度大小   
 */
    void Optimizer::OptimizeEssentialGraph(
					  Map* pMap, 
					  KeyFrame* pLoopKF, 
					  KeyFrame* pCurKF,
					  const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
					  const LoopClosing::KeyFrameAndPose &CorrectedSim3,
					  const map<KeyFrame *, set<KeyFrame *> > &LoopConnections, const bool &bFixScale)
    {
      
// 步骤1：构造优化器      
	// Setup optimizer
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
     // 步骤1.1：求解器类型   帧sim3位姿 spose 维度为 7  [sR t], 地图点  landmark 维度为 3
	 //  指定线性方程求解器使用Eigen的块求解器
	g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
	      new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
     // 步骤1.2：构造线性求解器
	g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
     // 步骤1.3：迭代优化算法  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// 使用LM算法进行非线性迭代
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
     // 步骤1.4：设置优化器  	
	solver->setUserLambdaInit(1e-16);
	optimizer.setAlgorithm(solver);

	const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();//全局地图 的 所有 关键帧 
	const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();//全局地图 的 所有 地图点

	const unsigned int nMaxKFid = pMap->GetMaxKFid();//最大关键帧 id
        // 仅经过Sim3传播调整，未经过优化的keyframe的pose
	vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid+1);// 存储 优化前 帧 的位姿
	// 经过Sim3传播调整，经过优化的keyframe的pose
	vector<g2o::Sim3,Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid+1);//  存储 优化后 帧 的位姿
	//  
	vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid+1);//保存g2o 顶点

	const int minFeat = 100;
	
// 步骤2：将地图中所有keyframe的pose作为顶点添加到优化器
         // 尽可能使用经过Sim3调整的位姿
        // 关键帧顶点 Set KeyFrame vertices
	for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
	{
	    KeyFrame* pKF = vpKFs[i];//关键帧
	    if(pKF->isBad())
		continue;
	    g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();// 顶点类型 sim3 相似变换
	    const int nIDi = pKF->mnId;//关键帧 id
	    LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);//查看该关键帧是否在 闭环优化帧内 
	    
       //步骤2.1： 如果该关键帧 在闭环时通过Sim3传播调整过，用校正后的位姿
	    if(it != CorrectedSim3.end())
	    {
		vScw[nIDi] = it->second;// Sim3传播调整过的位姿 
		VSim3->setEstimate(it->second);//设置 顶点初始 估计值
	    }
       //步骤2.2： 如果该关键帧在闭环时没有通过Sim3传播调整过，用自身的位姿
	    else
	    {
		Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
		Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
		g2o::Sim3 Siw(Rcw,tcw,1.0);//设置体积估计尺度为1
		vScw[nIDi] = Siw;//存储帧 对应的 sim3 位姿
		VSim3->setEstimate(Siw);//设置 顶点初始 估计值
	    }
       //步骤2.3： 闭环匹配上的帧不进行位姿优化
	    if(pKF == pLoopKF)//  闭环匹配上的关键帧
		VSim3->setFixed(true);// 固定不优化

	    VSim3->setId(nIDi);//顶点id
	    VSim3->setMarginalized(false);//
	    VSim3->_fix_scale = bFixScale; // 固定尺度大小  

	    optimizer.addVertex(VSim3);// 添加 顶点

	    vpVertices[nIDi]=VSim3;//保存顶点
	}


	set<pair<long unsigned int,long unsigned int> > sInsertedEdges;
	const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();//信息矩阵

	// Set Loop edges
// 步骤3：添加闭环新边( 帧 连接 帧 )：LoopConnections是闭环时因为MapPoints调整而出现的新关键帧连接关系（不是当前帧与闭环匹配帧之间的连接关系）	
        //  遍历  因闭环时 MapPoints 调整而新生成的边
	for(map<KeyFrame *, set<KeyFrame *> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
	{
	    KeyFrame* pKF = mit->first;//关键帧 
	    const long unsigned int nIDi = pKF->mnId;//id
	    const set<KeyFrame*> &spConnections = mit->second;// 与关键帧 相连的 关键帧
	    const g2o::Sim3 Siw = vScw[nIDi];//顶点帧 位姿
	    const g2o::Sim3 Swi = Siw.inverse();// 逆

	    for(set<KeyFrame*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit != send; sit++)
	    {
		const long unsigned int nIDj = (*sit)->mnId;// 相连关键帧 id
		if((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
		    continue;

		const g2o::Sim3 Sjw = vScw[nIDj];
		// 得到两个pose间的Sim3变换
		const g2o::Sim3 Sji = Sjw * Swi;//

		g2o::EdgeSim3* e = new g2o::EdgeSim3();// 边类型
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		e->setMeasurement(Sji);//测量值

		e->information() = matLambda;//信息矩阵

		optimizer.addEdge(e);//添加边

		sInsertedEdges.insert(make_pair(min(nIDi,nIDj),max(nIDi,nIDj)));
	    }
	}

// 步骤4：添加跟踪时形成的边、闭环匹配成功形成的边  Set normal edges
	for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
	{
	    KeyFrame* pKF = vpKFs[i];// 关键帧

	    const int nIDi = pKF->mnId;

	    g2o::Sim3 Swi;
	    
            // 尽可能得到未经过Sim3传播调整的位姿
	    LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
	    if(iti != NonCorrectedSim3.end())
		Swi = (iti->second).inverse();// 未经过Sim3传播调整的位姿
	    else
		Swi = vScw[nIDi].inverse();//  经过Sim3传播调整的位姿
		
     // 步骤4.1： 父子边  只添加扩展树的边（有父关键帧） 父关键帧<----->关键帧
	    KeyFrame* pParentKF = pKF->GetParent();
	    // Spanning tree edge
	    if(pParentKF)
	    {
		int nIDj = pParentKF->mnId;

		g2o::Sim3 Sjw;

		LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);
              // 尽可能得到未经过Sim3传播调整的位姿
		if(itj!=NonCorrectedSim3.end())
		    Sjw = itj->second;// 未经过Sim3传播调整的位姿
		else
		    Sjw = vScw[nIDj];//  经过Sim3传播调整的位姿
               // 父子 位姿 变换 
		g2o::Sim3 Sji = Sjw * Swi;

		g2o::EdgeSim3* e = new g2o::EdgeSim3();// 普通边
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		e->setMeasurement(Sji);
		e->information() = matLambda;// 信息矩阵 误差权重矩阵
		optimizer.addEdge(e);//添加边
	    }
	    
     // 步骤4.2：关键帧<---->闭环帧 添加在CorrectLoop函数中AddLoopEdge函数添加的闭环连接边（当前帧与闭环匹配帧之间的连接关系）
            // 使用经过Sim3调整前关键帧之间的相对关系作为边
	    // Loop edges
	    const set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
	    for(set<KeyFrame*>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
	    {
		KeyFrame* pLKF = *sit;
		if(pLKF->mnId<pKF->mnId)
		{
		    g2o::Sim3 Slw;

		    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);
                   // 尽可能得到未经过Sim3传播调整的位姿
		    if(itl!=NonCorrectedSim3.end())
			Slw = itl->second;
		    else
			Slw = vScw[pLKF->mnId];

		    g2o::Sim3 Sli = Slw * Swi;
		    g2o::EdgeSim3* el = new g2o::EdgeSim3();
		    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pLKF->mnId)));
		    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
		    // 根据两个Pose顶点的位姿算出相对位姿作为边 初始值 
		    el->setMeasurement(Sli);
		    el->information() = matLambda;
		    optimizer.addEdge(el);
		}
	    }
	    
       // 步骤4.3：关键帧<----->相邻帧  最有很好共视关系的关键帧也作为边进行优化
            // 使用经过Sim3调整前关键帧之间的相对关系作为边
	    // Covisibility graph edges
	    const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);// 100个相邻帧
	    for(vector<KeyFrame*>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
	    {
		KeyFrame* pKFn = *vit;// 关键帧 相邻帧
		// 非 父子帧边 无孩子  无闭环边
		if(pKFn && pKFn !=pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
		{
		    if(!pKFn->isBad() && pKFn->mnId < pKF->mnId)
		    {
			if(sInsertedEdges.count(make_pair(min(pKF->mnId,pKFn->mnId),max(pKF->mnId,pKFn->mnId))))
			    continue;

			g2o::Sim3 Snw;

			LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);
			
                        // 尽可能得到未经过Sim3传播调整的位姿
			if(itn!=NonCorrectedSim3.end())
			    Snw = itn->second;
			else
			    Snw = vScw[pKFn->mnId];

			g2o::Sim3 Sni = Snw * Swi;

			g2o::EdgeSim3* en = new g2o::EdgeSim3();
			en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFn->mnId)));
			en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
			en->setMeasurement(Sni);
			en->information() = matLambda;// 信息矩阵
			optimizer.addEdge(en);
		    }
		}
	    }
	}
// 步骤5：开始g2o优化
	// Optimize!
	optimizer.initializeOptimization();
	optimizer.optimize(20);//优化20次

	unique_lock<mutex> lock(pMap->mMutexMapUpdate);

	// SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
// 步骤6：设定帧关键帧优化后的位姿
	for(size_t i=0;i<vpKFs.size();i++)
	{
	    KeyFrame* pKFi = vpKFs[i];//关键帧

	    const int nIDi = pKFi->mnId;

	    g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
	    g2o::Sim3 CorrectedSiw =  VSim3->estimate();// 优化后的 关键帧的 sim3位姿
	    vCorrectedSwc[nIDi]=CorrectedSiw.inverse(); // 存入 优化后的位姿
	    Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
	    Eigen::Vector3d eigt = CorrectedSiw.translation();
	    double s = CorrectedSiw.scale();// 尺度

	    eigt *=(1./s); //[R t/s;0 1]
	    cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);
	    pKFi->SetPose(Tiw);// 欧式变换位姿
	}
// 步骤7：步骤5和步骤6优化得到 关键帧的位姿 后，MapPoints根据参考帧 优化前后的相对关系调整自己的位置
	// Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
	for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
	{
	    MapPoint* pMP = vpMPs[i];//地图点

	    if(pMP->isBad())
		continue;

	    int nIDr;
	    // 该MapPoint经过Sim3调整过，(LoopClosing.cpp，CorrectLoop函数，步骤2.2_
	    if(pMP->mnCorrectedByKF == pCurKF->mnId)
	    {
		nIDr = pMP->mnCorrectedReference;
	    }
	    else
	    {
	      // 通过情况下MapPoint的参考关键帧就是创建该MapPoint的那个关键帧
		KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
		nIDr = pRefKF->mnId;// 地图点的 参考帧 id
	    }

             // 得到MapPoint参考关键帧步骤5优化前的位姿
	    g2o::Sim3 Srw = vScw[nIDr];// 地图点参考帧 优化前的 位姿
	     // 得到MapPoint参考关键帧优化后的位姿
	    g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];// 地图点参考帧 优化后的 位姿

	    cv::Mat P3Dw = pMP->GetWorldPos();// 地图点 原坐标
	    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);//转成 eigen
	    // 先按优化前位姿从世界坐标系 转到 当前帧下 再按 优化后的位姿 转到世界坐标系下
	    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

	    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);//转成opencv mat
	    pMP->SetWorldPos(cvCorrectedP3Dw);//设置更新后的 坐标
	    pMP->UpdateNormalAndDepth();// 更新 地图点  平均观测方向 深度
	}
    }
    
/**
 * @brief 形成闭环时进行Sim3优化   优化 两个关键帧 之间的 Sim3变换
 *
 * 1. 顶点 Vertex:
 *     - g2o::VertexSim3Expmap()，两个关键帧的位姿
 *     - g2o::VertexSBAPointXYZ()，两个关键帧共有的MapPoints
 * 
 * 2. 边 Edge:
 *     - g2o::EdgeSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 * 
 *     - g2o::EdgeInverseSim3ProjectXYZ()，BaseBinaryEdge
 *         + Vertex：关键帧的Sim3，MapPoint的Pw
 *         + measurement：MapPoint在关键帧中的二维位置(u,v)
 *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
 *         
 * @param pKF1        KeyFrame
 * @param pKF2        KeyFrame
 * @param vpMatches1  两个关键帧的匹配关系
 * @param g2oS12          两个关键帧间的Sim3变换
 * @param th2                 核函数阈值
 * @param bFixScale       是否优化尺度，弹目进行尺度优化，双目不进行尺度优化
 */
    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
// 步骤1：初始化g2o优化器
     // 先构造求解器
	g2o::SparseOptimizer optimizer;
     // 构造线性方程求解器，Hx = -b的求解器
	// typedef BlockSolver< BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> > BlockSolverX
	g2o::BlockSolverX::LinearSolverType * linearSolver;
     // 使用dense的求解器，（常见非dense求解器有cholmod线性求解器和shur补线性求解器）
	linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
	g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
     // 使用 L-M 迭代   迭代优化算法  
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);// 使用LM算法进行非线性迭代
	// g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );// 高斯牛顿
	// g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );//狗腿算法	
     //  设置优化器  	
	optimizer.setAlgorithm(solver);

	// 相机内参数 Calibration
	const cv::Mat &K1 = pKF1->mK;
	const cv::Mat &K2 = pKF2->mK;

	// 相机位姿 Camera poses
	const cv::Mat R1w = pKF1->GetRotation();
	const cv::Mat t1w = pKF1->GetTranslation();
	const cv::Mat R2w = pKF2->GetRotation();
	const cv::Mat t2w = pKF2->GetTranslation();
	
// 步骤2：添加相似Sim3顶点
	// Set Sim3 vertex
	g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();  // sim3 顶点类型
	vSim3->_fix_scale=bFixScale;
	vSim3->setEstimate(g2oS12);// 初始估计值 两帧 之间的 相似变换
	vSim3->setId(0);//id
	vSim3->setFixed(false);// 优化Sim3顶点
	vSim3->_principle_point1[0] = K1.at<float>(0,2);// 光心横坐标 cx
	vSim3->_principle_point1[1] = K1.at<float>(1,2);// 光心纵坐标 cy
	vSim3->_focal_length1[0] = K1.at<float>(0,0);// 焦距 fx
	vSim3->_focal_length1[1] = K1.at<float>(1,1);// 焦距 fy
	vSim3->_principle_point2[0] = K2.at<float>(0,2);
	vSim3->_principle_point2[1] = K2.at<float>(1,2);
	vSim3->_focal_length2[0] = K2.at<float>(0,0);
	vSim3->_focal_length2[1] = K2.at<float>(1,1);
	optimizer.addVertex(vSim3);// 添加顶点
	
// 步骤3 ：添加 地图点 顶点
       // Set MapPoint vertices
	const int N = vpMatches1.size();// 帧2 的匹配地图点
	const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;             //pKF2对应的MapPoints到pKF1的投影
	vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;//pKF1对应的MapPoints到pKF2的投影
	vector<size_t> vnIndexEdge;

	vnIndexEdge.reserve(2*N);
	vpEdges12.reserve(2*N);
	vpEdges21.reserve(2*N);

	const float deltaHuber = sqrt(th2);

	int nCorrespondences = 0;

	for(int i=0; i<N; i++)
	{
	    if(!vpMatches1[i])
		continue;
	    
            // pMP1和pMP2是匹配的MapPoints
	    MapPoint* pMP1 = vpMapPoints1[i];// 帧1 地图点
	    MapPoint* pMP2 = vpMatches1[i];    // 帧2 地图点

	    const int id1 = 2*i+1;
	    const int id2 = 2*(i+1);

	    const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

	    if(pMP1 && pMP2)
	    {
		if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
		{
      // 步骤3.1 添加PointXYZ顶点		  
		    g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
		    cv::Mat P3D1w = pMP1->GetWorldPos();
		    cv::Mat P3D1c = R1w*P3D1w + t1w;
		    vPoint1->setEstimate(Converter::toVector3d(P3D1c));// 帧1 下的点坐标
		    vPoint1->setId(id1);// 帧1  地图点
		    vPoint1->setFixed(true);
		    optimizer.addVertex(vPoint1);

		    g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
		    cv::Mat P3D2w = pMP2->GetWorldPos();
		    cv::Mat P3D2c = R2w*P3D2w + t2w;
		    vPoint2->setEstimate(Converter::toVector3d(P3D2c));// 帧2下的点坐标
		    vPoint2->setId(id2);// 帧2 地图点
		    vPoint2->setFixed(true);
		    optimizer.addVertex(vPoint2);
		}
		else
		    continue;
	    }
	    else
		continue;

	    nCorrespondences++;
	    
// 步骤4 ： 添加两个顶点（3D点）到相机投影的边
       // 步骤4.1 ：添加 帧2 地图点 映射到 帧1特征点 的 边
	    // Set edge x1 = S12*X2
	    Eigen::Matrix<double,2,1> obs1;
	    const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
	    obs1 << kpUn1.pt.x, kpUn1.pt.y;// 帧1 特征点 的 实际值
	    
	    g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();// 边类型
	    e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));// 帧2 地图点
	    e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 帧2 到 帧1 变换顶点
	    e12->setMeasurement(obs1);// 帧1 特征点 的 实际值
	    const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
	    e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);// 信息矩阵 误差权重矩阵

	    g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
	    e12->setRobustKernel(rk1);// 核函数
	    rk1->setDelta(deltaHuber);
	    optimizer.addEdge(e12);
	    
       // 步骤4.2 ：添加 帧1 地图点 映射到 帧2特征点 的 边
	    // Set edge x2 = S21*X1
	    Eigen::Matrix<double,2,1> obs2;
	    const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
	    obs2 << kpUn2.pt.x, kpUn2.pt.y;// 帧2 特征点 的 实际值

	    g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

	    e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));// 帧1 地图点
	    e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));// 帧2 到 帧1 变换顶点
	    e21->setMeasurement(obs2);// 帧2 特征点 的 实际值
	    float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
	    e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

	    g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
	    e21->setRobustKernel(rk2);// 核函数
	    rk2->setDelta(deltaHuber);
	    optimizer.addEdge(e21);

	    vpEdges12.push_back(e12);
	    vpEdges21.push_back(e21);
	    vnIndexEdge.push_back(i);
	}
	

// 步骤5：g2o开始优化，先迭代5次	 
	optimizer.initializeOptimization();
	optimizer.optimize(5);


// 步骤6：剔除一些误差大的边
    // Check inliers
    // 进行卡方检验，大于阈值的边剔除
	int nBad=0;
	for(size_t i=0; i<vpEdges12.size();i++)
	{
	    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
	    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
	    if(!e12 || !e21)
		continue;

	    if(e12->chi2() > th2 || e21->chi2() > th2)// 误差较大 边噪声大
	    {
		size_t idx = vnIndexEdge[i];
		vpMatches1[idx]=static_cast<MapPoint*>(NULL);
		optimizer.removeEdge(e12);// 移除边
		optimizer.removeEdge(e21);
		vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
		vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
		nBad++;
	    }
	}

	int nMoreIterations;
	if(nBad > 0)
	    nMoreIterations=10;//除去外点后 再迭代10次
	else
	    nMoreIterations=5;

	if(nCorrespondences - nBad < 10)
	    return 0;
// 步骤7：再次g2o优化剔除误差大的边后剩下的边
	// Optimize again only with inliers
	optimizer.initializeOptimization();
	optimizer.optimize(nMoreIterations);
	
// 步骤8：再次进行卡方检验，统计 除去误差较大后的 内点个数
	int nIn = 0;
	for(size_t i=0; i<vpEdges12.size();i++)
	{
	    g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
	    g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
	    if(!e12 || !e21)
		continue;

	    if(e12->chi2()>th2 || e21->chi2()>th2)// 误差较大
	    {
		size_t idx = vnIndexEdge[i];
		vpMatches1[idx]=static_cast<MapPoint*>(NULL);
	    }
	    else
		nIn++;//内点个数
	}
	
// 步骤9：得到优化后的结果
	// Recover optimized Sim3
	g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
	g2oS12= vSim3_recov->estimate();

	return nIn;
    }


} //namespace ORB_SLAM
