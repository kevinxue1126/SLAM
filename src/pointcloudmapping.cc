/*
 * Semantic point cloud mapping  pointcloudmapping.cc class implementation function
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/io/pcd_io.h>
#include "Converter.h"

#include <boost/make_shared.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


#include <sys/time.h>
#include <unistd.h>
// timing
long getTimeUsec()
{
    
    struct timeval t;
    gettimeofday(&t,0);
    return (long)((long)t.tv_sec*1000*1000 + t.tv_usec);
}


// Rewrite Cluster's equals operator to facilitate searching by name
bool Cluster::operator ==(const std::string &x){
    return(this->object_name == x);
} 

// class constructor
PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;// Point cloud voxel grid size
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >(); // Global point cloud map shared pointer

    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );// Visualization thread Shared pointer Bind viewer() function
    //map_state = 0;
    //showThread   = make_shared<thread>( bind(&PointCloudMapping::update, this ) );//  The dizzy update shows the binding update() function
    
    // Different colors correspond to different objects
    // std::vector<cv::Scalar> colors;
    // colors_ptr = std::make_shared< std::vector<cv::Scalar> >();
    for (int i = 0; i < 21; i++) // background
    { // voc dataset 20 types of objects
        //colors_ptr->push_back(cv::Scalar( i*10 + 40, i*10 + 40, i*10 + 40));
       colors_.push_back(cv::Scalar( i*10 + 40, i*10 + 40, i*10 + 40));
// "background","aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair",
// "cow", "diningtable", "dog", "horse","motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"
    }
    colors_[5] = cv::Scalar(255,0,255); // bottle pink 
    colors_[9] = cv::Scalar(255,0,0);   // chair blue
    colors_[15] = cv::Scalar(0,0,255);  // people red
    colors_[20] = cv::Scalar(0,255,0);  // monitor green
    
    // object size
    for (int i = 0; i < 21; i++)  
    { // voc dataset 20 types of objects
      obj_size_.push_back(0.6);
    }
    obj_size_[5] = 0.06;  //  the bottle is considered to be the same object within 0.06 meters
    obj_size_[9] = 0.5;   //  chair
    obj_size_[15] = 0.35; //  people  
    obj_size_[20] = 0.25; //  monitor

    ncnn_detector_ptr = std::make_shared<Detector>();

   // statistical filter
   stat.setMeanK (50);	     	    
   stat.setStddevMulThresh (1.0);   
 
   // Spotlight Visualizer
   pcl_viewer_prt = std::make_shared<pcl::visualization::PCLVisualizer>();
   pcl_viewer_prt->setBackgroundColor(0.0, 0.0, 0.0);// background is black
   pcl_viewer_prt->setCameraPosition(
        0, 0, 0,                                // camera position perspective
        0, 0, 3,                                // view vector : View in meters
        0, -1, 0                                // Swap up vector in y direction
        );
    //pcl_viewer_prt->resetCamera();
    //pcl_viewer_prt->initCameraParameters ();
   //pcl_viewer_prt->addCoordinateSystem(0.5);
   //pcl_viewer_prt->setPosition(0, 0);
    pcl_viewer_prt->setSize(1280, 640);
    //pcl_viewer_prt->setSize(1280, 960);
    //pcl::PCDWriter pcdwriter;

   // point cloud saver
   pcd_writer_ptr = std::make_shared<pcl::PCDWriter>();
   
   map_state_ok = 0;
}

// Class shutdown function, similar to class destructor
void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);// execute shutdown thread
        shutDownFlag = true;
        keyFrameUpdated.notify_one();// Unblocks one of the threads waiting for the keyFrameUpdated condition variable object
    }
    viewerThread->join();// Wait for the visualization thread and return after the end
    //showThread->join();
}

// Insert a keyframe into the map
void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& imgRGB)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex); // insert a keyframe into the map
    keyframes.push_back( kf );             // add a keyframe to the keyframe array
    colorImgs.push_back( color.clone() );  // add image array to a deep copy of image
    depthImgs.push_back( depth.clone() );  // deep data array join deep copy
    RGBImgs.push_back( imgRGB.clone() );   // array of colormaps

    map_state_ok = 0;
    keyFrameUpdated.notify_one(); // Keyframe new thread, unblocks one for keyframe update and triggers point cloud visualization thread
    
}

// Calculate the point cloud of one frame according to the camera parameters and pixel map (rgb three-color) and depth map in the key frame (transform to the world coordinate system using the current key frame pose)
pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() ); // One frame point cloud shared pointer
/*
    // point cloud is null ptr    
    for ( int m=0; m<depth.rows; m+=3 )     
    {
        for ( int n=0; n<depth.cols; n+=3 )  
        {
            float d = depth.ptr<float>(m)[n];
            //std::cout << d << "\t" << std::endl;
            //if (d < 500 || d>6000)       

            if (d < 0.5 || d > 6) 
                continue;
            PointT p;
            p.z = d; 
            p.x = ( n - kf->cx) * p.z / kf->fx; 
            p.y = ( m - kf->cy) * p.z / kf->fy; 

            p.b = color.ptr<uchar>(m)[n*3+0]; 
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }
*/

        float* dep  = (float*)depth.data;// GrabImageRGBD() has been converted to CV_32F by dividing by 1000
        unsigned char* col= (unsigned char*)color.data;
        tmp->resize(color.rows * color.cols);
	tmp->width    =  color.cols;  
	tmp->height   =  color.rows;// ordered point cloud
        tmp->is_dense = false;// Non-dense point cloud, there will be bad points, may contain values ​​such as inf/NaN
        
        //for(unsigned int i = 0; i < cloud->points.size(); i++)
         //{
   #pragma omp parallel for   // omp multithreading parallel processing
        for(int r=0; r<color.rows; r++ ) // y
        {
         for(int c=0; c<color.cols; c++) // x
         {
            int i = r*color.cols + c;// total index
            float d = dep[i];
	    tmp->points[i].x = ( c - kf->cx) * d / kf->fx; // (x-cx)*z/fx
	    tmp->points[i].y = ( r - kf->cy) * d / kf->fy; // (y-cy)*z/fy
	    tmp->points[i].z = d;
	    tmp->points[i].r = col[i*3+2];
	    tmp->points[i].g = col[i*3+1];
            tmp->points[i].b = col[i*3+0];
         }
        }
            //cloud->points[i].a = 0.5;// translucent
        //}

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );// current keyframe pose four elements
    PointCloud::Ptr cloud(new PointCloud);// point cloud transformed to world coordinate system
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());// The point cloud under the current frame is transformed to the world coordinate system
    cloud->is_dense = false;// Non-dense point cloud, there will be bad point nan values, etc.

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::draw_rect_with_depth_threshold(
                                    cv::Mat bgr_img, 
                                    cv::Mat depth_img, 
                                    const cv::Rect_<float>& rect, const cv::Scalar& scalar,
                                    pcl::PointIndices & indices)
{

 unsigned char* color = (unsigned char*)bgr_img.data; //3 aisle
 float* depth = (float*)depth_img.data;              //1 aisle
 int beg = (int)rect.x + ((int)rect.y-1)*bgr_img.cols - 1;// 2d box start point

// 1. Calculate the average depth
 int count = 0;
 float sum = 0;
 for(int k=(int)rect.height*0.3; k<(int)rect.height*0.7; k++) // each line
 {   
   int start = beg + k*bgr_img.cols; // starting point
   int end   = start + (int)rect.width*0.7;// end
   for(int j = start + (int)rect.width*0.3 ; j<end; j++)//each column
   { 
     float d = depth[j];
     if (d < 0.5 || d > 6) // This has been converted to meters
         continue;
     sum += d;
     count++;
   }
 }
 float depth_threshold = 0.0;
 if(count>0) depth_threshold = sum / count;// average depth

// 2. Set the color value of the corresponding image roi according to the depth value
 for(int k=0; k< (int)rect.height-1; k++) // each line
 {   
   int start = beg + k*bgr_img.cols; // starting point
   int end   = start + (int)rect.width-1;// end
   for(int j=start; j<end-1; j++)//each column
   {
       if( abs(depth[j] - depth_threshold)<0.4 )// Objects with a difference of 0.4m from the average depth are considered to be detected objects
       {
        indices.indices.push_back (j); // record the point cloud index
        color[j*3+0] = scalar.val[0];  // red  
        color[j*3+1] = scalar.val[1];  // green
        color[j*3+2] = scalar.val[2];  // blue
       } 
   }
 }

}

void PointCloudMapping::sem_merge(Cluster cluster)
{
    // 1. Check the total number, if the database is empty, join directly
    int num_objects = clusters.size();
    if(num_objects==0)
    {
        clusters.push_back(cluster);
        return;
    }
    else
    {
        // 2. The object already exists in the database, find out whether the new object already exists in the database
	std::vector<Cluster>::iterator iter   = clusters.begin()-1;
	std::vector<Cluster>::iterator it_end = clusters.end(); 
        std::vector< std::vector<Cluster>::iterator> likely_obj;// iterator of objects with the same name
	while(true) 
        {
	    iter = find(++iter, clusters.end(), cluster.object_name);// Find by name
	    if (iter != it_end )// find one, store it
                likely_obj.push_back(iter);
	    else//can't find it
	        break;  
	}

        // 3. If not found, add it directly to the database
        std::vector<Cluster>::iterator best_close;// most recent index
        float center_distance=100;// corresponding distance
        if(likely_obj.size()==0)
        {
            clusters.push_back(cluster);
            return;
        }
        else//Find multiple objects with the same name as the database
        {
        // 4. Iterate through each object with the same name and find the one closest to the center point
            for(int j=0; j<likely_obj.size(); j++)
            {
                std::vector<Cluster>::iterator& temp_iter = likely_obj[j];
                Cluster& temp_cluster = *temp_iter;
                Eigen::Vector3f dis_vec = cluster.centroid - temp_cluster.centroid;// center point connection vector
                float dist = dis_vec.norm();
                if(dist < center_distance)
                {
                    center_distance = dist; // shortest distance
                    best_close      = temp_iter;// the corresponding index
                }
            }
        
        }

        // 5. If the distance is smaller than the object size, it is considered to be the same object in the same space, and the information of the object in the database is updated
        if(center_distance < obj_size_[cluster.class_id])
            // This scale has different values for different objects, you can set an array to store
        {
            //Cluster& best_cluster = *best_close;
            best_close->prob = (best_close->prob + cluster.prob)/2.0; // Comprehensive confidence
            best_close->centroid = (best_close->centroid + cluster.centroid)/2.0; // Center average
            // minimum
            float min_x = best_close->minPt[0] > cluster.minPt[0] ? cluster.minPt[0] : best_close->minPt[0];
            float min_y = best_close->minPt[1] > cluster.minPt[1] ? cluster.minPt[1] : best_close->minPt[1];
            float min_z = best_close->minPt[2] > cluster.minPt[2] ? cluster.minPt[2] : best_close->minPt[2];
            // maximum
            float max_x = best_close->maxPt[0] > cluster.maxPt[0] ? cluster.maxPt[0] : best_close->maxPt[0];
            float max_y = best_close->maxPt[1] > cluster.maxPt[1] ? cluster.maxPt[1] : best_close->maxPt[1];
            float max_z = best_close->maxPt[2] > cluster.maxPt[2] ? cluster.maxPt[2] : best_close->maxPt[2];
            // update
            best_close->minPt = Eigen::Vector3f(min_x,min_y,min_z);
            best_close->maxPt = Eigen::Vector3f(max_x,max_y,max_z);
        }
        else 
        {
        // 6. If the distance exceeds the size of the object, it is considered to be the same object in different positions, and it is directly put into the database
            clusters.push_back(cluster);
        }
    }
    return; 
}

void PointCloudMapping::add_cube(void)
{
    pcl_viewer_prt->removeAllShapes();// Remove previously displayed shapes
    for(int i=0; i<clusters.size(); i++)
    {
        Cluster& cluster  = clusters[i];
        std::string& name = cluster.object_name;  // object class name
        int&   class_id   = cluster.class_id;     // class id
        float& prob = cluster.prob;               // Confidence
	Eigen::Vector3f& minPt = cluster.minPt;   // The smallest x value, y value, z value among all points
	Eigen::Vector3f& maxPt = cluster.maxPt;   // The largest x value, y value, z value among all points
	Eigen::Vector3f& centr = cluster.centroid;// point cloud center point
        Eigen::Vector3f boxCe = (maxPt + minPt)*0.5f; // box center
        Eigen::Vector3f boxSi = maxPt - minPt;        // box size
          
	fprintf(stderr, "3d %s %.5f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
		        name.c_str(), prob, centr[0], centr[1], centr[2], 
                        maxPt[0], maxPt[1], maxPt[2], minPt[0], minPt[1], minPt[2]);
               // Print name, confidence, center point coordinates
	const Eigen::Quaternionf quat(Eigen::Quaternionf::Identity());// Attitude   Four Elements
	std::string name_new = name + boost::chrono::to_string(i);    // the name of the bounding box
	pcl_viewer_prt->addCube(boxCe, quat, 
                                boxSi[0], boxSi[1], boxSi[2], name_new.c_str()); // add box
	pcl_viewer_prt->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
	                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 
			name_new.c_str());
	pcl_viewer_prt->setShapeRenderingProperties(
                        pcl::visualization::PCL_VISUALIZER_COLOR,
                        colors_[class_id].val[2]/255.0, 
                        colors_[class_id].val[1]/255.0, colors_[class_id].val[0]/255.0,// color
			name_new.c_str());//       
    }
} 



// Visualize the point cloud formed by all saved keyframes
void PointCloudMapping::viewer()
{
    //pcl::PCDWriter pcdwriter;
    //pcl::visualization::CloudViewer viewer("viewer"); 
    //while(!pcl_viewer_prt->wasStopped ()) 
    int count= 0;
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex ); // close lock
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            // Update point cloud
            std::cout<< count << std::endl;
            if(count >= 1000){
              pcl_viewer_prt->spinOnce(100);// A segmentation fault will also occur pcl_viewer_prt will not be initialized in the future
              count = 1000;
            }
            count++;

            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex ); // keyframe update lock
            keyFrameUpdated.wait( lck_keyframeUpdated );// block

        }

        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );// keyframe lock
            N = keyframes.size();                   // The number of keyframes currently saved
            std::cout << "KeyframeSize: " << N << std::endl;
        }

        for ( size_t i=lastKeyframeSize; i<N ; i++ )// Continue adding point clouds to the map from the last keyframe that has been visualized
        {
            long time = getTimeUsec();// start the timer
            std::vector<Object> objects;
            ncnn_detector_ptr->Run(RGBImgs[i], objects); // Perform object detection on color map to get results

            if(objects.size()>0)
                std::cout<< "detect first obj: " << objects[0].object_name << std::endl;

            std::vector<pcl::PointIndices> vec_indices; // Point cloud index corresponding to each object
            std::vector<std::string> clusters_name;     // point cloud name
            std::vector<float> clusters_prob;           // point cloud confidence
            std::vector<int>   clusters_class_id;       // class id

            for (int t = 0; t < objects.size() ; t++) 
            { // Color each target
	        //cv::putText(img, detector.Names(box.m_class), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, colors[box.m_class], 2);
	        //cv::rectangle(img, box, colors[box.m_class], 2);
              const Object& obj = objects[t];
              if(obj.prob >0.54)// The prediction accuracy is above 0.55 to be considered correct
               {
                //const Object& obj = objects[t];
	        //cv::rectangle(colorImgs[i], obj.rect, colors_[obj.class_id], -1, 4);
                //cv::imwrite("result.jpg", colorImgs[i]);
                //cv::rectangle(RGBImgs[i], obj.rect, colors_[obj.class_id], -1, 4);// Color the area within the target object box
                pcl::PointIndices indices;
                // Color the ROI area based on the depth threshold
                draw_rect_with_depth_threshold(RGBImgs[i], depthImgs[i], obj.rect, colors_[obj.class_id], indices);
                vec_indices.push_back(indices);           // point cloud index
                clusters_name.push_back(obj.object_name); // name
                clusters_prob.push_back(obj.prob);        // Confidence
                clusters_class_id.push_back(obj.class_id);// type id  
               }
            }
            
            if((i%2==0)&&(objects.size()==0)) continue; // skip less important frames
       
            //cv::imwrite("result.jpg", colorImgs[i]);

            //PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );// Generate a point cloud for a new frame
             PointCloud::Ptr p = generatePointCloud( keyframes[i], RGBImgs[i], depthImgs[i] );// Generate a point cloud for a new frame
            // Process each frame to obtain semantically segmented point cloud information
            extract_indices.setInputCloud(p);// Set the input point cloud pointer
 //  #pragma omp parallel for   // =======================omp Multi-threaded parallel processing
            for (int n = 0; n < vec_indices.size() ; n++)// each point cloud
            {
               pcl::PointIndices & indices = vec_indices[n];       // index of each point cloud
               std::string&   cluster_name = clusters_name[n];    //  name
               float          cluster_prob = clusters_prob[n];    // confidence
               int            class_id     = clusters_class_id[n];// type id 
               extract_indices.setIndices (boost::make_shared<const pcl::PointIndices> (indices));
               // set index
               PointCloud::Ptr before (new PointCloud);
               extract_indices.filter (*before);//Extract point cloud for index
               // Statistical filtering to remove noise
               PointCloud::Ptr after_stat (new PointCloud);
               stat.setInputCloud (before);//Set the point cloud to be filtered
	       stat.filter (*after_stat); //storage point
               // Voxel grid downsampling
               PointCloud::Ptr after_voxel (new PointCloud);
               voxel.setInputCloud( after_stat );
               voxel.filter( *after_voxel );
               // Compute point cloud center
               Eigen::Vector4f cent;
               pcl::compute3DCentroid(*after_voxel, cent);
               // Calculate point cloud point range
               Eigen::Vector4f minPt, maxPt;
               pcl::getMinMax3D (*after_voxel, minPt, maxPt);
               
               // Create a new semantic target object
               Cluster cluster;
               cluster.object_name = cluster_name;// name
               cluster.class_id    = class_id;    // type id
               cluster.prob        = cluster_prob;// confidence
               cluster.minPt       = Eigen::Vector3f(minPt[0], minPt[1], minPt[2]);// minimum
               cluster.maxPt       = Eigen::Vector3f(maxPt[0], maxPt[1], maxPt[2]);// big
               cluster.centroid    = Eigen::Vector3f(cent[0],  cent[1],  cent[2]); // center
               
               // Fusion into the total clusters
               sem_merge(cluster);

            }
           // Convert an ordered point cloud to an unordered point cloud
           std::vector<int> temp;
           PointCloud::Ptr out_pt(new PointCloud());
           pcl::removeNaNFromPointCloud(*p, *out_pt, temp);
           *globalMap += *out_pt;                   // Add to the overall point cloud map
           time = getTimeUsec() - time;// end timer
           printf("KeyFrame %ld Semtic Mapping time: %ld ms\n", i, time/1000); // Display detection time
        }

        PointCloud::Ptr tmp(new PointCloud());      // Voxel filtered point cloud
        voxel.setInputCloud( globalMap );           // Voxel Lattice Filter, Input Origin Cloud
        voxel.filter( *tmp );                       // filtered point cloud
        globalMap->swap( *tmp );                    // The global point cloud map is replaced by the voxel grid filtered point cloud
        map_state_ok = 1;
        /*
        viewer.showCloud( globalMap );              // show point cloud
        cout << "show global map, size=" << globalMap->points.size() << endl;
        */

        lastKeyframeSize = N;                       // Iteratively update the keyframe id that has been updated last time

	// show point cloud
        add_cube(); 
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
        pcl_viewer_prt->removePointCloud("SemMap");// remove the original point cloud
        pcl_viewer_prt->addPointCloud<PointT> (globalMap, rgb, "SemMap");
        pcl_viewer_prt->setPointCloudRenderingProperties (
                       pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "SemMap");

        while(map_state_ok)
        {
          //obj_pt = cluster_large.c_ptr;
          pcl_viewer_prt->spinOnce (100);// 
          // pcl_viewer_prt->spin(); // Too fast the above test can't be done
          boost::this_thread::sleep (boost::posix_time::microseconds (100000));// 
         //usleep(3000);
        }
        cout << "global map, size=" << globalMap->points.size() << endl;
        //map_state = 1;
        if(globalMap->points.size()>0)
            pcd_writer_ptr->write<PointT>("global_color.pcd", *globalMap);
    }

    //pcd_writer_ptr->write<PointT>("global_color.pcd", *globalMap);
}


// Point cloud update display
void PointCloudMapping::update()
{
std::cout<< "update"<< std::endl; 
add_cube(); 
pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(globalMap);
pcl_viewer_prt->removePointCloud("SemMap");// remove the original point cloud
pcl_viewer_prt->addPointCloud<PointT> (globalMap, rgb, "SemMap");
pcl_viewer_prt->setPointCloudRenderingProperties (
               pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "SemMap");	       
//obj_pt = cluster_large.c_ptr;
pcl_viewer_prt->spinOnce (100);// 
boost::this_thread::sleep (boost::posix_time::microseconds (100000));   //over time
}



