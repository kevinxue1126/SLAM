/* This file is part of ORB-SLAM2-SSD-Semantic.
 * Newly added semantic map building
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h> // Voxel Lattice Filtering
#include <pcl/filters/statistical_outlier_removal.h>// statistical filter
#include <pcl/filters/extract_indices.h>            // Extract the corresponding point cloud according to the point cloud index
#include <pcl/filters/filter.h>                     // Take out nan points and change them into disordered points
#include <pcl/io/pcd_io.h> // Read and write point clouds
#include <pcl/visualization/pcl_visualizer.h> // PCLVisualizer 

#include <condition_variable>

#include "Thirdparty/ncnn/ncnn_dect.h"// ncnn ssd target detection

#include <Eigen/Core>

#include <vector>

using namespace ORB_SLAM2;

// target semantic information
typedef struct Cluster
{
 std::string object_name; // object class name
 int class_id;            // corresponding category id
 float prob;              // Confidence
 Eigen::Vector3f minPt;   // The smallest x value, y value, z value among all points
 Eigen::Vector3f maxPt;   // The largest x value, y value, z value among all points
 Eigen::Vector3f centroid;// point cloud center point
 bool operator ==(const std::string &x);
} Cluster;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGB PointT;// point type xyzrgb point + color
    typedef pcl::PointCloud<PointT> PointCloud;// Point cloud type

    PointCloudMapping( double resolution_ );// Class initialization (constructor) function
  
    // Inserting a keyframe will update the map once
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& imgRGB);
    void shutdown();// Equivalent to class destructor
    void viewer();  // Visualize point cloud functions
    void update();

protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);

    void draw_rect_with_depth_threshold(cv::Mat bgr_img, 
                                    cv::Mat depth_img, 
                                    const cv::Rect_<float>& rect, const cv::Scalar& scalar,
                                    pcl::PointIndices & indices);
    void sem_merge(Cluster cluster); // fused point cloud
    void add_cube(void);// Add a 3d logo frame to the visualizer

    PointCloud::Ptr globalMap;        // Point cloud map pointer
    std::shared_ptr<thread>  viewerThread; // Point cloud visualization thread

    bool    shutDownFlag    =false;   // close sign
    mutex   shutDownMutex;            // close thread mutex
 
    condition_variable  keyFrameUpdated; 
    // Keyframe Updates The <condition_variable> header file mainly contains classes and functions related to condition variables.
    mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    std::vector<KeyFrame*>       keyframes;  // Keyframe pointer
    std::vector<cv::Mat>         colorImgs;  // Grayscale
    std::vector<cv::Mat>         depthImgs;  // depth map    
    std::vector<cv::Mat>         RGBImgs;    // depth map    
    mutex                   keyframeMutex;   // Keyframe
    uint16_t                lastKeyframeSize =0;

    double resolution = 0.04;      // Default point cloud map accuracy
    pcl::VoxelGrid<PointT>  voxel; // The voxel filter object corresponding to the point

    //shared_ptr<std::vector<cv::Scalar>> colors_ptr;// different objects color objects   std::shared_ptr
    std::vector<cv::Scalar> colors_;   // the color of each object
    std::vector<float>      obj_size_; // size of each object
    std::shared_ptr<Detector> ncnn_detector_ptr; // ncnn ssd target detection std::shared_ptr
    

    std::vector<Cluster> clusters;// Array of semantic point cloud pointers
    pcl::ExtractIndices<PointT> extract_indices;// Index extraction point cloud
    // pcl::VoxelGrid<PointT>  voxel_; // voxel filter object
    pcl::StatisticalOutlierRemoval<PointT> stat; // Statistical filtering to remove outliers

    
    std::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer_prt;
    std::shared_ptr<pcl::PCDWriter>                    pcd_writer_ptr;


    // Is the map ready
    int map_state_ok;
};

#endif // POINTCLOUDMAPPING_H
