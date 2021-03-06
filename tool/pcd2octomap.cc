//
// Created by ichigoi7e on 13/07/2018.
//

#include <iostream>
#include <assert.h>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//octomap
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace std;

int main(int argc, char** argv)
{
    if(argc != 3) {
        cout << "Usage: pcd2colorOctomap <input_file> <output_file>" << endl;
        return -1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    pcl::io::loadPCDFile<pcl::PointXYZRGBA> (input_file, cloud);

    cout << "point cloud loaded, point size = " << cloud.points.size() << endl;

    //Declare the octomap variable
    cout << "copy data into octomap..." << endl;
    //Create an octree object with color, the parameter is resolution, here is set to 0.04
    octomap::ColorOcTree tree( 0.04 );

    for(auto p:cloud.points) {
        //Insert points from point cloud into octomap
        tree.updateNode(octomap::point3d(p.x, p.y, p.z), true);
    }

    //set color
    for(auto p:cloud.points) {
        tree.integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
    }

    //update octomap
    tree.updateInnerOccupancy();
    //Store the octomap, be careful to save it as a .ot file instead of a .bt file
    tree.write(output_file);
    cout << "done." << endl;

    return 0;
}
