
#ifndef DETECTOR_H
#define DETECTOR_H

#include "./include/net.h"  // ncnn header file
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>// ncnn header file
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

//Test result class
typedef struct Object
{
    cv::Rect_<float> rect;// frame
    std::string object_name;// object class name
    int class_id; // type id
    float prob;   // confidence
} Object;

class Detector
{
public:
  /** Default constructor */
  Detector();// network initialization

  /** Default destructor */
  ~Detector();// destructor
  
  void Run(const cv::Mat& bgr_img, std::vector<Object>& objects);
  void Show(const cv::Mat& bgr_img, std::vector<Object>& objects);
private:
   // ncnn::Net * det_net_mobile;  
   ncnn::Net * det_net_ptr;// detect network pointers
   ncnn::Mat * net_in_ptr; // network input pointer
};

#endif
