
#include "./ncnn_dect.h"

// ncnn register new layer
class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)

//class Detector
//{
// initialization
Detector::Detector()
{
// The two file addresses here need to be passed in
  det_net_ptr = new(ncnn::Net); // model
  net_in_ptr  = new(ncnn::Mat); // network input
  // Load model new layer registration
  det_net_ptr->register_custom_layer("Silence", Noop_layer_creator);// Avoid printing information about blobs that are not used in the log
    // original pretrained model from https://github.com/chuanqi305/MobileNetv2-SSDLite
    // https://github.com/chuanqi305/MobileNetv2-SSDLite/blob/master/ssdlite/voc/deploy.prototxt
  det_net_ptr->load_param("/home/ewenwan/ewenwan/project/ORB_SLAM2-src-pc-sem/Examples/RGB-D/ncnn/model/mobilenetv2_ssdlite_voc.param");
  det_net_ptr->load_model("/home/ewenwan/ewenwan/project/ORB_SLAM2-src-pc-sem/Examples/RGB-D/ncnn/model/mobilenetv2_ssdlite_voc.bin");

}

void Detector::Run(const cv::Mat& bgr_img, std::vector<Object>& objects)
{
// format network input
    const int target_size = 300;

    int src_img_w = bgr_img.cols;
    int src_img_h = bgr_img.rows;

   // *net_in_ptr = ncnn::Mat::from_pixels_resize(bgr_img.data, ncnn::Mat::PIXEL_BGR, bgr_img.cols, bgr_img.rows, target_size, target_size);
    // need to determine the format of the image
    *net_in_ptr = ncnn::Mat::from_pixels_resize(bgr_img.data, ncnn::Mat::PIXEL_RGB, bgr_img.cols, bgr_img.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0/127.5,1.0/127.5,1.0/127.5};
    net_in_ptr->substract_mean_normalize(mean_vals, norm_vals);// De-mean normalization
	//  Network operation to get results
    ncnn::Extractor net_extractor = det_net_ptr->create_extractor();// Create model extractor from network model difference
    net_extractor.set_light_mode(true);        
    // Open the light mode to save memory, and the memory of the intermediate results can be automatically recycled after each layer of operation
    net_extractor.set_num_threads(4);// omp number of threads
    // printf("run ... ");
    net_extractor.input("data", *net_in_ptr);
    ncnn::Mat out;
    net_extractor.extract("detection_out",out);

	static const char* class_names[] = {"background",
	"aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair",
	"cow", "diningtable", "dog", "horse",
	"motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tvmonitor"};

    objects.clear();
    for (int i=0; i<out.h; i++)
    {
        const float* values = out.row(i);// each test result

        Object object;// one
        object.object_name = std::string(class_names[int(values[0])]);
        object.prob = values[1];
        object.rect.x = values[2] * src_img_w;
        object.rect.y = values[3] * src_img_h;
        object.rect.width = values[4]  * src_img_w - object.rect.x;
        object.rect.height = values[5] * src_img_h - object.rect.y;

        objects.push_back(object);
    }
}

void Detector::Show(const cv::Mat& bgr_img, std::vector<Object>& objects)
{
    cv::Mat image = bgr_img.clone(); // deep copy

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

       // fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", obj.object_name.c_str(), obj.prob,
       //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));// rectangle
        char text[256];
        sprintf(text, "%s %.1f%%", obj.object_name.c_str(), obj.prob * 100);// displayed characters

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);// character size
        // rectangular frame limit
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        
        // show Matrix Box on Plot
        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);
        // image display text 
        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    // imwrite("result.jpg", image);
     cv::imshow("image", image);
     // cv::waitKey(0); // wait for a keypress, the window will not exit immediately
}


// destructor
Detector::~Detector()
{
    delete det_net_ptr;
    delete net_in_ptr;
}

