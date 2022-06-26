/**
* This file is part of ORB-SLAM2.
* Tuyang rgbd orbslam2 example
* Run ： ./ty_rgbd ../../Vocabulary/ORBvoc.bin ./my_rgbd_ty_api_adj.yaml
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>


#include "./ty/common/common.hpp" // Tuyang's header file

using namespace std;

// 1. parameter initialization
static char buffer[1024 * 1024 * 20];// Store sdk version and other information
static int  n;// usb device number
static volatile bool exit_main;

static int fps_counter = 0;
static clock_t fps_tm = 0;
const char* IP = NULL; // pointer to read-only characters IP, network device IP address
TY_DEV_HANDLE hDevice; // device handle
//int32_t color = 1, ir = 0, depth = 1; // rgb image ir infrared image depth depth image
//int i = 1; // capture image index

char img_color_file_name[15]; 

char* frameBuffer[2]; // frame buffer

double slam_sys_time=0.0, ttrack=0; // timestamp

// ORB_SLAM2::System SLAM();

int capture_ok_flag = 0;

cv::Mat color_img, depth_img;// Color and Grayscale
cv::Mat  undistort_result;   // Color Correction Chart
cv::Mat newDepth;            // Configured depth map

// Callback function data structure
struct CallbackData {
    int             index;
    TY_DEV_HANDLE   hDevice;
    DepthRender*    render;
    TY_CAMERA_DISTORTION color_dist;// distortion parameter
    TY_CAMERA_INTRINSIC color_intri;// intrinsic parameters
};
CallbackData cb_data;     // callback function data

int get_fps(); // frames /s
void eventCallback(TY_EVENT_INFO *event_info, void *userdata);// event callback function
void frameHandler(TY_FRAME_DATA* frame, void* userdata, ORB_SLAM2::System & SLAM2);// frame callback function
void frameHandler(TY_FRAME_DATA* frame, void* userdata);// frame callback function
int ty_RGBD_init(void); // image camera initialization

int main(int argc, char **argv)
{
// 1. detect command line input
    if(argc != 3)
    {
        cerr << endl << "Usage: ./my_rgbd path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

// 2. image camera initialization
    int init_flag = ty_RGBD_init();
    if(init_flag == -1) 
    {
	LOGD("=== camera init failed===");
        return -1;
    }


// 3. create a binocular system ORB_SLAM2::System
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);
   //SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);


    cout << endl << "-------" << endl;
    cout << "Start processing ..." << endl;
    
// 4. start acquisition
    LOGD("=== Start capture");
    ASSERT_OK( TYStartCapture(hDevice) );


// 4.5 get camera parameters
    LOGD("=== Read color rectify matrix");
    {
        TY_CAMERA_DISTORTION color_dist;// Camera Distortion Parameters
        TY_CAMERA_INTRINSIC color_intri;// Camera internal parameters
        TY_STATUS ret = TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                                    TY_STRUCT_CAM_DISTORTION, &color_dist, sizeof(color_dist));
        ret |= TYGetStruct(hDevice, TY_COMPONENT_RGB_CAM, 
                           TY_STRUCT_CAM_INTRINSIC, &color_intri, sizeof(color_intri));
        if (ret == TY_STATUS_OK)
        {
            cb_data.color_intri = color_intri;// Camera internal parameters
            cb_data.color_dist= color_dist;   // Camera Distortion Parameters
        }
        else
        { //reading data from device failed .set some default values....
            memset(cb_data.color_dist.data, 0, 12 * sizeof(float));
            memset(cb_data.color_intri.data, 0, 9 * sizeof(float));// intrinsic parameters
            cb_data.color_intri.data[0] = 1000.f;// fx
            cb_data.color_intri.data[4] = 1000.f;// fy
            cb_data.color_intri.data[2] = 600.f;// cx
            cb_data.color_intri.data[5] = 450.f;// cy
        }
    }


    LOGD("=== While loop to fetch frame");
    exit_main = false;
    TY_FRAME_DATA frame; // data for each frame
    
    while(!exit_main) 
    {
// 5. active acquisition of frame data (not called in active acquisition mode)
        int err = TYFetchFrame(hDevice, &frame, -1);// get a frame of data
        if( err != TY_STATUS_OK ) 
        {
            LOGD("... Drop one frame");
        } 
        else 
        { // process the acquired frame data
            //frameHandler(&frame, &cb_data, SLAM);
              frameHandler(&frame, &cb_data);
        }
        
        if(capture_ok_flag){
                // LOGD("... tracking ... ");
		capture_ok_flag = 0;
	    // 6. Record the timestamp tframe and time it
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
	#endif	
		slam_sys_time += ttrack ;

	// 7. Pass the left and right images and timestamps to the SLAM system
		SLAM.TrackRGBD(color_img, depth_img, slam_sys_time);

	// 8. Timing is over, time difference is calculated, processing time
	#ifdef COMPILEDWITHC11
		std::chrono::steady_clock::time_point        t2 = std::chrono::steady_clock::now();
	#else
		std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
	#endif

		ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
		
        }
	
	
    }

// 9. end, close the slam system, close all threads
    // Stop all threads
    SLAM.Shutdown();

// 10. Device stops collecting
    ASSERT_OK( TYStopCapture(hDevice) );
// 11. Turn off the device
    ASSERT_OK( TYCloseDevice(hDevice) );
// 12. release API
    ASSERT_OK( TYDeinitLib() );
    // MSLEEP(10); // sleep to ensure buffer is not used any more

// 13. Free up memory space
    delete frameBuffer[0];
    delete frameBuffer[1];

    LOGD("=== Main done!");

// 14. Save camera track
    SLAM.SaveTrajectoryTUM("my_ty_rgbd_CameraTrajectory.txt");
    //SLAM.SaveKeyFrameTrajectoryTUM("my_ty_rgbd_KeyFrameTrajectory.txt");   

    return 0;
}


//================================================
//=== Calculate the number of frames fps
//================================================
#ifdef _WIN32
// frame number===================
int get_fps() {
   const int kMaxCounter = 250;
   fps_counter++;
   if (fps_counter < kMaxCounter) {
     return -1;
   }
   int elapse = (clock() - fps_tm);
   int v = (int)(((float)fps_counter) / elapse * CLOCKS_PER_SEC);
   fps_tm = clock();

   fps_counter = 0;
   return v;
 }

#else
int get_fps() {
  const int kMaxCounter = 200;
  struct timeval start;
  fps_counter++;
  if (fps_counter < kMaxCounter) {
    return -1;
  }

  gettimeofday(&start, NULL);
  int elapse = start.tv_sec * 1000 + start.tv_usec / 1000 - fps_tm;
  int v = (int)(((float)fps_counter) / elapse * 1000);
  gettimeofday(&start, NULL);
  fps_tm = start.tv_sec * 1000 + start.tv_usec / 1000;

  fps_counter = 0;
  return v;
}
#endif


//=========================================================
//=== Callback function for processing camera frame
//=========================================================
void frameHandler(TY_FRAME_DATA* frame, void* userdata, ORB_SLAM2::System & SLAM2) {
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index);// 帧id

   // int ret = get_fps();
   //   if (ret > 0)
   //   printf("fps: %d\n", ret);

    // cv::Mat depth, irl, irr, color;
    // parseFrame(*frame, &depth, &irl, &irr, &color, 0);
    // cv::Mat color_img, depth_img;
    parseFrame(*frame, &depth_img, 0, 0, &color_img, 0);

    //if(!depth.empty()){
    //    cv::Mat colorDepth = pData->render->Compute(depth);
    //    cv::imshow("ColorDepth", colorDepth);
    //}
    //if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    //if(!irr.empty()){ cv::imshow("RightIR", irr); }
    // cv::namedWindow("Color", CV_WINDOW_NORMAL);
    //if(!color.empty()){ cv::imshow("Color", color); }

if((!depth_img.empty()) && (!color_img.empty()))
{
    // 15. Record the timestamp tframe and time it
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point    t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif	
        slam_sys_time += ttrack ;

// 16. Pass the left and right images and timestamps to the SLAM system
        SLAM2.TrackStereo(color_img, depth_img, slam_sys_time);

// 17. Timing is over, time difference is calculated, processing time
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point        t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
}

    int key = cv::waitKey(1);
    switch(key & 0xff) 
    {
    case 0xff:
        break;
    case 'q':
        exit_main = true;
        break;
    default:
        LOGD("Unmapped key %d", key);
    }

    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

//=========================================================
//=== Callback function for processing camera frame
//=========================================================
void frameHandler(TY_FRAME_DATA* frame, void* userdata) {
    CallbackData* pData = (CallbackData*) userdata;
    // LOGD("=== Get frame %d", ++pData->index);// 帧id

   // int ret = get_fps();
   //   if (ret > 0)
   //   printf("fps: %d\n", ret);

    // cv::Mat depth_src, irl, irr, color_src, point3D;
    cv::Mat irl, irr, point3D;
    parseFrame(*frame, &depth_img, &irl, &irr, &color_img, &point3D);

    //if(!depth_img.empty()){
    //    cv::Mat colorDepth = pData->render->Compute(depth_img);
    //    cv::imshow("ColorDepth", colorDepth);
    //}
    //if(!irl.empty()){ cv::imshow("LeftIR", irl); }
    //if(!irr.empty()){ cv::imshow("RightIR", irr); }
    // cv::namedWindow("Color", CV_WINDOW_NORMAL);
    // if(!color_img.empty()){ cv::imshow("Color", color_img); }

    // Straighten color images
    if(!color_img.empty())
    {
        //cv::Mat undistort_result(color_img.size(), CV_8UC3);// Correction results
        undistort_result = cv::Mat(color_img.size(), CV_8UC3);// Correction results
        TY_IMAGE_DATA dst;        // target image
        dst.width = color_img.cols;   // width number of columns
        dst.height = color_img.rows;  // height row
        dst.size = undistort_result.size().area() * 3;// 3 channels
        dst.buffer = undistort_result.data;
        dst.pixelFormat = TY_PIXEL_FORMAT_RGB; //RGB format

        TY_IMAGE_DATA src;        // source image
        src.width = color_img.cols;
        src.height = color_img.rows;
        src.size = color_img.size().area() * 3;
        src.pixelFormat = TY_PIXEL_FORMAT_RGB;
        src.buffer = color_img.data; 
        //undistort camera image 
        //TYUndistortImage accept TY_IMAGE_DATA from TY_FRAME_DATA , pixel format RGB888 or MONO8
        //you can also use opencv API cv::undistort to do this job.
        ASSERT_OK(TYUndistortImage(&pData->color_intri, &pData->color_dist, NULL, &src, &dst));
        color_img = undistort_result;

       // cv::Mat resizedColor;
       // cv::resize(color, resizedColor, depth.size(), 0, 0, CV_INTER_LINEAR);
       // cv::imshow("color", resizedColor);
    }


    // do Registration
    // cv::Mat newDepth; 
    if(!point3D.empty() && !color_img.empty()) 
    {
        ASSERT_OK( TYRegisterWorldToColor2(pData->hDevice, (TY_VECT_3F*)point3D.data, 0, 
                   point3D.cols * point3D.rows, color_img.cols, color_img.rows, (uint16_t*)buffer, sizeof(buffer)
                    ));
        newDepth = cv::Mat(color_img.rows, color_img.cols, CV_16U, (uint16_t*)buffer);
        cv::Mat resized_color;
        cv::Mat temp;
        //you may want to use median filter to fill holes in projected depth image or do something else here
        cv::medianBlur(newDepth,temp,5);
        newDepth = temp;
        //resize to the same size for display
        // cv::resize(newDepth, newDepth, depth_src.size(), 0, 0, 0);
        // cv::resize(color, resized_color, depth_src.size());

       // cv::Mat depthColor = pData->render->Compute(newDepth);
       // cv::imshow("Registrated ColorDepth", depthColor);

       // depthColor = depthColor / 2 + resized_color / 2; 
       // cv::imshow("projected depth", depthColor);
    } 

    if((!depth_img.empty()) && (!color_img.empty()) && (!point3D.empty()))
    {
	capture_ok_flag = 1;
    }

    int key = cv::waitKey(1);
    switch(key & 0xff) 
    {
    case 0xff:
        break;
    case 'q':
        exit_main = true;
        break;
    default:
        LOGD("Unmapped key %d", key);
    }
    //LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame->userBuffer, frame->bufferSize);
    ASSERT_OK( TYEnqueueBuffer(pData->hDevice, frame->userBuffer, frame->bufferSize) );
}

//==================================================
//====event callback function
//==================================================
void eventCallback(TY_EVENT_INFO *event_info, void *userdata)
{
    if (event_info->eventId == TY_EVENT_DEVICE_OFFLINE) {
        LOGD("=== Event Callback: Device Offline!");
        // Note: 
        //     Please set TY_BOOL_KEEP_ALIVE_ONOFF feature to false if you need to debug with breakpoint!
    }
    else if (event_info->eventId == TY_EVENT_LICENSE_ERROR) {
        LOGD("=== Event Callback: License Error!");
    }
}


//===================================================
//======Image camera initialization
//===================================================
int ty_RGBD_init(void)
{
    //cv::namedWindow("Color", CV_WINDOW_NORMAL);//CV_WINDOW_NORMAL就是0
//1. Initialize the API
    LOGD("=== Init lib");
    ASSERT_OK( TYInitLib() );
    TY_VERSION_INFO* pVer = (TY_VERSION_INFO*)buffer;
    ASSERT_OK( TYLibVersion(pVer) );// Get sdk software version information========
    LOGD("     - lib version: %d.%d.%d", pVer->major, pVer->minor, pVer->patch);

// 2. Turn on the device
    if(IP) // Turn on the network device
    {
        LOGD("=== Open device %s", IP);
        ASSERT_OK( TYOpenDeviceWithIP(IP, &hDevice) );
    } 
    else   // Turn on the USB device
    {
        LOGD("=== Get device info");
        ASSERT_OK( TYGetDeviceNumber(&n) );// Get device number
        LOGD("     - device number %d", n);

        TY_DEVICE_BASE_INFO* pBaseInfo = (TY_DEVICE_BASE_INFO*)buffer;
        ASSERT_OK( TYGetDeviceList(pBaseInfo, 100, &n) );

        if(n == 0) {
            LOGD("=== No device got");
            return -1;
        }

        LOGD("=== Open device 0");
        ASSERT_OK( TYOpenDevice(pBaseInfo[0].id, &hDevice) );
    }

// 3. Operating components
    int32_t allComps;
    ASSERT_OK( TYGetComponentIDs(hDevice, &allComps) );//Query component status
    // TYGetEnabledComponents(hDevice, &allComps) 
    if(allComps & TY_COMPONENT_RGB_CAM) 
    { // Enable RGB component function
        LOGD("=== Has RGB camera, open RGB cam");
        ASSERT_OK( TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM) );
    }

/*
    int32_t componentIDs = 0;
    LOGD("=== Configure components, open depth cam");
    if (depth) 
    {// 
        componentIDs = TY_COMPONENT_DEPTH_CAM;
    }

    if (ir) 
    {// 
        componentIDs |= TY_COMPONENT_IR_CAM_LEFT;
    }

    if (depth || ir) 
    {
        ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );
    }
*/
    LOGD("=== Configure components");
    int32_t componentIDs = TY_COMPONENT_POINT3D_CAM | TY_COMPONENT_RGB_CAM;// Turning on the 3d point cloud component + RGB component is equivalent to turning on all
    ASSERT_OK( TYEnableComponents(hDevice, componentIDs) );
    


// 4. Configuration parameters feature resolution, etc.
    LOGD("=== Configure feature, set resolution to 640x480.");
    LOGD("Note: DM460 resolution feature is in component TY_COMPONENT_DEVICE,");
    LOGD("      other device may lays in some other components.");
    TY_FEATURE_INFO info;

    TY_STATUS ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) 
    { 
      // Set the resolution depth map
      int err = TYSetEnum(hDevice,            TY_COMPONENT_DEPTH_CAM, 
                          TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
        ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);   
    } 
    
    ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_RGB_CAM, TY_ENUM_IMAGE_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) 
    { 
      // Set Resolution Colormap
      int err = TYSetEnum(hDevice,            TY_COMPONENT_RGB_CAM, 
                          TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_640x480);
        ASSERT(err == TY_STATUS_OK || err == TY_STATUS_NOT_PERMITTED);   
    }

// 5. Allocate memory space
    LOGD("=== Prepare image buffer");
    
    // Query the size of each framebuffer under the current configuration
    int32_t frameSize;
    //frameSize = 1280 * 960 * (3 + 2 + 2);
    // If configured as 640*480 then: frameSize = 640 * 480 * (3 + 2 + 2)
    ASSERT_OK( TYGetFrameBufferSize(hDevice, &frameSize) );
    LOGD("     - Get size of framebuffer, %d", frameSize);
    // ASSERT( frameSize >= 640 * 480 * 2 );
    
    // Allocate framebuffer and push into buffer queue in driver
    LOGD("     - Allocate & enqueue buffers");
    // char* frameBuffer[2];
    frameBuffer[0] = new char[frameSize];
    frameBuffer[1] = new char[frameSize];
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize) );
    LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
    ASSERT_OK( TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize) );

// 6. Register the callback function (not called in active acquisition mode).
    LOGD("=== Register callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    LOGD("      To avoid copying data, we pop the framebuffer from buffer queue and");
    LOGD("      give it back to user, user should call TYEnqueueBuffer to re-enqueue it.");
    DepthRender render;
    // CallbackData cb_data;
    cb_data.index = 0;
    cb_data.hDevice = hDevice;
    cb_data.render = &render;
    //ASSERT_OK( TYRegisterCallback(hDevice, frameHandler, &cb_data) );

    LOGD("=== Register event callback");
    LOGD("Note: Callback may block internal data receiving,");
    LOGD("      so that user should not do long time work in callback.");
    ASSERT_OK(TYRegisterEventCallback(hDevice, eventCallback, NULL));
    
    // Cancel Active send data mode
    LOGD("=== Disable trigger mode");
    ty_status = TYGetFeatureInfo(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, &info);
    if ((info.accessMode & TY_ACCESS_WRITABLE) && (ty_status == TY_STATUS_OK)) {
        ASSERT_OK(TYSetBool(hDevice, TY_COMPONENT_DEVICE, TY_BOOL_TRIGGER_MODE, false));
    }
    return 0;
}

