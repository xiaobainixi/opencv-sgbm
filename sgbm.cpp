#include <pcl/visualization/cloud_viewer.h>
#include <iostream>  
#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h>  
#include <opencv2/opencv.hpp>  
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include "elas.h"
#include <time.h>
#include <math.h>
#include <fstream>
using namespace cv;
using namespace std;
using namespace pcl;
typedef PointXYZRGB PointT; 
Size calib_img_size ;
  Rect validRoi[2];
int user_data;
Mat Q;
clock_t start, finish; 
Mat K1 = Mat::zeros(3,3, CV_32F); 
  Mat K2 ;
  Mat D1 ;
  Mat D2 ;

Mat lmapx, lmapy, rmapx, rmapy;
double length;
struct callback_args
{  
    // structure used to pass arguments to the callback function  
    pcl::PointCloud<PointT>::Ptr clicked_points_3d;  
    pcl::visualization::PCLVisualizer::Ptr viewerPtr;  
};  

void insertDepth32f(cv::Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}

void pp_callback(const pcl::visualization::PointPickingEvent& event, void* args)  
{  
    struct callback_args* data = (struct callback_args *)args;  
    if (event.getPointIndex() == -1)  
        return;  
    PointT current_point;  
    event.getPoint(current_point.x, current_point.y, current_point.z);  
    data->clicked_points_3d->points.push_back(current_point);  
    // Draw clicked points in red:  
    pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);  
    data->viewerPtr->removePointCloud("clicked_points");  
    data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");  
    data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");  
    std::cout << sqrt(current_point.x*current_point.x + current_point.y*current_point.y + current_point.z*current_point.z) << std::endl;  
}  



cv::Mat toCvMatInverse(const cv::Mat &Tcw)
{
  cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
  cv::Mat tcw = Tcw.rowRange(0,3).col(3);
  cv::Mat Rwc = Rcw.t();
  cv::Mat twc = -Rwc*tcw;

  cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
  Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
  twc.copyTo(Twc.rowRange(0,3).col(3));

  return Twc.clone();
}

void findRectificationMap(FileStorage& calib_file) 
{
  Rect validRoi[2];
  cout << "Starting rectification" << endl;
  Mat TSC1 ;
  Mat TSC2;
  
  Mat P1, P2;
  Mat R1, R2;
  
  Mat R;
  Vec3d T;

  calib_file["M1"] >> K1;
  calib_file["M2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  //calib_file["TSC1"] >> TSC1;
  calib_file["R"] >> R;
  calib_file["T"] >> T;
  //calib_file["TSC1"] >> TSC1;
  //calib_file["TSC2"] >> TSC2;
  int x,y;
  calib_file["x"] >> x;
  calib_file["y"] >> y;
  calib_img_size = Size(x, y);
  //Mat TT = toCvMatInverse(TSC2)*TSC1;
  //cv::Mat Rcw = TT.rowRange(0,3).colRange(0,3);
  //cv::Mat tcw = TT.rowRange(0,3).col(3);
  //length = sqrt(tcw.at<double>(0,0)*tcw.at<double>(0,0) + tcw.at<double>(0,1)*tcw.at<double>(0,1) + tcw.at<double>(0,2)*tcw.at<double>(0,2));
  Size finalSize = calib_img_size;
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, T, R1, R2, P1, P2, Q, 
                CV_CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  cv::initUndistortRectifyMap(K1, D1, R1, P1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, finalSize, CV_32F, rmapx, rmapy);
  cout << "Done rectification" << endl;
}

void viewerOneOff(visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor(255, 255,255);
}


int main()
{
    FileStorage calib_file = FileStorage("/home/l/tt/dashuixing.yml", FileStorage::READ);
      findRectificationMap(calib_file);
    Mat leftc =  imread( "/home/l/orbshuju/l/1.png");  
    Mat rightc = imread( "/home/l/orbshuju/r/1.png");
    Size img_size = leftc.size();
    Mat disp;

    //imshow("ImageL Before Rectify", leftc);
    //imshow("ImageR Before Rectify", rightc);

    Mat rgbRectifyImageL, rgbRectifyImageR;
    remap(leftc, rgbRectifyImageL, lmapx, lmapy, INTER_LINEAR);
    remap(rightc, rgbRectifyImageR, rmapx, rmapy, INTER_LINEAR);

    /*
    把校正结果显示出来
    */
    Mat rectifyImageL, rectifyImageR;
    cvtColor(leftc,rectifyImageL,CV_BGR2GRAY);  //伪彩色图
    cvtColor(rgbRectifyImageR,rectifyImageR,  CV_BGR2GRAY);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    //imshow("ImageL After Rectify", rgbRectifyImageL);
    //imshow("ImageR After Rectify", rgbRectifyImageR);
    Mat left =  leftc;//rectifyImageL;
    Mat right = rightc;//rectifyImageR;
    Mat color = rectifyImageL;

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(img_size.width, img_size.height);
    w = cvRound(img_size.width * sf);
    h = cvRound(img_size.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(leftc, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),                //获得被截取的区域    
        cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rightc, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validRoi[1].x * sf), cvRound(validRoi[1].y*sf),
        cvRound(validRoi[1].width * sf), cvRound(validRoi[1].height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);


  cvWaitKey(0);




      
    int numberOfDisparities = 256;//((left.rows / 8) +15) & -16;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);

    sgbm->setPreFilterCap(12);

    int sgbmWinSize = 5;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = left.channels();

    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(0.1);
    sgbm->setSpeckleWindowSize(30);//100
    sgbm->setSpeckleRange(5);//32
    sgbm->setDisp12MaxDiff(1);
    //int alg = STEREO_SGBM;
    //if (alg == STEREO_HH)
    //    sgbm->setMode(cv::StereoSGBM::MODE_HH);
    //else if (alg == STEREO_SGBM)c
    //     sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    //else if (alg == STEREO_3WAY)
    //sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->compute(left, right, disp);
    Mat absdisp = abs(disp);
    char aaa = -1;

    cout<<"16S:"<<(double)absdisp.ptr<char>(100)[400]<<"tt"<<abs(aaa)<<endl;
//Mat leftdpf = disp;

Mat leftdpf = Mat::zeros(img_size, CV_8UC1);
cout<<"numberOfDisparities"<<numberOfDisparities<<endl;
absdisp.convertTo(leftdpf, CV_8U, 255/(numberOfDisparities*16.));


/*
Mat mat32f,mat8u;
leftdpf.convertTo(mat32f,CV_32F,1.0/255.0);
insertDepth32f(mat32f);
mat32f.convertTo(mat8u,CV_8U,255.0);
leftdpf = mat8u;
*/
  imwrite("leftdpf.png",leftdpf);
    //disp2Depth(leftdpf,depth);
cout<<"8U"<<(double)leftdpf.ptr<uchar>(100)[400]<<endl;

PointCloud<PointXYZRGB>::Ptr cloud_a(new PointCloud<PointXYZRGB>);
PointCloud<PointXYZRGB> cloud_b;
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    int rowNumber = left.rows;
    int colNumber = left.cols;

    cloud_a->height = rowNumber;
    cloud_a->width = colNumber;
    cloud_a->points.resize(cloud_a->width * cloud_a->height);
    cloud_b.points.resize(20);  
    Mat V = Mat(4, 1, CV_64FC1);
    Mat pos = Mat(4, 1, CV_64FC1);

    cout<<Q<<endl;
    
    for (unsigned int u = int(rowNumber*1/10); u < rowNumber*9/10; ++u)
    {
        for (unsigned int v = int(colNumber*1/10); v < colNumber*9/10; ++v)
        {
            //*unsigned int num = rowNumber*colNumber-(u*colNumber + v)-1;
            unsigned int num = u*colNumber + v;
            
            double d = double(leftdpf.ptr<uchar>(u)[v]);
            if((double)leftdpf.ptr<uchar>(u)[v] == 0 ||(double)leftdpf.ptr<uchar>(u)[v] == 8)
                continue;

            V.at<double>(0,0) = (double)(v);
            V.at<double>(1,0) = (double)(u);
            V.at<double>(2,0) = (double) d;
            V.at<double>(3,0) = 1.;
            pos = Q * V; // 3D homogeneous coordinate

            double X = pos.at<double>(0,0) / pos.at<double>(3,0);
    	    double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
    	    double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
            if(Z>10)
                continue;
            
            cloud_a->points[num].b = color.at<Vec3b>(u, v)[0];//[0]
            cloud_a->points[num].g = color.at<Vec3b>(u, v)[0];//[1]
            cloud_a->points[num].r = color.at<Vec3b>(u, v)[0];//[2]

            cloud_a->points[num].x = X;
            cloud_a->points[num].y = Y;
            cloud_a->points[num].z = Z;
            
        }
    }

    
    *cloud = *cloud_a;//+cloud_b;
    pcl::io::savePCDFile( "result.pcd", *cloud_a );
    /*
    PointCloud<PointXYZRGB>::Ptr tmp (new PointCloud<PointXYZRGB>);
    StatisticalOutlierRemoval<PointT> statistical_filter;
    statistical_filter.setMeanK(50);
    statistical_filter.setStddevMulThresh(1.0);
    statistical_filter.setInputCloud(cloud);
    statistical_filter.filter( *tmp );
    */
    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC; 
    cout<<"duration" <<duration<<"dian"<<endl;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));

    viewer->addPointCloud(cloud);
    //pcl::io::savePCDFile( "result.pcd", *tmp );
    //viewer.runOnVisualizationThreadOnce(viewerOneOff);
    struct callback_args cb_args;  
    pcl::PointCloud<PointT>::Ptr clicked_points_3d(new pcl::PointCloud<PointT>);  
    cb_args.clicked_points_3d = clicked_points_3d;  
    cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);  
    viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);  
    viewer->spin();
    while (!viewer->wasStopped())
    {
        user_data = 9;
    }
    return 0;
}
