#include <QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>	//SurfFeatureDetector实际在该头文件中
#include "opencv2/features2d/features2d.hpp"	//FlannBasedMatcher实际在该头文件中
#include "opencv2/calib3d/calib3d.hpp"	//findHomography所需头文件
#include <opencv2/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/affine.hpp>
using namespace std;
using namespace cv;
using namespace xfeatures2d;
Mat image;
Mat imageGray;
int thresh=150;
int MaxThresh=200;
Mat src;
int minHessian = 5000;
void trackBar(int, void*);  //SURF阈值控制
void Trackbar(int,void*);  //Harris阈值控制
bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints,
                                 float reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography);
//cv::Vec3b target=cv::Vec3b(15,18,152);
bool getDistance(const cv::Vec3b& color){
//        return color[1]-color[0] > 0 &&
//                color[1]-color[2] > 0 &&
//                 color[1] != 0 ;
    return color[0]<210 && color[0]>100;
}


//int main(int argc, char *argv[])
//{
//    QCoreApplication a(argc, argv);


//    return a.exec();
//}
int main(){
    std::cout<<"....."<<std::endl<<"1-->color\n2-->Harris\n3-->SURFpipei\n4-->shuangyuzhi\n5-->SURFjiance"<<endl;
    char a;
    std::cin>>a;
    //条件BGR
    if(a==49){
        for(int m=0;m<55;m++){
            char num=0;
            std::cin>>num;
            cv::Mat src = cv::imread(QString("C:\\Users\\Duke\\Desktop\\tree\\%1.jpg").arg(num).toStdString());
            int ratio;
            if(src.size().width<400) ratio = 1;
            else ratio = src.size().width/400;
            if(!src.data){
                return -1;
            }
            cv::Mat resized;
            cv::resize(src,resized,cv::Size(src.cols/ratio,src.rows/ratio));
            cv::Mat result(resized.size(),CV_8U,cv::Scalar(255));
            cv::Mat_ <cv::Vec3b>::const_iterator it=resized.begin<cv::Vec3b> ();
            cv::Mat_ <cv::Vec3b>::const_iterator itend=resized.end<cv::Vec3b> ();
            cv::Mat_ <uchar>::iterator itout=result.begin<uchar>();
            for(;it!=itend;++it,++itout){
                if(getDistance(*it)){
                    *itout=255;//0代表黑色
                }else{
                    *itout=0;
                }
            }
    //        cv::namedWindow("src");
    //        cv::imshow("src",resized);
    //        cv::namedWindow("resized");
    //        cv::imshow("resized",result);
    //        cv::imwrite("C:\\Users\\Duke\\Desktop\\tree\\4_2.jpg",resized);//重采样后的彩色图
            cv::imwrite(QString("C:\\Users\\Duke\\Desktop\\tree\\%1test2.jpg").arg(num).toStdString(),result);//提取后的结果

        }
            }
    //Harris角点检测
    if(a==50){
        image=imread("C:\\Users\\Duke\\Desktop\\test\\3test2.jpg");
        cvtColor(image,imageGray,CV_RGB2GRAY);
        //GaussianBlur(imageGray,imageGray,Size(3,3),1); //滤波
        namedWindow("Corner Detected");
        createTrackbar("threshold:","Corner Detected",&thresh,MaxThresh,Trackbar);
        imshow("Corner Detected",image);
        Trackbar(0,0);
        waitKey();
    }

    if(a==51){
        Mat imgObject = imread("C:\\Users\\Duke\\Desktop\\test\\2_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        Mat imgScene = imread("C:\\Users\\Duke\\Desktop\\test\\1_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        //        imshow("object",imgObject);
        //        imshow("scene",imgScene);
        if (!imgObject.data || !imgScene.data)
        {

            //ui->textLog->append( " --(!) Error reading images " );
        }

        //double begin = clock();

        ///-- Step 1: 使用SURF算子检测特征点
        //int minHessian = 500;
        //SurfFeatureDetector detector(minHessian);//opencv 2.4.x
        Ptr<SURF> detector = SURF::create();//opencv 3.x.x
        vector<KeyPoint> keypointsObject, keypointsScene;
        //detector.detect(imgObject, keypointsObject);//opencv 2.4.x
        detector->detect(imgObject, keypointsObject);//opencv 3.x.x
        detector->detect(imgScene, keypointsScene);
        //        qDebug() << "object--number of keypoints: " << keypointsObject.size() << endl;
        //        qDebug() << "scene--number of keypoints: " << keypointsScene.size() << endl;

        ///-- Step 2: 使用SURF算子提取特征（计算特征向量）
        //SurfDescriptorExtractor extractor;//opencv 2.4.x
        Ptr<SURF> extractor = SURF::create();//opencv 3.x.x
        Mat descriptorsObject, descriptorsScene;
        extractor->compute(imgObject, keypointsObject, descriptorsObject);
        extractor->compute(imgScene, keypointsScene, descriptorsScene);

        ///-- Step 3: 使用FLANN法进行匹配
        FlannBasedMatcher matcher;
        vector< DMatch > allMatches;
        matcher.match(descriptorsObject, descriptorsScene, allMatches);
        //        qDebug() << "number of matches before filtering: " << allMatches.size() << endl;

        //-- 计算关键点间的最大最小距离
        double maxDist = 0;
        double minDist = 100;
        for (int i = 0; i < descriptorsObject.rows; i++)
        {
            double dist = allMatches[i].distance;
            if (dist < minDist)
                minDist = dist;//0.1
            if (dist > maxDist)
                maxDist = dist;//0.6
        }
//        printf("	max dist : %f \n", maxDist);
//        printf("	min dist : %f \n", minDist);

        //-- 过滤匹配点，保留好的匹配点（这里采用的标准：distance<3*minDist）
        vector< DMatch > goodMatches;
        for (int i = 0; i < descriptorsObject.rows; i++)
        {
            if (allMatches[i].distance < 3 * minDist){
                goodMatches.push_back(allMatches[i]);
            }
        }
        //qDebug() << "number of matches after filtering: " << goodMatches.size() << endl;

        //-- 显示匹配结果
        Mat resultImg;
        drawMatches(imgObject, keypointsObject, imgScene, keypointsScene,
                    goodMatches, resultImg, Scalar::all(-1), Scalar::all(-1), vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS //不显示未匹配的点
                    );
        //-- 输出匹配点的对应关系
        //        for (int i = 0; i < goodMatches.size(); i++)
        //            ui->textLog->append(QObject::tr("good match %1: keypointsObject [%2]  -- keypointsScene [%3]\n").arg(i).arg(goodMatches[i].queryIdx).arg(goodMatches[i].trainIdx));

        ///-- Step 4: 使用findHomography找出相应的透视变换//获取满足Ratio Test的最小匹配的距离KNN算法
        vector<Point2f> object;
        vector<Point2f> scene;
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            //-- 从好的匹配中获取关键点: 匹配关系是关键点间具有的一 一对应关系，可以从匹配关系获得关键点的索引
            //-- e.g. 这里的goodMatches[i].queryIdx和goodMatches[i].trainIdx是匹配中一对关键点的索引
            object.push_back(keypointsObject[goodMatches[i].queryIdx].pt);
            scene.push_back(keypointsScene[goodMatches[i].trainIdx].pt);
        }
        Mat H = findHomography(object, scene, CV_RANSAC);//单应性

        ////一种新的透视变换判断算子
        //Mat H;
        refineMatchesWithHomography(keypointsObject, keypointsScene,1.2, goodMatches, H);
        ///-- Step 5: 使用perspectiveTransform映射点群，在场景中获取目标位置
        std::vector<Point2f> objCorners(4);
        objCorners[0] = cvPoint(0, 0);
        objCorners[1] = cvPoint(imgObject.cols, 0);
        objCorners[2] = cvPoint(imgObject.cols, imgObject.rows);
        objCorners[3] = cvPoint(0, imgObject.rows);
        std::vector<Point2f> sceneCorners(4);
        perspectiveTransform(objCorners, sceneCorners, H);

        //-- 在被检测到的目标四个角之间划线;

        line(resultImg, sceneCorners[0] + Point2f(imgObject.cols, 0), sceneCorners[1] + Point2f(imgObject.cols, 0), Scalar(0, 255, 0), 4);
        line(resultImg, sceneCorners[1] + Point2f(imgObject.cols, 0), sceneCorners[2] + Point2f(imgObject.cols, 0), Scalar(0, 255, 0), 4);
        line(resultImg, sceneCorners[2] + Point2f(imgObject.cols, 0), sceneCorners[3] + Point2f(imgObject.cols, 0), Scalar(0, 255, 0), 4);
        line(resultImg, sceneCorners[3] + Point2f(imgObject.cols, 0), sceneCorners[0] + Point2f(imgObject.cols, 0), Scalar(0, 255, 0), 4);

        //-- 显示检测结果
        imshow("detection result", resultImg);

        //double end = clock();
        //qDebug() << "\nSURF--elapsed time: " << (end - begin) / CLOCKS_PER_SEC * 1000 << " ms\n";
        goodMatches.clear();
        resultImg.release();
        waitKey(0);
    }
    if(a==52){
        //------------【1】读取源图像并检查图像是否读取成功------------
        Mat srcImage = imread("C:\\Users\\Duke\\Desktop\\test\\2_2.jpg");
        if (!srcImage.data)
        {
            cout << "读取图片错误，请重新输入正确路径！\n";
            system("pause");
            return -1;
        }
        //imshow("src", srcImage);
        //------------【2】灰度转换--------------------------
        Mat srcGray;
        cvtColor(srcImage, srcGray, CV_RGB2GRAY);
        //imshow("gray", srcGray);
        //------------【3】初始化相关变量--------------------
        Mat dstTempImage1, dstTempImage2, dstImage;
        const int maxVal = 255;     //预设最大值
        int low_threshold = 100;     //较小的阈值量
        int high_threshold = 160;   //较大的阈值量
        //--------------【4】双阈值化过程-----------------------
        //小阈值对源灰度图像进行二进制阈值化操作
        threshold(srcGray, dstTempImage1, low_threshold, maxVal, THRESH_BINARY);
        //imshow("yuzhi", dstTempImage1);
        //大阈值对源灰度图像进行反二进制阈值化操作
        threshold(srcGray, dstTempImage2, high_threshold, maxVal, THRESH_BINARY_INV);
        //imshow("fanxiangyuzhi", dstTempImage2);
        //矩阵"与运算"得到二值化结果
        bitwise_and(dstTempImage1, dstTempImage2, dstImage); //对像素加和
        imshow("shuangyuzhi", dstImage);
        imwrite("C:\\Users\\Duke\\Desktop\\test\\2dst2.jpg",dstImage);
        waitKey(0); //窗口保持等待
    }
    if(a==53){
        src = imread("C:\\Users\\Duke\\Desktop\\test\\1test2.jpg");
        if (src.empty())
        {
            printf("can not load image \n");
            return -1;
        }
        namedWindow("input", WINDOW_AUTOSIZE);
        imshow("input", src);

        namedWindow("output", WINDOW_AUTOSIZE);
        createTrackbar("minHessian","output",&minHessian, 15000, trackBar);

        waitKey(0);
    }
    else{
                }
    }
}
void Trackbar(int,void*)
{
    Mat dst,dst8u,dstshow,imageSource;
    dst=Mat::zeros(image.size(),CV_32FC1);
    imageSource=image.clone();
    /*参数的意义如下：
    *第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可，且需为单通道8位或者浮点型图像。
    * 第二个参数，OutputArray类型的dst，函数调用后的运算结果存在这里，即这个参数用于存放Harris角点检测的输出结果，和源图片有一样的尺寸和类型。
    * 第三个参数，int类型的blockSize，表示邻域的大小，更多的详细信息在cornerEigenValsAndVecs（）中有讲到。
    * 第四个参数，int类型的ksize，表示Sobel()算子的孔径大小。
    * 第五个参数，double类型的k，Harris参数。
    * 第六个参数，int类型的borderType，图像像素的边界模式，注意它有默认值BORDER_DEFAULT。
    */

    cornerHarris(imageGray,dst,5,5,0.04,BORDER_DEFAULT);
    normalize(dst,dst8u,0,255,CV_MINMAX);  //归一化
    convertScaleAbs(dst8u,dstshow);
    imshow("dst",dst);
    //imshow("dstshow",dstshow);  //dst显示
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            if(dstshow.at<uchar>(i,j)>thresh)  //阈值判断
            {
                circle(imageSource,Point(j,i),2,Scalar(0,0,255),2); //标注角点
            }
        }
    }
    imshow("Corner Detected",imageSource);

}
void trackBar(int, void*)
{
    Mat dst;
    // SURF特征检测
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> keypoints;
    detector->detect(src, keypoints, Mat());
    // 绘制关键点
    drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("output", dst);
}
//寻找源图与目标图像之间的透视变换https://blog.csdn.net/hust_bochu_xuchao/article/details/52153167
bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints,
                                 float reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography)
{
    const int minNumberMatchesAllowed = 8;
    if (matches.size() < minNumberMatchesAllowed)
        return false;
    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
        //srcPoints[i] = trainKeypoints[i].pt;
        //dstPoints[i] = queryKeypoints[i].pt;
    }
    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
    std::vector<cv::DMatch> inliers;
    //    omp_set_num_threads(4);//多线程并行
    //#pragma omp parallel for
    for (size_t i = 0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}
//得到匹配点后，就可以使用OpenCV3.0中新加入的函数findEssentialMat()来求取本征矩阵了。得到本征矩阵后，再使用另一个函数对本征矩阵进行分解，并返回两相机之间的相对变换R和T。

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
    //根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
    double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
    Point2d principle_point(K.at<double>(2), K.at<double>(5));

    //根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
    Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return false;

    double feasible_count = countNonZero(mask);
    //        qDebug() << (int)feasible_count << " -in- " << p1.size() << endl;
    //对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
    if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
        return false;

    //分解本征矩阵，获取相对变换
    int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

    //同时位于两个相机前方的点的数量要足够大
    if (((double)pass_count) / feasible_count < 0.7)
        return false;

    return true;
}
