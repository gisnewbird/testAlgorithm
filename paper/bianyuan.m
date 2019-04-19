%I=imread('guanceng.jpg');%提取图像
I = imread('2.jpg');
img=rgb2gray(I);
[m,n]=size(img);
BW1=edge(img,'sobel'); %用Sobel算子进行边缘检测

BW2=edge(img,'roberts');%用Roberts算子进行边缘检测

BW3=edge(img,'prewitt'); %用Prewitt算子进行边缘检测

BW4=edge(img,'log'); %用Log算子进行边缘检测

BW5=edge(img,'canny'); %用Canny算子进行边缘检测

h=fspecial('gaussian',5);%?高斯滤波

BW6=edge(img,'canny');%高斯滤波后使用Canny算子进行边缘检测

subplot(2,3,1), imshow(BW1);

title('sobel edge check');

subplot(2,3,2), imshow(BW2);

title('roberts edge check');

subplot(2,3,3), imshow(BW3);

title('prewitt edge check');

subplot(2,3,4), imshow(BW4);

title('log edge check');

subplot(2,3,5), imshow(BW5);

title('canny edge check');

subplot(2,3,6), imshow(BW6);

title('gasussian&canny edge check');