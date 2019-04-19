%读取原图   
img = imread( 'guanceng_re.jpg' );
%img = imread( 'C:\Users\Duke\Desktop\1.jpg' );                                            
img=rgb2gray(img);
h=fspecial('gaussian',5);%?高斯滤波
BW6=edge(img,'canny');%高斯滤波后使用Canny算子进行边缘检测
imshow(BW6)
% 转二值图像    
bw = im2bw( BW6 );                           
    
 %轮廓提取       
contour = bwperim(bw);                      
figure    
imshow(contour);    
title('轮廓')  
imwrite(contour,'contour.jpg');