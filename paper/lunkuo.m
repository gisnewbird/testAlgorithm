%��ȡԭͼ   
img = imread( 'guanceng_re.jpg' );
%img = imread( 'C:\Users\Duke\Desktop\1.jpg' );                                            
img=rgb2gray(img);
h=fspecial('gaussian',5);%?��˹�˲�
BW6=edge(img,'canny');%��˹�˲���ʹ��Canny���ӽ��б�Ե���
imshow(BW6)
% ת��ֵͼ��    
bw = im2bw( BW6 );                           
    
 %������ȡ       
contour = bwperim(bw);                      
figure    
imshow(contour);    
title('����')  
imwrite(contour,'contour.jpg');