clear;
clc;
ima=rgb2gray(imread('guanceng.jpg'));
imshow(ima);
 k=2;
[mu,mask]=kmeans(ima,k);
figure,imshow(mask,[]);title('k=2');
