%I=imread('guanceng.jpg');%��ȡͼ��
I = imread('2.jpg');
img=rgb2gray(I);
[m,n]=size(img);
BW1=edge(img,'sobel'); %��Sobel���ӽ��б�Ե���

BW2=edge(img,'roberts');%��Roberts���ӽ��б�Ե���

BW3=edge(img,'prewitt'); %��Prewitt���ӽ��б�Ե���

BW4=edge(img,'log'); %��Log���ӽ��б�Ե���

BW5=edge(img,'canny'); %��Canny���ӽ��б�Ե���

h=fspecial('gaussian',5);%?��˹�˲�

BW6=edge(img,'canny');%��˹�˲���ʹ��Canny���ӽ��б�Ե���

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