% The RGB to gray
path='D:7-21-simplewaystems11HOG adaboost';
file='*.tif';
pic=dir([path,file]);
filename=str2mat(pic.name); %取得文件名
%调用函数
sortfile=LengthSortStr(filename);
num=size(pic,1);
image_filename='D:7-21-simplewaystems11HOG adaboost';
mkdir(image_filename,'stemsgray')
output_default_path = fullfile(image_filename,'stemsgray');
cd(output_default_path)
% message = 'Select the output folder';
% uiwait(msgbox(message));
% output_folder = uigetdir(output_default_path);
for i=1:num; 
    tiff{i}=imread([path,sortfile(i,:)]);
    images{i}=rgb2gray(tiff{i}); % color to gray
    images{i}=imadjust(images{i}); % ajusting image quality
    filenames= strcat('stemsgray',num2str(i),'.tif');
    imwrite(images{i},filenames)
end


% The gray to bw
clear;clc
path='D:7-21-simplewaystems11HOG adabooststemsgray';
file='*.tif';
pic=dir([path,file]);% read the number of images 
filename=str2mat(pic.name); %obtain filenames
sortfile=LengthSortStr(filename);% sort function
num=size(pic,1);
image_filename='D:7-21-simplewaystems11HOG adaboost';
mkdir(image_filename,'stemsgraybw')
output_default_path = fullfile(image_filename,'stemsgraybw');
cd(output_default_path)
for i=1:num; 
    tiff{i}=imread([path,sortfile(i,:)]); 
    images{i}=im2bw(tiff{i},0.5)
    filenames= strcat('wbjin',num2str(i),'.tif');
    imwrite(images{i},filenames);
end



% The RGB to PCA

path='D:simplewaystemssimplywaycodebadresults';
file='*.tif';
pic=dir([path,file]);
filename=str2mat(pic.name); %取得文件名
%调用函数
sortfile=LengthSortStr(filename);
num=size(pic,1);
tiff={};
for i=1:num; 
tiff{i}=imread([path,sortfile(i,:)]); 
R(:,:)=tiff{i}(:,:,1); % 3D to 2D
G(:,:)=tiff{i}(:,:,2);
B(:,:)=tiff{i}(:,:,3);
[m n]=size(R);% size of 2D
R1=reshape(R,prod(size(R)),1); % Multi-row and Single-column
G1=reshape(G,prod(size(G)),1);
B1=reshape(B,prod(size(B)),1);
RGB1=[R1,G1,B1]; % three columns of RGB
[coeff,score,latent] = pca(double(RGB1));% PCA caculation
score1=score(:,1);% the first principal is selected.
% percetanges=cumsum(latent)./sum(latent);
pcaiamges{i}=reshape(score1,m,n);
pcaiamges1{i}=im2bw(pcaiamges{i})
filenames= strcat('bw',num2str(i),'.tif');
imwrite(pcaiamges1{i},filenames);
end
