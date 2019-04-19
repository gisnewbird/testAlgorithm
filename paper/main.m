%% ������
clc;
clear;
close all;

%% �����������
load('C:/Users/Duke/Desktop/cameraParm.mat');


%% �῵������ڲ�����ʼ��
% [400.4469         0         0
%        0     400.4469       0
%        0         0       1.0000];
                                                          
%% ����ͼ������
disp('load data!\n');

%for i=1:15
%    image_dir{i}=['C:/Users/Duke/Desktop/MeshRecon/img/','1 (',num2str(i),')','.jpg'];
%    I = imread(image_dir{i});
%    images{i} = rgb2gray(I);
%end

%% ����ο�ͼ��
%I = images{1};
I = rgb2gray(imread('C:\Users\Duke\Desktop\MeshRecon\img\1 (1).jpg'))
prevPoints   = detectSURFFeatures(I);

% ���������ʾ
 figure
 imshow(I); hold on;
 plot(prevPoints.selectStrongest(200));

% ����������
prevFeatures = extractFeatures(I, prevPoints);

% Create an empty viewSet object to manage the data associated with each
% view.
vSet = viewSet;


% ��һ�������ο�ͼ��Ĺ��죩�Լ����λ�˵Ľ���
viewId = 1;
vSet = addView(vSet, viewId, 'Points', prevPoints, 'Orientation', eye(3),...
    'Location', [0 0 0]);

%% �������ʣ�µ�ͼ��
for i=2:numel(images)
    % ��������
    I = images{i};
    currPoints   = detectSURFFeatures(I);
   
    % ���������ʾ
%     figure;
%     imshow(I); hold on;
%     plot(currPoints.selectStrongest(200));
    
    % ��������
    currFeatures = extractFeatures(I, currPoints);
    indexPairs = matchFeatures(prevFeatures, currFeatures);

    % ƥ����λ����Ϣ
    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = currPoints(indexPairs(:, 2));
    
    % ƥ��������
    matchedPoints1_Loca = matchedPoints1.Location;
    matchedPoints2_Loca = matchedPoints2.Location;
    
    % ��ʼƥ��֮���ͼ��
    figure; 
    showMatchedFeatures(images{i-1},images{i},matchedPoints1,matchedPoints2);
    title('Candidate matched points (including outliers)')
    
    % ����Լ����Ϣ���Ƶļ���
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, cameraParams);

    % ��ӵ�ǰ��ͼ����ͼ������
    vSet = addView(vSet, i, 'Points', currPoints);
    
    % ��ǰƥ���������
    vSet = addConnection(vSet, i-1, i, 'Matches', indexPairs(inlierIdx,:));
    %vSet = addConnection(vSet, i-1, i, 'Matches', indexPairs(:,:));
     
    % ���λ��
    prevPose = poses(vSet, i-1);
    prevOrientation = prevPose.Orientation{1};
    prevLocation    = prevPose.Location{1};
    
    % ���������λ�������ȫ������ϵͳ
    orientation = prevOrientation * relativeOrient;
    location    = prevLocation + relativeLoc * prevOrientation;
    vSet = updateView(vSet, i, 'Orientation', orientation, ...
        'Location', location);
    
    % ����Ѱ��
    % ƥ�����е�ͼ��
    tracks = findTracks(vSet);
    
    % ��ȡ���������λ�ˣ����ڳ�ʼ�ؽ�
    camPoses = poses(vSet);
    
    % ��ʼ�ؽ������ǻ�
    xyzPoints = triangulateMultiview(tracks, camPoses, cameraParams);
    
    % �������λ�ú��������
    % ��������
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
        'PointsUndistorted', true);
    
    % ���ºʹ洢�����λ��
    vSet = updateView(vSet, camPoses);

    % ��һ�εĴ���
    prevFeatures = currFeatures;
    prevPoints   = currPoints;
end


%% ���ӻ�
% ���λ�õ���ʾ
camPoses = poses(vSet);
figure;
helperPlotCameras(camPoses);

% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 5);
xyzPoints = xyzPoints(goodIdx, :);

% ��ά�����ʾ
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on;
title('Refined Camera Poses');


%% ���ܻ�
% ���ǵ�
prevPoints = detectMinEigenFeatures(I, 'MinQuality', 0.001);

% Create the point tracker object to track the points across views.
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 6);

% Initialize the point tracker.
prevPoints = prevPoints.Location;
initialize(tracker, prevPoints, I);

% Store the dense points in the view set.
vSet = updateConnection(vSet, 1, 2, 'Matches', zeros(0, 2));
vSet = updateView(vSet, 1, 'Points', prevPoints);

% Track the points across all views.
for i = 2:numel(images)
    % Track the points.
    [currPoints, validIdx] = step(tracker, I);

    % Clear the old matches between the points.
    if i < numel(images)
        vSet = updateConnection(vSet, i, i+1, 'Matches', zeros(0, 2));
    end
    vSet = updateView(vSet, i, 'Points', currPoints);

    % Store the point matches in the view set.
    matches = repmat((1:size(prevPoints, 1))', [1, 2]);
    matches = matches(validIdx, :);
    vSet = updateConnection(vSet, i-1, i, 'Matches', matches);
end

% Find point tracks across all views.
tracks = findTracks(vSet);

% Find point tracks across all views.
camPoses = poses(vSet);

% Triangulate initial locations for the 3-D world points.
xyzPoints = triangulateMultiview(tracks, camPoses,...
    cameraParams);

% Refine the 3-D world points and camera poses.
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
    xyzPoints, tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
    'PointsUndistorted', true);

%% ���ܻ���ʾ
% Display the refined camera poses.
figure;
helperPlotCameras(camPoses);

% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 5);

% Display the dense 3-D world points.
pcshow(xyzPoints(goodIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on;

