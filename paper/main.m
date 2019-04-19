%% 清理环境
clc;
clear;
close all;

%% 导入相机参数
load('C:/Users/Duke/Desktop/cameraParm.mat');


%% 尼康相机的内参数初始化
% [400.4469         0         0
%        0     400.4469       0
%        0         0       1.0000];
                                                          
%% 导入图像数据
disp('load data!\n');

%for i=1:15
%    image_dir{i}=['C:/Users/Duke/Desktop/MeshRecon/img/','1 (',num2str(i),')','.jpg'];
%    I = imread(image_dir{i});
%    images{i} = rgb2gray(I);
%end

%% 处理参考图像
%I = images{1};
I = rgb2gray(imread('C:\Users\Duke\Desktop\MeshRecon\img\1 (1).jpg'))
prevPoints   = detectSURFFeatures(I);

% 特征点的显示
 figure
 imshow(I); hold on;
 plot(prevPoints.selectStrongest(200));

% 特征描述子
prevFeatures = extractFeatures(I, prevPoints);

% Create an empty viewSet object to manage the data associated with each
% view.
vSet = viewSet;


% 第一幅（即参考图像的构造）以及相机位姿的建立
viewId = 1;
vSet = addView(vSet, viewId, 'Points', prevPoints, 'Orientation', eye(3),...
    'Location', [0 0 0]);

%% 依次添加剩下的图像
for i=2:numel(images)
    % 特征点检测
    I = images{i};
    currPoints   = detectSURFFeatures(I);
   
    % 特征点的显示
%     figure;
%     imshow(I); hold on;
%     plot(currPoints.selectStrongest(200));
    
    % 特征描述
    currFeatures = extractFeatures(I, currPoints);
    indexPairs = matchFeatures(prevFeatures, currFeatures);

    % 匹配点的位置信息
    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = currPoints(indexPairs(:, 2));
    
    % 匹配点的坐标
    matchedPoints1_Loca = matchedPoints1.Location;
    matchedPoints2_Loca = matchedPoints2.Location;
    
    % 初始匹配之后的图像
    figure; 
    showMatchedFeatures(images{i-1},images{i},matchedPoints1,matchedPoints2);
    title('Candidate matched points (including outliers)')
    
    % 几何约束信息估计的计算
    [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, cameraParams);

    % 添加当前视图到视图集合中
    vSet = addView(vSet, i, 'Points', currPoints);
    
    % 当前匹配的特征点
    vSet = addConnection(vSet, i-1, i, 'Matches', indexPairs(inlierIdx,:));
    %vSet = addConnection(vSet, i-1, i, 'Matches', indexPairs(:,:));
     
    % 相机位姿
    prevPose = poses(vSet, i-1);
    prevOrientation = prevPose.Orientation{1};
    prevLocation    = prevPose.Location{1};
    
    % 计算相机的位姿相对于全局坐标系统
    orientation = prevOrientation * relativeOrient;
    location    = prevLocation + relativeLoc * prevOrientation;
    vSet = updateView(vSet, i, 'Orientation', orientation, ...
        'Location', location);
    
    % 迹的寻找
    % 匹配所有的图像
    tracks = findTracks(vSet);
    
    % 获取相机的所有位姿，用于初始重建
    camPoses = poses(vSet);
    
    % 初始重建，三角化
    xyzPoints = triangulateMultiview(tracks, camPoses, cameraParams);
    
    % 修正点的位置和相机参数
    % 捆集调整
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
        'PointsUndistorted', true);
    
    % 更新和存储相机的位姿
    vSet = updateView(vSet, camPoses);

    % 下一次的处理
    prevFeatures = currFeatures;
    prevPoints   = currPoints;
end


%% 可视化
% 相机位置的显示
camPoses = poses(vSet);
figure;
helperPlotCameras(camPoses);

% Exclude noisy 3-D points.
goodIdx = (reprojectionErrors < 5);
xyzPoints = xyzPoints(goodIdx, :);

% 三维点的显示
pcshow(xyzPoints, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on;
title('Refined Camera Poses');


%% 稠密化
% 检测角点
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

%% 稠密化显示
% Display the refined camera poses.
figure;
helperPlotCameras(camPoses);

% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 5);

% Display the dense 3-D world points.
pcshow(xyzPoints(goodIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on;

