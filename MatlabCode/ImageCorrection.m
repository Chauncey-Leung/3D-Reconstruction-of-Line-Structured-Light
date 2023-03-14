load('cameraParams.mat');
%将扫描所得图片去畸变和“裁剪”

RowScanPath = '..\Image\Scan\RawImage\';
CorrectedSavePath = '..\Image\Scan\BellImage\';
RowFileName = 'RawImage';
CorrectedFileName = 'Bell';
NUMOFSCANIMAGE = 200;
%% 图像矫正
for i = 1:NUMOFSCANIMAGE
    I=rgb2gray(imread([RowScanPath,RowFileName,num2str(i),'.jpg']));
    J = undistortImage(I,cameraParams);
     %简单的裁剪，为了更好的提取，只关注能照射到的部分，具体需要试
    J(:,1:220) = 0;    
    J(:,580:1280) = 0;
    J(1:250,:) = 0;
    J(540:720,:) = 0;
    imwrite(J,[CorrectedSavePath,CorrectedFileName,num2str(i),'.jpg']);
end

