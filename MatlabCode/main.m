clc;
clear;
close all;

%% 主函数
%操作介绍
%预定义参数可以更好的移植代码和增强可读性
%首先使用matlab自带app “Camera Calibrator”对棋盘格图片进行标定，得到相机的内外参数
%将标定结果导入，进行光平面标定
%利用光平面标定结果和扫描图片生成世界坐标
%利用标定出的旋转轴，建立模型。

%% 预定义参数
%注意最后两张图片要有相应的带激光的照片
%世界坐标系中y或者x轴应尽量与旋转轴一致
%也可以选择其他图片，修改程序中World和Temporary的序号即可
NUMBEROFPHOTO = 38;         %标定时使用照片的总数
WORLD = NUMBEROFPHOTO;    %用来做世界坐标系的图片序号
TEMPORARY = NUMBEROFPHOTO-1;  %用来做临时坐标系的图片序号
LENGTHOFCHECKERBOARD = 5;   %与标定时输入参数一致
PICTURENUM = 200;           %扫描图片数量
THETAINDEX = 1.8;           %步进电机每次步进角度量 单位°
                            %PICTURENUM * THETAINDEX = 360，三维模型可闭合

INDEX = 15;             %旋转轴与Y坐标轴距离

%路径
WorldCenterLineData = load('..\Image\LightPlaneCalibration\CenterPoint_f\Cali_world_las.txt');
TempCenterLineData = load('..\Image\LightPlaneCalibration\CenterPoint_f\Cali_temp_las.txt');
ScanPath = '..\Image\Scan\CenterPoint_f\';
txtFileName = 'Bell';
SavePath = '..\';
SaveName = 'coordinate.txt';
%导入进来的坐标   第一列x 第二列y
%WorldChessPoint 第一列y 第二列x

% 旋转轴标定参数
C=[ 12.5281  36.6011 -17.6365];
x0=C(1);
y0=C(2);
z0=C(3);
n = [ 4.7500e-04  9.4243e-01 -3.3440e-01];
nx=n(1);
ny=n(2);
nz=n(3);

%%  标定结果导入 **需转置矩阵**
load('cameraParams.mat');
intriMatrix = cameraParams.IntrinsicMatrix';    %内参
%  World coordinate system
R = cameraParams.RotationMatrices(:, :, WORLD)';    %旋转矩阵
T = cameraParams.TranslationVectors(WORLD, :)';     %平移矩阵
%  Temporary coordinate system
R_t = cameraParams.RotationMatrices(:, :, TEMPORARY)';
T_t = cameraParams.TranslationVectors(TEMPORARY, :)';

%%  计算光平面方程
WorldChessPoint = cameraParams.ReprojectedPoints(:, :, WORLD);
TempChessPoint = cameraParams.ReprojectedPoints(:, :, TEMPORARY);

% 为了避免有部分载物台或别的不在棋盘格平面的光条影响，取中间的点
WorldXYZ  = mygetPoints(WorldCenterLineData, 0.3, 0.7, WorldChessPoint, LENGTHOFCHECKERBOARD);
TempXYZ_t = mygetPoints(TempCenterLineData,  0.3, 0.7, TempChessPoint,  LENGTHOFCHECKERBOARD);
TempXYZ_w = Temp2World(TempXYZ_t,R_t,T_t,R,T);
%最小二乘法拟合平面
[a,b,c,d] = fitLightPlane([WorldXYZ;TempXYZ_w],0);

%%  生成点云
WCS = [];          % 所有点的坐标
for frame_index = 1 : PICTURENUM
    fprintf('   Processing %d th image...\n', frame_index);
    laserPixel = load([ScanPath,txtFileName,num2str(frame_index),'.txt']);
    for pixel_index = 1 : length(laserPixel)
        [Xw, Yw, Zw] = pcs2wcs(laserPixel(pixel_index, 2), laserPixel(pixel_index,1), intriMatrix, R, T, a, b, c, d);
        % 没有旋转轴标定
%         r = sqrt((Yw - INDEX)^2 + Zw^2);
%         theta = atan2(-Zw, INDEX - Yw);
%         Yw = r * cos(theta - frame_index/PICTURENUM * 2 * pi);
%         Zw = r * sin(theta - frame_index/PICTURENUM * 2 * pi);
%         WCS = [WCS; [Xw, Yw, Zw]];
        %旋转轴标定
        P= [Yw,Xw, Zw, 1];
        alpha=frame_index/PICTURENUM * 2 * pi;
        K = 1 - cos(alpha);
        M1 = nx * x0 + ny * y0 + nz * z0;
        T1 = [nx^2 * K + cos(alpha),  nx * ny * K - nz * sin(alpha),  nx * nz * K + ny * sin(alpha),  (x0 - nx * M1) * K + (nz * y0 - ny * z0) * sin(alpha)
        nx * ny * K + nz * sin(alpha),  ny^2 * K + cos(alpha),  ny * nz * K - nx * sin(alpha),  (y0 - ny * M1) * K + (nx * z0 - nz * x0) * sin(alpha)
        nx * nz * K - ny * sin(alpha),  ny * nz * K + nx * sin(alpha),  nz^2 * K + cos(alpha),  (z0 - nz * M1) * K + (ny * x0 - nx * y0) * sin(alpha)
        0, 0, 0, 1];
        P_rot=T1*P';
        WCS = [WCS; [P_rot(1) ,P_rot(2), P_rot(3)]];
    end
end
%三维重建坐标导出
save([SavePath,SaveName], 'WCS', '-ascii');
fprintf('   Processing Complete！ Saved as ‘coordinate.txt‘ \n');
