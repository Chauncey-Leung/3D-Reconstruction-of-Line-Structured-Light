clc;
clear;
close all;

%% ������
%��������
%Ԥ����������Ը��õ���ֲ�������ǿ�ɶ���
%����ʹ��matlab�Դ�app ��Camera Calibrator�������̸�ͼƬ���б궨���õ�������������
%���궨������룬���й�ƽ��궨
%���ù�ƽ��궨�����ɨ��ͼƬ������������
%���ñ궨������ת�ᣬ����ģ�͡�

%% Ԥ�������
%ע���������ͼƬҪ����Ӧ�Ĵ��������Ƭ
%��������ϵ��y����x��Ӧ��������ת��һ��
%Ҳ����ѡ������ͼƬ���޸ĳ�����World��Temporary����ż���
NUMBEROFPHOTO = 38;         %�궨ʱʹ����Ƭ������
WORLD = NUMBEROFPHOTO;    %��������������ϵ��ͼƬ���
TEMPORARY = NUMBEROFPHOTO-1;  %��������ʱ����ϵ��ͼƬ���
LENGTHOFCHECKERBOARD = 5;   %��궨ʱ�������һ��
PICTURENUM = 200;           %ɨ��ͼƬ����
THETAINDEX = 1.8;           %�������ÿ�β����Ƕ��� ��λ��
                            %PICTURENUM * THETAINDEX = 360����άģ�Ϳɱպ�

INDEX = 15;             %��ת����Y���������

%·��
WorldCenterLineData = load('..\Image\LightPlaneCalibration\CenterPoint_f\Cali_world_las.txt');
TempCenterLineData = load('..\Image\LightPlaneCalibration\CenterPoint_f\Cali_temp_las.txt');
ScanPath = '..\Image\Scan\CenterPoint_f\';
txtFileName = 'Bell';
SavePath = '..\';
SaveName = 'coordinate.txt';
%�������������   ��һ��x �ڶ���y
%WorldChessPoint ��һ��y �ڶ���x

% ��ת��궨����
C=[ 12.5281  36.6011 -17.6365];
x0=C(1);
y0=C(2);
z0=C(3);
n = [ 4.7500e-04  9.4243e-01 -3.3440e-01];
nx=n(1);
ny=n(2);
nz=n(3);

%%  �궨������� **��ת�þ���**
load('cameraParams.mat');
intriMatrix = cameraParams.IntrinsicMatrix';    %�ڲ�
%  World coordinate system
R = cameraParams.RotationMatrices(:, :, WORLD)';    %��ת����
T = cameraParams.TranslationVectors(WORLD, :)';     %ƽ�ƾ���
%  Temporary coordinate system
R_t = cameraParams.RotationMatrices(:, :, TEMPORARY)';
T_t = cameraParams.TranslationVectors(TEMPORARY, :)';

%%  �����ƽ�淽��
WorldChessPoint = cameraParams.ReprojectedPoints(:, :, WORLD);
TempChessPoint = cameraParams.ReprojectedPoints(:, :, TEMPORARY);

% Ϊ�˱����в�������̨���Ĳ������̸�ƽ��Ĺ���Ӱ�죬ȡ�м�ĵ�
WorldXYZ  = mygetPoints(WorldCenterLineData, 0.3, 0.7, WorldChessPoint, LENGTHOFCHECKERBOARD);
TempXYZ_t = mygetPoints(TempCenterLineData,  0.3, 0.7, TempChessPoint,  LENGTHOFCHECKERBOARD);
TempXYZ_w = Temp2World(TempXYZ_t,R_t,T_t,R,T);
%��С���˷����ƽ��
[a,b,c,d] = fitLightPlane([WorldXYZ;TempXYZ_w],0);

%%  ���ɵ���
WCS = [];          % ���е������
for frame_index = 1 : PICTURENUM
    fprintf('   Processing %d th image...\n', frame_index);
    laserPixel = load([ScanPath,txtFileName,num2str(frame_index),'.txt']);
    for pixel_index = 1 : length(laserPixel)
        [Xw, Yw, Zw] = pcs2wcs(laserPixel(pixel_index, 2), laserPixel(pixel_index,1), intriMatrix, R, T, a, b, c, d);
        % û����ת��궨
%         r = sqrt((Yw - INDEX)^2 + Zw^2);
%         theta = atan2(-Zw, INDEX - Yw);
%         Yw = r * cos(theta - frame_index/PICTURENUM * 2 * pi);
%         Zw = r * sin(theta - frame_index/PICTURENUM * 2 * pi);
%         WCS = [WCS; [Xw, Yw, Zw]];
        %��ת��궨
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
%��ά�ؽ����굼��
save([SavePath,SaveName], 'WCS', '-ascii');
fprintf('   Processing Complete�� Saved as ��coordinate.txt�� \n');
