load('cameraParams.mat');
%��ɨ������ͼƬȥ����͡��ü���

RowScanPath = '..\Image\Scan\RawImage\';
CorrectedSavePath = '..\Image\Scan\BellImage\';
RowFileName = 'RawImage';
CorrectedFileName = 'Bell';
NUMOFSCANIMAGE = 200;
%% ͼ�����
for i = 1:NUMOFSCANIMAGE
    I=rgb2gray(imread([RowScanPath,RowFileName,num2str(i),'.jpg']));
    J = undistortImage(I,cameraParams);
     %�򵥵Ĳü���Ϊ�˸��õ���ȡ��ֻ��ע�����䵽�Ĳ��֣�������Ҫ��
    J(:,1:220) = 0;    
    J(:,580:1280) = 0;
    J(1:250,:) = 0;
    J(540:720,:) = 0;
    imwrite(J,[CorrectedSavePath,CorrectedFileName,num2str(i),'.jpg']);
end

