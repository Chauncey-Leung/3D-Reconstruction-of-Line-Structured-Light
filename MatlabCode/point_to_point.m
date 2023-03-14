load('Rotation.mat');
R=[-0.0282,0.999,-0.035;
    -0.9864,-0.0350,-0.2133
    -0.2143,0.0281,0.9764
    ];
T=[7.9358,12.5124,141.982]';
sum=21;
R_ = cameraParams.RotationMatrices;    %旋转矩阵
T_ = cameraParams.TranslationVectors;     %平移矩阵
ImagePoint = cameraParams.ReprojectedPoints;
TempPoint= zeros(21,3,56);
WorldPoint = zeros(size(TempPoint));
LENGTHOFCHECKERBOARD=5;
num=0;
P=zeros(21*56,3);
for j=1:56
u=ImagePoint(j,2,:);
v=ImagePoint(j,1,:);
%选择除了第一个点的任意一点。
for i=1:sum
    TempPoint(i,:,j) = [cameraParams.WorldPoints(j,1) cameraParams.WorldPoints(j,2) 0];
    WorldPoint(i,:,j) = Temp2World(TempPoint(i,:,j),R_(:,:,i)',T_(i,:)',R,T);
    num=num+1;
    P(num,:)=WorldPoint(i,:,j);
end
end
save('..\Image\RotationAxisCalibration\RotationAxisWorldPoint.txt', 'P', '-ascii');
save('..\Image\RotationAxisCalibration\P.txt', 'P', '-ascii');