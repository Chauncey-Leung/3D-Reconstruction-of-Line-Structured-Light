%% 预览三维图，只是为了方便快速浏览，不用于建模
figure;
Pixel = load('..\coordinate.txt');
x = Pixel(:, 1);
y = Pixel(:, 2);
z = Pixel(:, 3);
scatter3(x, y , z, '.', 'r');
xlabel('x');
ylabel('y');
zlabel('z');

%% 查看世界坐标系与临时坐标系两光条与拟合出的光平面的关系
[a,b,c,d] = fitLightPlane([WorldXYZ;TempXYZ_w],1);
hold on;
xyz = [WorldXYZ;TempXYZ_w];
x = xyz(:, 1);
y = xyz(:, 2);
z = xyz(:, 3);
h = scatter3(x, y , z, '.', 'r');
xlabel('x');
ylabel('y');
zlabel('z');
hold off
axis([0 50 -25 25 -25 25]);

%for test

%% 
%需要先跑main.m才有相应的值
%这一段可以看出来 matlab x y坐标是反的
%可以分别尝试注释内容与未注释内容
%也可以观察棋盘格坐标点的序号
row = 720;	%摄像机图像参数
column = 1280;
image_1=zeros(row,column);      % 等大的全黑背景
max = length(WorldChessPoint);
for index = 1 : max
    image_1=zeros(row,column);      % 等大的全黑背景
    image_1(round(WorldChessPoint(index, 2)), round(WorldChessPoint(index, 1))) = 255;
	%image_1(round(WorldChessPoint(index, 1)), round(WorldChessPoint(index, 2))) = 255;
    imshow(image_1);
    pause(1);
end
figure;
imshow(image_1);


% 上述是动态的，这个静态的，可以更快的观察
row = 720;
column = 1280;
image_2=zeros(row,column);      % 等大的全黑背景
max = length(WorldChessPoint);
for index = 1 : max
    image_2(round(WorldChessPoint(index, 1)), round(WorldChessPoint(index, 2))) = 255;
end
figure;
imshow(image_2);


%% 可以用来观察坐标图像，路径和文件名设置好就能跑
%文件中仅存有坐标
ScanPath = '..\Image\Scan\CenterPoint_f\';
txtFileName = 'image';
laserPixel = load([ScanPath,txtFileName,num2str(2),'.txt']);
image_3=zeros(row,column);      % 等大的全黑背景
max2 = length(laserPixel);
for index = 1 : max2
    image_3(round(laserPixel(index, 1)), round(laserPixel(index, 2))) = 255;
end
figure;
imshow(image_3);

%% 
%同上，但是这个是main.m中加载出的坐标数据
row = 720;
column = 1280;
image_4=zeros(row,column);      % 等大的全黑背景
max = length(WorldCenterLineData);
for index = 1 : max
    image_4(round(WorldCenterLineData(index, 1)), round(WorldCenterLineData(index, 2))) = 255;
end
figure;
imshow(image_4);

