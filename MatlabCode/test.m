%% Ԥ����άͼ��ֻ��Ϊ�˷����������������ڽ�ģ
figure;
Pixel = load('..\coordinate.txt');
x = Pixel(:, 1);
y = Pixel(:, 2);
z = Pixel(:, 3);
scatter3(x, y , z, '.', 'r');
xlabel('x');
ylabel('y');
zlabel('z');

%% �鿴��������ϵ����ʱ����ϵ����������ϳ��Ĺ�ƽ��Ĺ�ϵ
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
%��Ҫ����main.m������Ӧ��ֵ
%��һ�ο��Կ����� matlab x y�����Ƿ���
%���Էֱ���ע��������δע������
%Ҳ���Թ۲����̸����������
row = 720;	%�����ͼ�����
column = 1280;
image_1=zeros(row,column);      % �ȴ��ȫ�ڱ���
max = length(WorldChessPoint);
for index = 1 : max
    image_1=zeros(row,column);      % �ȴ��ȫ�ڱ���
    image_1(round(WorldChessPoint(index, 2)), round(WorldChessPoint(index, 1))) = 255;
	%image_1(round(WorldChessPoint(index, 1)), round(WorldChessPoint(index, 2))) = 255;
    imshow(image_1);
    pause(1);
end
figure;
imshow(image_1);


% �����Ƕ�̬�ģ������̬�ģ����Ը���Ĺ۲�
row = 720;
column = 1280;
image_2=zeros(row,column);      % �ȴ��ȫ�ڱ���
max = length(WorldChessPoint);
for index = 1 : max
    image_2(round(WorldChessPoint(index, 1)), round(WorldChessPoint(index, 2))) = 255;
end
figure;
imshow(image_2);


%% ���������۲�����ͼ��·�����ļ������úþ�����
%�ļ��н���������
ScanPath = '..\Image\Scan\CenterPoint_f\';
txtFileName = 'image';
laserPixel = load([ScanPath,txtFileName,num2str(2),'.txt']);
image_3=zeros(row,column);      % �ȴ��ȫ�ڱ���
max2 = length(laserPixel);
for index = 1 : max2
    image_3(round(laserPixel(index, 1)), round(laserPixel(index, 2))) = 255;
end
figure;
imshow(image_3);

%% 
%ͬ�ϣ����������main.m�м��س�����������
row = 720;
column = 1280;
image_4=zeros(row,column);      % �ȴ��ȫ�ڱ���
max = length(WorldCenterLineData);
for index = 1 : max
    image_4(round(WorldCenterLineData(index, 1)), round(WorldCenterLineData(index, 2))) = 255;
end
figure;
imshow(image_4);

