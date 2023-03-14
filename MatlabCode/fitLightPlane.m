function [a,b,c,d] = fitLightPlane(Points, plotStatue)
%平面方程ax+by+cz+d=0
%最小二乘法拟合平面
if nargin < 2, plotStatue = 0;    %plotStatue,默认为0
end

DataSize = length(Points);
x=Points(:,1);                       %定义点的x坐标
y=Points(:,2);                       %定义点的y坐标
z=Points(:,3);                       %定义点的z坐标
x_avr = sum(x) / DataSize;       % 求取法方程组系数，下同
y_avr = sum(y) / DataSize;
z_avr = sum(z) / DataSize;
xx_avr = sum(x.*x) / DataSize;
yy_avr = sum(y.*y) / DataSize;
zz_avr = sum(z.*z) / DataSize;
xy_avr = sum(x.*y) / DataSize;
xz_avr = sum(x.*z) / DataSize;
yz_avr = sum(y.*z) / DataSize;
% A * X = B
% 法方程组的系数矩阵A
A=[xx_avr xy_avr x_avr;
   xy_avr yy_avr y_avr;
   x_avr y_avr 1];
B=[xz_avr;yz_avr;z_avr];
Solution=A \ B;                     %求取系数
a = Solution(1);
b = Solution(2);
c = -1;
d = Solution(3);

%归一化
a = a / d;
b = b / d;
c = c / d;
d = d / d;
%% 显示光平面
if plotStatue ~= 0
    y_buff = linspace(-50, 50, 101);
    x_buff = linspace(0, 50, 51);
    [X_buff,Y_buff] = meshgrid(x_buff, y_buff);
    Z_buff = -((a/c)*X_buff + (b / c)*Y_buff + d/c);
    mesh(X_buff,Y_buff,Z_buff);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('激光平面');
    clear x_buff y_buff X_buff Y_buff Z_buff
end
end