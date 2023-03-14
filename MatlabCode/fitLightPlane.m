function [a,b,c,d] = fitLightPlane(Points, plotStatue)
%ƽ�淽��ax+by+cz+d=0
%��С���˷����ƽ��
if nargin < 2, plotStatue = 0;    %plotStatue,Ĭ��Ϊ0
end

DataSize = length(Points);
x=Points(:,1);                       %������x����
y=Points(:,2);                       %������y����
z=Points(:,3);                       %������z����
x_avr = sum(x) / DataSize;       % ��ȡ��������ϵ������ͬ
y_avr = sum(y) / DataSize;
z_avr = sum(z) / DataSize;
xx_avr = sum(x.*x) / DataSize;
yy_avr = sum(y.*y) / DataSize;
zz_avr = sum(z.*z) / DataSize;
xy_avr = sum(x.*y) / DataSize;
xz_avr = sum(x.*z) / DataSize;
yz_avr = sum(y.*z) / DataSize;
% A * X = B
% ���������ϵ������A
A=[xx_avr xy_avr x_avr;
   xy_avr yy_avr y_avr;
   x_avr y_avr 1];
B=[xz_avr;yz_avr;z_avr];
Solution=A \ B;                     %��ȡϵ��
a = Solution(1);
b = Solution(2);
c = -1;
d = Solution(3);

%��һ��
a = a / d;
b = b / d;
c = c / d;
d = d / d;
%% ��ʾ��ƽ��
if plotStatue ~= 0
    y_buff = linspace(-50, 50, 101);
    x_buff = linspace(0, 50, 51);
    [X_buff,Y_buff] = meshgrid(x_buff, y_buff);
    Z_buff = -((a/c)*X_buff + (b / c)*Y_buff + d/c);
    mesh(X_buff,Y_buff,Z_buff);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('����ƽ��');
    clear x_buff y_buff X_buff Y_buff Z_buff
end
end