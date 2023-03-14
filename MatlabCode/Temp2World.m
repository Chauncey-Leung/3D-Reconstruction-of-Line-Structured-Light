function XYZ = Temp2World(XYZ_t,R_t,T_t,R,T)
%XYZ 应该是 n×3 的坐标矩阵
[xyzLength,~] = size(XYZ_t);
XYZ = zeros(xyzLength,3);

% xx = XYZ_t(:,2);
% yy = XYZ_t(:,1);
% zz = XYZ_t(:,3);
% XYZ_t = [xx,yy,zz];

for index = 1:xyzLength
    CCS = R_t * XYZ_t(index,:)' + T_t;      %Temp——>Camera
    WCS = pinv(R) * (CCS - T);              %Camera——>World
    x = WCS(1, 1);
    y = WCS(2, 1);
    z = WCS(3, 1);
    XYZ(index,:) = [x,y,z];
end
end