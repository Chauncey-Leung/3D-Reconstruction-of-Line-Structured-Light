function [Xw, Yw, Zw] = pcs2wcs(u, v, iM, R, T, a, b, c, d)
%坐标变换关系＋光平面方程 4个方程4个未知数，直接解即可
%未知数Xw,Yw,Zw,Zc
%由于matlab app 中xy互换，输入时需对调u，v，输出时再换回来。
M1 = [iM, [0; 0; 0]];
M2 = [R, T; 0, 0, 0, 1];
M = M1 * M2;

H =[ [M(1:3,1:3);a, b, c], [-u, -v, -1, 0]' ];
Result = [-M(1,4), -M(2,4), -M(3,4), -d]';
WCS = H \ Result;

%输出结果，x y结果要反转
Xw = WCS(2, 1);
Yw = WCS(1, 1);
Zw = WCS(3, 1);
end