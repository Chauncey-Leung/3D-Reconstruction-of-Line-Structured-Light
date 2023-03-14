function [Xw, Yw, Zw] = pcs2wcs(u, v, iM, R, T, a, b, c, d)
%����任��ϵ����ƽ�淽�� 4������4��δ֪����ֱ�ӽ⼴��
%δ֪��Xw,Yw,Zw,Zc
%����matlab app ��xy����������ʱ��Ե�u��v�����ʱ�ٻ�������
M1 = [iM, [0; 0; 0]];
M2 = [R, T; 0, 0, 0, 1];
M = M1 * M2;

H =[ [M(1:3,1:3);a, b, c], [-u, -v, -1, 0]' ];
Result = [-M(1,4), -M(2,4), -M(3,4), -d]';
WCS = H \ Result;

%��������x y���Ҫ��ת
Xw = WCS(2, 1);
Yw = WCS(1, 1);
Zw = WCS(3, 1);
end