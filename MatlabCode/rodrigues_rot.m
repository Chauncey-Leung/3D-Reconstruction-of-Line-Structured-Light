function P_rot=rodrigues_rot(P, n0, n1)
%      P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
%     :param P: 三维点坐标集合
%     :param n0: 原平面法向量
%     :param n1: 最佳拟合平面法向量
%     :return P_rot: 旋转后的点集合
%     """
%     # 归一化
    n0 = n0 /normest(n0);
    n1 = n1 /normest(n1);
%     # k为旋转轴
    k = cross(n0,n1);
    k = k /normest(k);
%     # theta为旋转角度
    theta = acos(dot(n0, n1));

    P_rot = P * cos(theta) + cross(k, P) * sin(theta) + k * dot(k, P) * (1 - cos(theta));
end
