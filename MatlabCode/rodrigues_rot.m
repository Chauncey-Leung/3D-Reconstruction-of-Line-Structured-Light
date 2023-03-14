function P_rot=rodrigues_rot(P, n0, n1)
%      P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
%     :param P: ��ά�����꼯��
%     :param n0: ԭƽ�淨����
%     :param n1: ������ƽ�淨����
%     :return P_rot: ��ת��ĵ㼯��
%     """
%     # ��һ��
    n0 = n0 /normest(n0);
    n1 = n1 /normest(n1);
%     # kΪ��ת��
    k = cross(n0,n1);
    k = k /normest(k);
%     # thetaΪ��ת�Ƕ�
    theta = acos(dot(n0, n1));

    P_rot = P * cos(theta) + cross(k, P) * sin(theta) + k * dot(k, P) * (1 - cos(theta));
end
