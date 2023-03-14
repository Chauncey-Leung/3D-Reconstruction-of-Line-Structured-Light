function XYZ = mygetPoints(CenterLineData, StartPoint, EndPoint, ChessPoint, LengthOfCheckerBoard)
%ȡ��ָ������ĵ����꣬��ת������������ϵ
LineLength = length(CenterLineData);
StartIndex = round(LineLength * StartPoint);
EndIndex   = round(LineLength * EndPoint);

XYZ = zeros(EndIndex - StartIndex + 1, 3);
for index = StartIndex : EndIndex
    u = CenterLineData(index,1);
    v = CenterLineData(index,2);
    [x,y,z] = getOnePoint(ChessPoint, u, v, LengthOfCheckerBoard);  %��������ϵ�µĵ�
    XYZ(index-StartIndex+1,:) = [x,y,z];
end

end