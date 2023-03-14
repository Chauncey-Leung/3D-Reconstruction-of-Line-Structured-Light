function XYZ = mygetPoints(CenterLineData, StartPoint, EndPoint, ChessPoint, LengthOfCheckerBoard)
%取出指定区域的点坐标，并转换到世界坐标系
LineLength = length(CenterLineData);
StartIndex = round(LineLength * StartPoint);
EndIndex   = round(LineLength * EndPoint);

XYZ = zeros(EndIndex - StartIndex + 1, 3);
for index = StartIndex : EndIndex
    u = CenterLineData(index,1);
    v = CenterLineData(index,2);
    [x,y,z] = getOnePoint(ChessPoint, u, v, LengthOfCheckerBoard);  %世界坐标系下的点
    XYZ(index-StartIndex+1,:) = [x,y,z];
end

end