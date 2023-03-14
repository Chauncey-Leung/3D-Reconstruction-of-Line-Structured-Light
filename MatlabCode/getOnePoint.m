function [x,y,z] = getOnePoint(ChessPoint, RowIndex, ColumnIndex ,RealLengthOfCheckerBoard)
%delta_row相邻两点的距离（在图像坐标系中）
%这个delta_row 可以取平均值 更准确，目前这样也行
%这个delta_row计算需要注意一下，每行不一定有5个点，取多个点平均值误差小点。
%不理解可以用下边这行注释掉的进行替换。
%delta_row = sqrt((ChessPoint(1, 1) - ChessPoint(2, 1))^2 + (ChessPoint(1, 2) - ChessPoint(2, 2))^2) / 1;
delta_row = sqrt((ChessPoint(1, 1) - ChessPoint(5, 1))^2 + (ChessPoint(1, 2) - ChessPoint(5, 2))^2) / 4;
x = (ChessPoint(1, 2) - RowIndex) / delta_row * RealLengthOfCheckerBoard;
y = (ColumnIndex - ChessPoint(1, 1)) / delta_row * RealLengthOfCheckerBoard;
z = 0;
end