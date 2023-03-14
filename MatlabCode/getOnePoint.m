function [x,y,z] = getOnePoint(ChessPoint, RowIndex, ColumnIndex ,RealLengthOfCheckerBoard)
%delta_row��������ľ��루��ͼ������ϵ�У�
%���delta_row ����ȡƽ��ֵ ��׼ȷ��Ŀǰ����Ҳ��
%���delta_row������Ҫע��һ�£�ÿ�в�һ����5���㣬ȡ�����ƽ��ֵ���С�㡣
%�����������±�����ע�͵��Ľ����滻��
%delta_row = sqrt((ChessPoint(1, 1) - ChessPoint(2, 1))^2 + (ChessPoint(1, 2) - ChessPoint(2, 2))^2) / 1;
delta_row = sqrt((ChessPoint(1, 1) - ChessPoint(5, 1))^2 + (ChessPoint(1, 2) - ChessPoint(5, 2))^2) / 4;
x = (ChessPoint(1, 2) - RowIndex) / delta_row * RealLengthOfCheckerBoard;
y = (ColumnIndex - ChessPoint(1, 1)) / delta_row * RealLengthOfCheckerBoard;
z = 0;
end