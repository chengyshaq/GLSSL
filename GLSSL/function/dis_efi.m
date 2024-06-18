function [res] = dis_efi(vec, m)
%DIS_EFI �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
row = length( vec );
split = 0:(row/m):row;
val = zeros( m, 1 );
res = sortrows( vec, 1 );
for k=1:m
    val(k) = res( round(split(k+1)), 1 );
end
val = unique( val );

res = ones( row, 1 );
for k=1:length(val)
    res( vec > val(k), 1 ) = k+1;
end

