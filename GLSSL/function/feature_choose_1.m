function [output_data,output_index] = feature_choose_1(X,data,temp,f_number,number)
%FEATURE_CHOOSE_1 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
len2 = length(f_number);
[order,index] = sort(temp,'descend');
for j = 1:len2
    a = f_number(j);
    index(index == a) = [];
end

output_index = index(1:number);
output_data = X(:,output_index);
end

