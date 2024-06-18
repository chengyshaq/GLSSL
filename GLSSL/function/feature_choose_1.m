function [output_data,output_index] = feature_choose_1(X,data,temp,f_number,number)
%FEATURE_CHOOSE_1 此处显示有关此函数的摘要
%   此处显示详细说明
len2 = length(f_number);
[order,index] = sort(temp,'descend');
for j = 1:len2
    a = f_number(j);
    index(index == a) = [];
end

output_index = index(1:number);
output_data = X(:,output_index);
end

