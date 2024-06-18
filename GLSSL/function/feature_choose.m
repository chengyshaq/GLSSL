function [output_index,output_data] = feature_choose(data,temp,number)
%FEATURE_CHOOSE 此处显示有关此函数的摘要
%   此处显示详细说明
[order,index] = sort(temp,'descend');

output_index = index(1:number);
output_data = data(:,output_index);
end

