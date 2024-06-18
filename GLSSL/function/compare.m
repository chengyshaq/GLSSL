function [output] = compare(label)
%COMPARE 此处显示有关此函数的摘要
%   此处显示详细说明
len = length(label);
for i = 1:len
    for j = (i+1):len
        x = size(label{i},2);
        y = size(label{j},2);
        if y > x
            T = label{j};
            label{j} = label{i};
            label{i} = T;
        end
    end
end
output = label;
end

