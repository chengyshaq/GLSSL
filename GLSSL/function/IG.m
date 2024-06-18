function [output_data,d] = IG(X,data, label,label_sum,para,maxi)
%IG 此处显示有关此函数的摘要
%   此处显示详细说明
dratio = para.dratio;
m = maxi.m;
view_num = length(data);
output_data = cell(view_num,1);
d_range = zeros(1,view_num);
for ii = 1:view_num
    d_range(1,ii) = size(data{ii},2);
end
d = floor(min(d_range)*dratio);
for vv = 1 : view_num
    f_num = size(data{vv},2);
    f_index = [];
    sum = 0;
    number = zeros(m,1);
    index = zeros(f_num,1);
    output_data{vv} = cell(m,1);
    for ii = 1 : m
        answer = label{ii};
        l_num = size(answer,2);
        if ii == m
            number(ii,1) = f_num - sum;
        else
            number(ii,1) = ceil(f_num * (l_num/label_sum));
        end
        temp = IG_significance(data{vv},answer,number(ii,1) );
        [output_data{vv}{ii},output_index] = feature_choose_1(X{vv},data{vv},temp,f_index,number(ii,1));
        for jj = 1:length(output_index)
            index(sum+jj,1) = output_index(jj,1);
        end
        f_index = index;
        sum = sum + number(ii,1);
    end
end
end

