function [output] = subspace_learning(X,Y,label_sum,d,para)
%SUBSPACE_LEARNING �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
sub_num = length(Y);
view_num = length(X);
data = cell(view_num,1);
output = cell(sub_num,1);
d_rat = zeros(sub_num,1);
sum = 0;
for vv = 1:sub_num
     if vv == sub_num
        d_rat(vv,1) = d - sum;
     else
        d_rat(vv,1) = ceil(d * (size(Y{vv},2)/label_sum));
        sum = sum + d_rat(vv,1);
     end
%     d_range = zeros(1,view_num);
     for ii = 1:view_num
         
        data{ii} = X{ii}{vv};
%         d_range(1,ii) = size(data{ii},2);
     end
%     d = floor(min(d_range)*dratio);
     output{vv} = subspace(data,d_rat(vv,1),para);
end
end
