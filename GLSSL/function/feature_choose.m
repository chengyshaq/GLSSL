function [output_index,output_data] = feature_choose(data,temp,number)
%FEATURE_CHOOSE �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[order,index] = sort(temp,'descend');

output_index = index(1:number);
output_data = data(:,output_index);
end

