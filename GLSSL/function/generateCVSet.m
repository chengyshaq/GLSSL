function [train_data,test_data,train_label,test_label,train_target,test_target] = generateCVSet(data,Y,target)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
view_num = length(Y);
[Ndata,Nlabel]=size(target);
train_num=ceil(Ndata*0.8);
indexperm=randperm(Ndata);
train_index=indexperm(1,1:train_num);
test_index=indexperm(1,train_num+1:end);

train_target = target(train_index,:);
test_target = target(test_index,:);
train_data=data(:,train_index);
test_data= data(:,test_index);
train_label = cell(view_num,1);
test_label = cell(view_num,1);
for vv= 1:view_num

    train_label{vv}=Y{vv}(train_index,:);
    test_label{vv}=Y{vv}(test_index,:);
end
end

