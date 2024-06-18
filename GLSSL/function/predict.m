function [W] = predict(X,Y,optmParameter)
%PREDICT 此处显示有关此函数的摘要
%   此处显示详细说明
view_num = length(Y);
epsilon = optmParameter.epsilon;
miniLossMargin = optmParameter.minimumLossMargin;
maxIter = optmParameter.maxIter2;
W = cell(view_num,1);
iter = 1;

oldloss = 0;
totalloss = 0;
   

model_W = [];
label = [];
%     train_label=Y{vv};
%     label=[label,train_label];
        
W = predict_label(X,Y,optmParameter);
%     model_W = [model_W,W{vv}];
%     loss = norm((X * model_W - label),'fro')^2;
%         if loss < epsilon
% 
%             break;
%         elseif  iter >= maxIter
% 
%             break;       
%         end
% 
% 
%        if abs(oldloss - totalloss) <= miniLossMargin
% 
%            break;
%        elseif totalloss <=0
%            break;
%        else
%            oldloss = totalloss;
%        end
%        if iter>maxIter
% 
%        end
%        iter=iter+1; 
%     end
        
end

