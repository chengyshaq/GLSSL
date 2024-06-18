function [model_LLSF] = predict_label_1(X,Y,optmParameter)
%PREDICT_LABEL_1 此处显示有关此函数的摘要
%   此处显示详细说明
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

   %% initializtion 初始化参数
    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);%见论文中的推导，线性函数求初始回归系数。   
    W_s_1 = W_s;
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );%esp一个很小的数，因为Y为分母，不能等于零，

    iter    = 1;
    oldloss = 0;
    
    Lip = sqrt(2*(norm(XTX)^2 + norm(alpha*R)^2));%公式15；norm(A)表示A的2范数

    bk = 1;
    bk_1 = 1; %初始化b0，b1,见论文伪代码算法1
    
   %% proximal gradient 近端梯度
    while iter <= maxIter  %当迭代次数小于最大迭代次数

       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);%论文算法1伪代码
       Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - XTY) + alpha * W_s_k*R);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip); 
       
       predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));%公式3第一部分
       correlation     = trace(R*W_s'*W_s);%标记相关性计算，公式（3）第二部分
       sparsity    = sum(sum(W_s~=0));%提取类属属性，公式3第三部分
       totalloss = predictionLoss + alpha*correlation + beta*sparsity;%总的模型损失函数，即优化目标函数，即公式3
      
       if abs(oldloss - totalloss) <= miniLossMargin %若每次迭代老的损失减去新产生的损失小于最小损失误差的绝对值，结束
           break;
       elseif totalloss <=0 %若上述条件不成立，执行该条件操作，新产生的损失小于等于0，也停止
           break;
       else      %若上述所有的条件都不成立，执行下面语句
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    model_LLSF = W_s;%输出模型的系数矩阵

end


%% soft thresholding operator 软阈值运算
function W = softthres(W_t,lambda)  %公式11
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end

