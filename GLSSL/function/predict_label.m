function [output_W,output_M] = predict_label(X,Y,optmParameter)
%PREDICT_LABEL 此处显示有关此函数的摘要
%   此处显示详细说明
% alpha            = optmParameter.alpha;
beta             = optmParameter.beta;
gamma            = optmParameter.gamma;
lamda = optmParameter.lamda;
% theta = optmParameter.theta;
miniLossMargin   = optmParameter.minimumLossMargin;
maxIter = optmParameter.maxIter;

view_num = length(Y);
XTY = cell(view_num,1);
W_s = cell(view_num,1);
W_s_1 = cell(view_num,1);
W_s_k = cell(view_num,1);
Gw_s_k = cell(view_num,1);
R = cell(view_num,1);
Lip = zeros(view_num,1);
t_num = zeros(view_num,1);
num_dim = size(X,1);
S = cell(view_num,view_num);
XTX = X*X';
label = [];
t_sum = 0;
for vv = 1:view_num 
%     if vv == view_num
%        t_num(vv,1) = 1 - t_sum;
%     else
%        t_num(vv,1) = size(Y{vv},2)./size(label,2);
%        t_sum = t_sum + t_num(vv,1);
%     end
    for ii = 1:view_num
        S{vv,ii} = pdist2( Y{ii}'+eps, Y{vv}'+eps, 'cosine' );
    end
    XTY{vv} = X*Y{vv};
    W_s{vv}  = (XTX + gamma*eye(num_dim)) \ (XTY{vv});
    W_s_1{vv} = W_s{vv};
    label = [label,Y{vv}];
    R{vv,1} = pdist2( Y{vv}'+eps, Y{vv}'+eps, 'cosine' );
    Lip(vv,1) = sqrt((norm(XTX)^2 + norm(lamda*R{vv})^2));
%     theta(vv,1) = ceil(size(Y{vvv},2)/laebl_sum);
    
end
XTL = X*label;
RL = pdist2( label'+eps, label'+eps, 'cosine' );
LipL = sqrt((norm(XTX)^2 + norm(lamda*RL)^2));
W_l  = (XTX + gamma*eye(num_dim)) \ (XTL);
W_l_1 = W_l;
%见论文中的推导，线性函数求初始回归系数。   
%esp一个很小的数，因为Y为分母，不能等于零，
iter    = 1;
oldloss = 0;
predictionLoss = 0;
correlation = 0;
sparsity = 0;
    
%公式15；norm(A)表示A的2范数

bk = 1;
bk_1 = 1; %初始化b0，b1,见论文伪代码算法1
    
%% proximal gradient 近端梯度
while iter <= maxIter  %当迭代次数小于最大迭代次数
    
    W_l_k = W_l + (bk_1 - 1)/bk * (W_l - W_l_1);
    Gw_l_k = W_l_k - 1/LipL * ((XTX*W_l_k - XTL) + lamda* W_l_k*RL);
    bk_1   = bk;
    bk     = (1 + sqrt(4*bk^2 + 1))/2;
    W_l_1 = W_l;
    W_l = softthres(Gw_l_k,beta/LipL);
    for vv = 1:view_num     
       W_s_k{vv}  = W_s{vv} + (bk_1 - 1)/bk * (W_s{vv} - W_s_1{vv});%论文算法1伪代码
%        S_sum = zeros(size(W_s_k{vv}));
%        for ii = 1:view_num
%            if vv == ii
%            else
%                 S_sum = S_sum + W_s_k{ii} * 
%            end
%        end
       Gw_s_k{vv} = W_s_k{vv} - 1/Lip(vv,1) * ((XTX*W_s_k{vv} - XTY{vv}) + lamda * W_s_k{vv}*R{vv} +  );   
       W_s_1{vv}  = W_s{vv};
       W_s{vv}    = softthres(Gw_s_k{vv},beta/Lip(vv,1)); 
       
       predictionLoss = predictionLoss + 0.5*trace((X'*W_s{vv} - Y{vv})'*(X'*W_s{vv} - Y{vv}));%公式3第一部分
       correlation     = correlation + 0.5*lamda*trace(R{vv,1}*W_s{vv}'*W_s{vv});%标记相关性计算，公式（3）第二部分
       sparsity    = sparsity + beta*sum(sum(W_s{vv}~=0));%提取类属属性，公式3第三部分
    end
       
%    predictionLoss = predictionLoss +0.5*trace((X'*W_l-label)'*(X'*W_l - label));
%    correlation = correlation + 0.5 *lamda*trace(RL*W_l'*W_l);
%    sparsity = sparsity + beta *sum(sum(W_l~=0));
   totalloss = predictionLoss + correlation + sparsity ;%总的模型损失函数，即优化目标函数，即公式3
      
   if abs(oldloss - totalloss) <= miniLossMargin %若每次迭代老的损失减去新产生的损失小于最小损失误差的绝对值，结束
       break;
   elseif totalloss <=0 %若上述条件不成立，执行该条件操作，新产生的损失小于等于0，也停止
       break;
   else      %若上述所有的条件都不成立，执行下面语句
       oldloss = totalloss;
   end
       
   iter=iter+1;
end
output_W = W_s;%输出模型的系数矩阵
output_M = W_l;

end


%% soft thresholding operator 软阈值运算
function W = softthres(W_t,lambda)  %公式11
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end


