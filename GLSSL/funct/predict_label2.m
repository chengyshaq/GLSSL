function [output_M,output_W] = predict_label2(V,Z,Y,optmParameter)
%PREDICT_LABEL2 此处显示有关此函数的摘要
%   此处显示详细说明
maxIter = optmParameter.maxIter;
gamma = optmParameter.gamma;
beta = optmParameter.beta;
lamda = optmParameter.lamda;
view_num = length(Y);
W_s = cell(view_num,1);
M_s = cell(view_num,1);
W_s_1 = cell(view_num,1);
M_s_1 = cell(view_num,1);
W_s_k = cell(view_num,1);
M_s_k = cell(view_num,1);
Gw_s_k = cell(view_num,1);
Gm_s_k =cell(view_num,1);
iter = 1;
oldloss = 0;
miniLossMargin = 10^-3;
fea_dim = size(Z,2);
ZTZ = Z'*Z;
VTV = V'*V;
ZTV = Z'*V;
VTZ = V'*Z;
for vv = 1:view_num
    lab_dim = size(Y{vv},2); 
    W_s{vv} = rand(fea_dim,lab_dim);  
    M_s{vv} = (2 * VTV + gamma *eye(fea_dim))\(V'*Y{vv} + VTZ*W_s{vv});
    W_s{vv} = (2 * ZTZ + gamma* eye(fea_dim))\(Z'*Y{vv} + ZTV*M_s{vv});
    W_s_1{vv} = W_s{vv};
    M_s_1{vv} = M_s{vv};    
end
W_Lip = sqrt(2*(norm(ZTZ)^2) + 2*(norm(lamda * ZTZ)^2));
M_Lip = sqrt(2*(norm(VTV)^2) + 2*(norm(lamda * VTV)^2));
bk = 1;
bk_1 = 1;
while iter <= maxIter
    Wloss = 0;
    Mloss = 0;
    sparsity = 0;
    for ii =1 : view_num
        W_s_k{ii} = W_s{ii} + (bk_1-1)/bk * (W_s{ii}-W_s_1{ii});
        M_s_k{ii} = M_s{ii} + (bk_1 -1 )/bk * (M_s{ii} - M_s_1{ii});
        Gw_s_k{ii} = W_s_k{ii} - 1/W_Lip*((ZTZ*W_s_k{ii} - Z'*Y{ii})+lamda*(ZTZ*W_s_k{ii}-ZTV*M_s_k{ii}));
        Gm_s_k{ii} = M_s_k{ii} - 1/M_Lip*((VTV*M_s_k{ii} - V'*Y{ii})+lamda*(VTV*M_s_k{ii}-VTZ*W_s_k{ii}));
        
        W_s_1{ii} = W_s{ii};
        W_s{ii} = softthres(Gw_s_k{ii},beta/W_Lip);
        M_s_1{ii} = M_s{ii};
        M_s{ii} = softthres(Gm_s_k{ii},beta/M_Lip);
        Wloss = Wloss + norm((Z*W_s{ii}-Y{ii}),'fro')^2;
        Mloss = Mloss + norm((V*M_s{ii}-Y{ii}),'fro')^2;
        loss = norm((Z*W_s{ii} - V * M_s{ii}),'fro')^2;
        sparsity = sparsity + sum(sum(W_s{ii}~=0)) + sum(sum(M_s{ii}~=0));
    end
    bk_1 = bk;
    bk = (1+sqrt(4*bk^2+1))/2;
    totalloss = 0.5*Wloss + 0.5*Mloss + 0.5 *lamda*loss + 0.5*beta*sparsity;
    
     if abs(oldloss - totalloss) <= miniLossMargin %若每次迭代老的损失减去新产生的损失小于最小损失误差的绝对值，结束
           break;
       elseif totalloss <=0 %若上述条件不成立，执行该条件操作，新产生的损失小于等于0，也停止
           break;
       else      %若上述所有的条件都不成立，执行下面语句
           oldloss = totalloss;
       end
       
       iter=iter+1;
end
output_M = M_s;
output_W = W_s;
end

%% soft thresholding operator 软阈值运算
function W = softthres(W_t,lambda)  %公式11
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end




