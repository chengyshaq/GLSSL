function [model_LLSF] = predict_label_1(X,Y,optmParameter)
%PREDICT_LABEL_1 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    alpha            = optmParameter.alpha;
    beta             = optmParameter.beta;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

   %% initializtion ��ʼ������
    num_dim = size(X,2);
    XTX = X'*X;
    XTY = X'*Y;
    W_s   = (XTX + gamma*eye(num_dim)) \ (XTY);%�������е��Ƶ������Ժ������ʼ�ع�ϵ����   
    W_s_1 = W_s;
    R     = pdist2( Y'+eps, Y'+eps, 'cosine' );%espһ����С��������ΪYΪ��ĸ�����ܵ����㣬

    iter    = 1;
    oldloss = 0;
    
    Lip = sqrt(2*(norm(XTX)^2 + norm(alpha*R)^2));%��ʽ15��norm(A)��ʾA��2����

    bk = 1;
    bk_1 = 1; %��ʼ��b0��b1,������α�����㷨1
    
   %% proximal gradient �����ݶ�
    while iter <= maxIter  %����������С������������

       W_s_k  = W_s + (bk_1 - 1)/bk * (W_s - W_s_1);%�����㷨1α����
       Gw_s_k = W_s_k - 1/Lip * ((XTX*W_s_k - XTY) + alpha * W_s_k*R);
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       W_s_1  = W_s;
       W_s    = softthres(Gw_s_k,beta/Lip); 
       
       predictionLoss = trace((X*W_s - Y)'*(X*W_s - Y));%��ʽ3��һ����
       correlation     = trace(R*W_s'*W_s);%�������Լ��㣬��ʽ��3���ڶ�����
       sparsity    = sum(sum(W_s~=0));%��ȡ�������ԣ���ʽ3��������
       totalloss = predictionLoss + alpha*correlation + beta*sparsity;%�ܵ�ģ����ʧ���������Ż�Ŀ�꺯��������ʽ3
      
       if abs(oldloss - totalloss) <= miniLossMargin %��ÿ�ε����ϵ���ʧ��ȥ�²�������ʧС����С��ʧ���ľ���ֵ������
           break;
       elseif totalloss <=0 %������������������ִ�и������������²�������ʧС�ڵ���0��Ҳֹͣ
           break;
       else      %���������е���������������ִ���������
           oldloss = totalloss;
       end
       
       iter=iter+1;
    end
    model_LLSF = W_s;%���ģ�͵�ϵ������

end


%% soft thresholding operator ����ֵ����
function W = softthres(W_t,lambda)  %��ʽ11
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end

