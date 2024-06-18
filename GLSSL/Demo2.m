clc;clear;
addpath('function');
addpath('funct');
load ('MVMLyeast.mat');
maxi.ganma = 10^-1; %[10^-4,10^4]
maxi.m = 2;

para.lambda  = 10^-5;
para.alpha = 10^5;
para.C = 10^-1;
para. choose = 1;
para.dratio = 0.9;
para.maxIter = 60;

optmParameter.lamda   = 2^-2;  % 2.^[-10:10] % label correlation 鏍囪鐩稿叧鎬?
optmParameter.beta   = 2^-6; % 2.^[-10:10] % sparsity 绋?枏鎬? -10 
optmParameter.gamma   = 0.01; % {0.1, 1, 10} % initialization for W   鍒濆鍖朩
optmParameter.minimumLossMargin = 10^-4;%鏈?綆鎹熷け闂撮殧锛屼负浜嗙▼搴忔彁鏃╃粨鏉?
optmParameter.maxIter = 100;%鏈?綆鎹熷け闂撮殧锛屼负浜嗙▼搴忔彁鏃╃粨鏉?
round = 5;

%% 鏍囩鍒嗙被
[fc_label,label_sum,sort] = matrix(target,maxi);

%% 鏁版嵁澶勭悊
[data, label]=trans(dataMVML,fc_label,maxi); 

%% 鐗瑰緛鍒嗙被 
[fc_data,d] = IG(dataMVML,data,label,label_sum,para,maxi);

%% 缁勭被灞炲睘鎬у瓙绌洪棿
tic
Z = subspace(dataMVML,d,para);
V = subspace_learning(fc_data,fc_label,label_sum,d,para);
data_V = [];
 for vv = 1:length(V)
     data_V = [data_V;V{vv}];
 end
for run = 1:round
    [train_data_V,test_data_V,train_data_Z,test_data_Z,train_label,test_label,train_target ,test_target] = generateCVSet1(data_V,Z,fc_label,target );
    [M,W] = predict_label2(train_data_V',train_data_Z',train_label,optmParameter);
%     end
%%   
         output = cell(length(W),1);
         Outputs = [];
         for vv = 1:length(W)
             output{vv} = test_data_V' * M{vv} + test_data_Z' * W{vv};
             Outputs = [Outputs,output{vv}];
         end
         Outputs = Outputs';
         Pre_Labels = sign(Outputs - (max(Outputs(:))-min(Outputs(:)))/2);
         Pre_Labels(Pre_Labels == -1)=0;
         test_target = test_target(:,sort);

%%     
    result{run,1}  = EvaluationAll(Pre_Labels,Outputs,test_target');
    time(1,run) = toc;

end
%% 鏍囩棰勬祴
[Avg_Result] = PrintGLSSLAvgResult(result,time,5);