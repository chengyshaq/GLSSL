function [label,p,sort] = matrix(Y,maxi)
%FUNCTIONS 此处显示有关此函数的摘要
%   此处显示详细说明
ganma = maxi.ganma;
m = maxi.m;
[q,p] = size(Y);
W = zeros(p,p);
d = zeros(1,p);
D = zeros(p,p);
L = zeros(p,p);
A = zeros(p,p);
VB = zeros(p,m);
B = zeros(p,m);
label = cell(m,1);
sort = zeros(1,p);
% tic
for i = 1:p
    for j = 1:p
        sum =0;
        if i == j
            W(i,j)=0;
        else
            for k = 1:q
                sum = sum + (Y(k,i)-Y(k,j))^2;
            end
            W(i,j) = exp(-ganma * sum);
            
        end
        d(1,i) = d(1,i) + W(i,j);
    end
    D(i,i) = diag(d(1,i));
end
L = D - W;
A = D^(-1/2) * L * D^(-1/2);
[V,E] = eig(A);
%% 特征值最小的两个值对应的特征向量
for i = 1:p
    for j = i:p
        if E(i,i) > E(j,j)
            t1 = E(i,i);
            E(i,i) = E(j,j);
            E(j,j) = t1;
            
            t2 = V(:,i);
            V(:,i) = V(:,j);
            V(:,j) = t2;
        end
    end
end
%% 计算矩阵B
for i =1:m
    VB(:,i) = V(:,i);
end
for i = 1:p
    Vij = 0;
    for j = 1:m
        Vij = Vij + (VB(i,j))^2;
    end
    B(i,:) = VB(i,:)./(Vij)^(1/2);
end
output = kmeans(B,m);
%% 标签分类
t=1;
for i = 1:m
    k=1;
    for j = 1:p
        if output(j,1) == i
            label{i}(:,k) = Y(:,j);
            k = k+1;
            sort(t) = j;
            t = t+1;
        end
    end   
end
 
end

