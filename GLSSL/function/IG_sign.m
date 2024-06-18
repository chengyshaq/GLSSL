function [output,newrel,order,d] = IG_sign(data,label,label_sum,para)
%IG_SIGN 此处显示有关此函数的摘要
%   此处显示详细说明
%% 按标签数量排序
dratio = para.dratio;
label = compare(label);
xlen = length(data);
ylen = length(label);
newrel = cell(xlen,1);
order = cell(xlen,1);
output = cell(xlen,1);
d_range = zeros(1,xlen);

for ii = 1:xlen
    d_range(1,ii) = size(data{ii},2);
end
d = floor(min(d_range)*dratio);

for s = 1:xlen
    newrel{s} =cell(ylen,1);
    order{s} = cell(ylen,1);
    feature = data{s};
    f_index = [];
    for t = 1:ylen     
        answer = label{t};
        fcol = size(feature,2 ); 
        lcol = size(answer,2 ); 

        f_ent = zeros( fcol, 1 );
        fl_ent = zeros( fcol, lcol ); 

        for k=1:fcol
            f_ent( k, 1 ) = p_entropy(feature(:,k)); 
            for m=1:lcol  
                fl_ent(k,m) = p_entropy( [feature(:,k) answer(:,m)] );
            end
        end
        l_ent = zeros( 1, lcol );
        for m=1:lcol
            l_ent( 1, m ) = p_entropy( answer(:,m) );
        end
%% 特征与类标签的互信息的计算：I(S;L)=H(S)-H(S;L)+H(L); f_ent(k)- fl_ent(k,m)+ l_ent(m)
        rel = zeros( fcol, 1 );
        for k=1:fcol
            for m=1:lcol
                rel(k) = rel(k) + f_ent(k) + l_ent(m) ...
                    - fl_ent(k,m);
            end
        end
        [newrel{s},order{s}]=sort(rel,'descend');
        [output{s}{t},index] = feature_choose(feature,order{s},f_index,lcol,label_sum,fcol);
        f_index = index;
    end
end
end

