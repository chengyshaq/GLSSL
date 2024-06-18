function [temp]= IG_significance( data, answer,number)

ret = zeros( number, 1);
fcol = size(data, 2 ); 
lcol = size(answer, 2 ); 

% f_ent = zeros( fcol, 1 );
% fl_ent = zeros( fcol, lcol ); 
temp = zeros( fcol, 1 );%存放特征数量
tempx = zeros( fcol, lcol );
for i=1:fcol%计算每个特征与所有的类标签互信息
    for j=1:lcol
%         P=(p_entropy(data(:,i))+p_entropy(answer(:,j))-p_entropy( [data(:,i) answer(:,j)]));
        tempx(i, j) = tempx(i, j) + (p_entropy(data(:,i))+p_entropy(answer(:,j))-p_entropy( [data(:,i) answer(:,j)]));
    end
    temp(i, 1) = temp(i, 1) + max(tempx(i,:)); 
end
[tt ret( 1, 1 )] = min( temp, [], 1 );%%ret存放最大的值，tt是最大值的位置
relevance = temp;
if number > 1
    redundancy = zeros( fcol, 1 );%计算特征之间的冗余性
    for i=2:number
        for j=1:fcol 
            if j == ret(i-1, 1)
            else
                redundancy( j, 1 ) = redundancy( j, 1 ) + p_entropy(data(:,i))+p_entropy(data(:,ret(i-1, 1)))-p_entropy([data(:,i) data(:,ret(i-1, 1))]);
            end
        end
%         temp = relevance - redundancy;%特征与类标签的相关性-特征与特征的冗余性
           temp = relevance - ((redundancy)/(i-1));
        temp( ret( 1:(i-1), 1 ), 1 ) = intmin;%将选中的特征最小化，防止被再次选中
        [tt ret( i, 1 )] = max( temp, [], 1 );%最大相关，最小冗余mrmr
    end
end
end
