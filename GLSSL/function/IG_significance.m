function [temp]= IG_significance( data, answer,number)

ret = zeros( number, 1);
fcol = size(data, 2 ); 
lcol = size(answer, 2 ); 

% f_ent = zeros( fcol, 1 );
% fl_ent = zeros( fcol, lcol ); 
temp = zeros( fcol, 1 );%�����������
tempx = zeros( fcol, lcol );
for i=1:fcol%����ÿ�����������е����ǩ����Ϣ
    for j=1:lcol
%         P=(p_entropy(data(:,i))+p_entropy(answer(:,j))-p_entropy( [data(:,i) answer(:,j)]));
        tempx(i, j) = tempx(i, j) + (p_entropy(data(:,i))+p_entropy(answer(:,j))-p_entropy( [data(:,i) answer(:,j)]));
    end
    temp(i, 1) = temp(i, 1) + max(tempx(i,:)); 
end
[tt ret( 1, 1 )] = min( temp, [], 1 );%%ret�������ֵ��tt�����ֵ��λ��
relevance = temp;
if number > 1
    redundancy = zeros( fcol, 1 );%��������֮���������
    for i=2:number
        for j=1:fcol 
            if j == ret(i-1, 1)
            else
                redundancy( j, 1 ) = redundancy( j, 1 ) + p_entropy(data(:,i))+p_entropy(data(:,ret(i-1, 1)))-p_entropy([data(:,i) data(:,ret(i-1, 1))]);
            end
        end
%         temp = relevance - redundancy;%���������ǩ�������-������������������
           temp = relevance - ((redundancy)/(i-1));
        temp( ret( 1:(i-1), 1 ), 1 ) = intmin;%��ѡ�е�������С������ֹ���ٴ�ѡ��
        [tt ret( i, 1 )] = max( temp, [], 1 );%�����أ���С����mrmr
    end
end
end
