function [train,target] = trans(data,target,maxi)
%TRANS �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
mm = maxi.m;
xlen = length(data);
ylen = length(target);
train = cell(xlen,1);

for iii = 1:xlen
    [m,n]=size(data{iii});
    train{iii}=[];
    for i=1:n
        res=dis_efi( data{iii}(:,i), mm );
        train{iii}=[train{iii} res];
    end
end 
for jjj = 1:ylen
%     target{jjj} = target{jjj}';
    index=find(target{jjj}==0);
    target{jjj}(index)=-1;
%     target{jjj} = target{jjj}';
end

