function [mu,mask]=kmeans(ima,k)
%   ���ܣ�����k-means�㷨��ͼ����зָ�
%   ����: ima-����ĻҶ�ͼ��           k-������
%   ���:  mu-��ֵ������                mask-������ͼ��
ima=double(ima);
copy=ima;         
ima=ima(:);
mi=min(ima);      
ima=ima-mi+1;     
s=length(ima);
% ����ͼ��Ҷ�ֱ��ͼ
m=max(ima)+1;
h=zeros(1,m);
hc=zeros(1,m);
for i=1:s
  if(ima(i)>0) h(ima(i))=h(ima(i))+1;end;
end
ind=find(h);
hl=length(ind);
% ��ʼ������
mu=(1:k)*m/(k+1);
% start process
while(true)
  oldmu=mu;
  % ���еķ���  
  for i=1:hl
      c=abs(ind(i)-mu);
      cc=find(c==min(c));
      hc(ind(i))=cc(1);
  end
  %���¼����ֵ  
  for i=1:k, 
      a=find(hc==i);
      mu(i)=sum(a.*h(a))/sum(h(a));
  end
  if(mu==oldmu) break;end;
  end
% calculate mask
s=size(copy);
mask=zeros(s);
for i=1:s(1),
for j=1:s(2),
  c=abs(copy(i,j)-mu);
  a=find(c==min(c));  
  mask(i,j)=a(1);
end
end
mu=mu+mi-1;   
