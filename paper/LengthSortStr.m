
    function varargout=LengthSortStr(str)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Argment: str�ļ�����ɵ��ַ�������
    % : �������ͬsort����
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ch_ascii=1; %��ascii����Ӧ���ַ������ļ���ǰ�����ֵԽСԽ��

    [N,MAXLEN]=size(str); %�����Լ�ÿ�еĳ���
    newname=[''];

    for i=1:N
    BlackSpace=0;
    while (str(i,MAXLEN-BlackSpace)==' ') %�����ж��ٸ��ո�
    BlackSpace=BlackSpace+1;
    end
    %�����к���Ŀո���ch_ascii������ǰ�棬��ȥ����β�Ŀո�
    newstr(i,:)=[repmat(char(ch_ascii),1,BlackSpace), str(i,1:MAXLEN-BlackSpace)];
    end
    [sortstr,order]=sortrows(newstr); %���ַ�������ҪҪ�õ���������
    sortstr=str(order,:); %������������ԭʼ��������

    if nargout<=1
    varargout{1}=sortstr;
    elseif nargout==2
    varargout{1}=sortstr;
    varargout{2}=order;
    end

