
    function varargout=LengthSortStr(str)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Argment: str文件名组成的字符串数组
    % : 输出参数同sort函数
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ch_ascii=1; %用ascii码相应的字符放在文件名前，这个值越小越好

    [N,MAXLEN]=size(str); %个数以及每行的长度
    newname=[''];

    for i=1:N
    BlackSpace=0;
    while (str(i,MAXLEN-BlackSpace)==' ') %本行有多少个空格
    BlackSpace=BlackSpace+1;
    end
    %将本行后面的空格用ch_ascii补在最前面，并去掉行尾的空格
    newstr(i,:)=[repmat(char(ch_ascii),1,BlackSpace), str(i,1:MAXLEN-BlackSpace)];
    end
    [sortstr,order]=sortrows(newstr); %新字符排序，主要要得到排序的序号
    sortstr=str(order,:); %由这个序号生成原始的排序名

    if nargout<=1
    varargout{1}=sortstr;
    elseif nargout==2
    varargout{1}=sortstr;
    varargout{2}=order;
    end

