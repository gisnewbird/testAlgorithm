
image = imread('C:\Users\Duke\Desktop\1.jpg');

R = image(:,:,1);

G = image(:,:,2);

B = image(:,:,3);

img_all = G+B+R;

img_merge = R*2+B/255+G/255;
[m,n]=size(R);
for i=1:m
    for j=1:n
        if (R(i,j)>80&&R(i,j)<190)&&(G(i,j)>50&&G(i,j)<190)&&(B(i,j)>0&&B(i,j)<190)
            image(i,j)=1;
        else
            image(i,j)=0;
        end
    end
end
imshow(image)


