tic
%load images
imsize=40;
images=zeros(10,imsize,imsize);
for i=1:10
	im=imread (['generated_data/segmented_2D_40x40_' num2str(i) '.jpg']) ;
	images(i,:,:)=double(im);
end

%find means of distributions at each pixel
means=reshape(mean(images),imsize,imsize);

%we should use the Two-sample Kolmogorov–Smirnov test
%to find if adjacent distributions are the same

%instead, we use the difference of means
v_diff=abs(means(2:end,:)-means(1:end-1,:));
v_diff=v_diff./max(v_diff(:));

h_diff=abs(means(:,2:end)-means(:,1:end-1));
h_diff=h_diff./max(h_diff(:));

%save to csv files
csvwrite('test_h_diff.csv',h_diff);
csvwrite('test_v_diff.csv',v_diff);

toc

whos

%{
whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  ===== 
        ans         5x5                        200  double
        h_diff    255x254                   518160  double
        i           1x1                          8  double
        im        255x255                    65025  uint8
        images     10x255x255              5202000  double
        means     255x255                   520200  double
        v_diff    254x255                   518160  double

Total is 909866 elements using 6823753 bytes
%}
