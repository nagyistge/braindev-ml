% for i=1:10
%     im{i}=imread(['generated_data/segmented_2D_best_' num2str(i) '.jpg']);
% end
% 
% ims=zeros(10,size(im{1},1),size(im{1},2));
% %fit a gaussian to each pixel
% for i=1:10
%     ims(i,:,:)=im{i};
% end
% 
% m=reshape(mean(ims),size(ims,2),size(ims,3));
% v=reshape(var(ims),size(ims,2),size(ims,3));
% 
% subplot(2,2,1)
% imagesc(m)
% title('mean of each pixel')
% subplot(2,2,3)
% imagesc(v)
% title('variance of each pixel')
% 
% %the difference between the statistics of pixels 1,1 and 1,2 is:
% sqrt(([m(1,1) v(1,1)]-[m(1,2) v(1,2)])*([m(1,1) v(1,1)]-[m(1,2) v(1,2)])')
% 
% for i=1:size(m,1)
%     for j=1:size(m,2)-1
%         h(i,j)=sqrt(([m(i,j) v(i,j)]-[m(i,j+1) v(i,j+1)])*([m(i,j) v(i,j)]-[m(i,j+1) v(i,j+1)])');
%     end
% end
% 
% subplot(2,2,2)
% imagesc(h)
% title('horizontal similarity between pixels')
% %unfortunately, here we have the underlying assumption of equal variance
% %across segments. A segment with high variance will look like a boundary.
% 
% %make a random initialization of a same/different map for adjacent pixels
% h_sim=rand(size(m,1),size(m,2)-1)>0.5;
% v_sim=rand(size(m,1)-1,size(m,2))>0.5;
% %0 is same segment, 1 is different segment
% 
% %the horizontal transition matrix is:
% H_trans=[0.9 0.1;0.5 0.5];
% V_trans=[0.9 0.1;0.5 0.5];
% 
% %for each segment in _sim, calculate the mean and variance using m and v

%% binary cat segmentation %%
close
im_cats=1+(rgb2gray(imread('cats.jpg'))<100);
im_cats=im_cats(1:2:end,1:2:end);

p_cat=0.5;
%0-not cat, 1-cat
%random initialization
%labels=1+(rand(size(im_cats))<p_cat);
%initialization with known labels
labels=im_cats;

hist(labels(:),2)

subplot(1,2,1)
imagesc(im_cats);
subplot(1,2,2)
imagesc(labels)

%calaulate P(x|l)*P(l)
Pxl=[0.3 0.7;0.7 0.3]
Pl=[1-p_cat;p_cat]

Pxl(labels(1,1),im_cats(1,1))*Pl(labels(1,1))

%calculate P(X|L)*P(L)
PXL=Pxl(sub2ind(size(Pxl),reshape(im_cats,numel(im_cats),1),reshape(labels,numel(labels),1)));
PL=Pl(labels(:));
PXLPL=PXL;%.*PL;

%calculate a clique potential
Pll=[0.05 0.95];%top-different, bottom-same
Pll(1+(labels(6,7)~=labels(6,8)))

MAP=-bitmax;

for i=1:100000
    %repeatedly, make a perturbation, calculate the thing again, and keep the
    %new one if it's more likely
    %1)just one changed pixel
    flipme=randsample(numel(im_cats),1);
    labels(flipme)=3-labels(flipme);
    
    %calculate all clique potentials
    %horizontal
    HU=Pll(2-(labels(:,1:end-1)~=labels(:,2:end)));
    %vertical
    VU=Pll(2-(labels(1:end-1,:)~=labels(2:end,:)));
    %total
    U=sum(HU(:))+sum(VU(:));
    
    PXL=Pxl(sub2ind(size(Pxl),reshape(im_cats,numel(im_cats),1),reshape(labels,numel(labels),1)));
    PL=Pl(labels(:));
    PXLPL=PXL.*PL;
    logPX=sum(log(PXLPL));
    
    if(MAP-(logPX+U)>0)
        %flip back
        labels(flipme)=3-labels(flipme);
        %labels=old_labels;
    else
        MAP=logPX+U;
    end
    logPX_time(i)=MAP;
    if(mod(i,100)==0)
        disp(num2str(i));
        subplot(1,3,1)
        imagesc(im_cats);
        subplot(1,3,2)
        imagesc(reshape(PXLPL,size(im_cats)));
        subplot(1,3,3)
        imagesc(labels);        
        %imagesc(reshape(PXLPL,size(im_cats)))
        title([num2str(MAP)])
        pause(0.05)
    end
end

subplot(1,2,1)
imagesc(im_cats);
subplot(1,2,2)
imagesc(labels)