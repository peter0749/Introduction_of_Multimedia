clear;
fprec = fopen('precision.txt','w');%Output precision
def_img_scale = [192 168];%Dimension of images
def_img_size = 192*168;%Number of rows
DirPrefix = 'CroppedYale';%Prefix of the folder
tdir = dir(fullfile(DirPrefix,'yale*'));
Dirs = [];
for i = 1:size(tdir,1)
	Dirs = [Dirs ; fullfile(DirPrefix,tdir(i).name)];
end
pages = [0];%Number of pictures of the i-th people in i-th folder
images = [];%Total images, sort by the order of Folders and images in each folder.
fprintf('Loading...');
for i = 1:size(Dirs,1)
    
	tdir = dir(fullfile(Dirs(i,:),'*.pgm'));
	ls_images = [];
	for j = 1:size(tdir,1)-1%Discard black images
		ls_images = [ls_images ; fullfile(Dirs(i,:),tdir(j).name)];
	end
   temp_images = [];
   pages = [pages size(ls_images,1)];
   for j = 1:size(ls_images,1)
       single_image = reshape(imresize(imread(ls_images(j,:)), def_img_scale), def_img_size, 1);
      temp_images = [temp_images  single_image];
   end
   %Each picture is the i-th col. in images
   images = [ images  temp_images ];
end
images = single(images);%Change to single for speed.
id_offset = cumsum(pages);%Compute the index for each folder in "images" matrix.
NNebor = [];%For SSD
NNebor0 = [];%For SAD
fprintf('Loaded images successfully! Begin to compute SSD, SAD...');
for i = 1:size(images,2)
%for i = 1:300
   %ess = images(i,:);
   %ess = repmat(ess, size(images,1), 1);
   %difftab = (images - ess).^2;
   difftab = bsxfun(@minus, images, images(:,i));
   difftab0 = abs(difftab);
   difftab = difftab.^2;
   diffsum0 = sum(difftab0,1);
   diffsum = sum(difftab,1);
   diffsum0(i) = inf;
   diffsum(i) = inf;
   [diffmin0, minCOL0] = min(diffsum0);
   [diffmin, minCOL] = min(diffsum);
   
   %mindiff = [-1 inf];%index, val
   %for j = [1:size(images,1)]
   %   if(j~=i)
   %       temp = sum((images(j,:) - ess).^2);
   %       if(temp < mindiff(2))
   %           mindiff = [j temp];
   %       end
   %   end
   %end
   NNebor = [NNebor minCOL];
   NNebor0 = [NNebor0 minCOL0];
end

SAD_C = 0; SSD_C = 0;
%Note: The content in sum() is a index of vector which elements are logical
%numbers, 0 and 1.
%Summarize them to get the total number of correct NN for each pictures.
for i = 1:size(id_offset,2)-1
    SAD_C = SAD_C + sum(NNebor0(id_offset(i)+1:id_offset(i+1)) <= id_offset(i+1));
    SSD_C = SSD_C + sum(NNebor(id_offset(i)+1:id_offset(i+1)) <= id_offset(i+1));
end
SAD_P = SAD_C / size(images,2)
SSD_P = SSD_C / size(images,2)
fprintf(fprec, 'SAD: %f\r\nSDD: %f\r\n',SAD_P, SSD_P);
fclose(fprec);
csvwrite('NNTable.csv',[NNebor ; NNebor0]);%write the result to table
%Plot two images for demostration...
i=floor(rand()*size(images,2));
j=NNebor(i);
colormap gray;
k1 = reshape(uint8(round(images(:,i))),192,168);
k2 = reshape(uint8(round(images(:,j))),192,168);
subplot(1,2,1);
image(k1(:,:));axis image;
subplot(1,2,2);
image(k2(:,:));axis image;
imwrite([k1 k2], 'sample.pgm');%Save the demo images.

%image(temp_images);
