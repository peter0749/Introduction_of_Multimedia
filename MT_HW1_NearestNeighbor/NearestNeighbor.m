clear;
colormap gray;
def_img_scale = [192 168];
def_img_size = 192*168;
DirPrefix = 'CroppedYale\';
Dirs = ls(strcat(DirPrefix,'yale*'));
Dirs = strcat(DirPrefix,Dirs);
pages = [0];
images = [];
fprintf('Loading...');
for i = 1:size(Dirs,1)
   ls_images = strcat( Dirs(i,:) , '\', ls(strcat(Dirs(i,:) , '\*.pgm')));
   temp_images = [];
   pages = [pages size(ls_images,1)];
   for j = 1:size(ls_images,1)
       single_image = reshape(imresize(imread(ls_images(j,:)), def_img_scale), 1, def_img_size);
      temp_images = [temp_images ; single_image];
   end
   %temp_images is the i-th col. in images
   images = [ images ; temp_images ];
end
images = single(images);
id_offset = cumsum(pages);
NNebor = [];%For SSD
NNebor0 = [];%For SAD
fprintf('Loaded images successfully! Begin to compute SSD, SAD...');
for i = 1:size(images,1)
%for i = 1:300
   %ess = images(i,:);
   %ess = repmat(ess, size(images,1), 1);
   %difftab = (images - ess).^2;
   difftab = bsxfun(@minus, images, images(i,:));
   difftab0 = abs(difftab);
   difftab = difftab.^2;
   diffsum0 = sum(difftab0,2);
   diffsum = sum(difftab,2);
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

SAD_C = 0; SDD_C = 0;
for i = 1:size(id_offset,2)-1
    SAD_C = SAD_C + sum(NNebor0(id_offset(i)+1:id_offset(i+1)) <= id_offset(i+1));
    SDD_C = SDD_C + sum(NNebor(id_offset(i)+1:id_offset(i+1)) <= id_offset(i+1));
end
SAD_P = SAD_C / size(images,1)
SDD_P = SDD_C / size(images,1)

%image(temp_images);