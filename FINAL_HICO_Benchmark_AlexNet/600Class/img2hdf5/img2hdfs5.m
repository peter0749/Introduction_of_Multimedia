clear all;
ImgSize=128; %As same as AlexNet
dirPrefix = 'images';
trainPrefix = 'train2015';
testPrefix = 'test2015';
LabelPrefix = 'Labels';
anno = load('anno.mat');

%anno.anno_test(anno.anno_test==0) = 0.159 + 0.682.*randn(1); 
%anno.anno_test(anno.anno_test==-1) = 0;
%anno.anno_test(isnan(anno.anno_test)) = 0;
anno.anno_test(isnan(anno.anno_test)) = -2;
anno.anno_test = anno.anno_test+2; %shift for caffe
%anno.anno_test(anno.anno_test~=1) = 0;

%anno.anno_train(anno.anno_train==0) = 0.159 + 0.682.*randn(1); 
%anno.anno_train(anno.anno_train==-1) = 0;
%anno.anno_train(isnan(anno.anno_train)) = 0;
anno.anno_train(isnan(anno.anno_train)) = -2;
anno.anno_train = anno.anno_train+2;
%anno.anno_train(anno.anno_train~=1) = 0;

single_size = 1024;

%Random Shuffle
%rshuffle = randperm(size(anno.anno_train,2));
%anno.list_train = anno.list_train(rshuffle,:);
%anno.anno_train = anno.anno_train(:,rshuffle);

trainList = [];
testList = [];
for i = 1:length(anno.list_train)
    temp = fullfile(dirPrefix,trainPrefix,anno.list_train(i));
    trainList = [trainList; temp];
end

for i = 1:length(anno.list_test)
    temp = fullfile(dirPrefix,testPrefix,anno.list_test(i));
    testList = [testList; temp];
end
%End of parsing filepath

trainNum = ceil(length(anno.list_train) / single_size);
testNum  = ceil(length(anno.list_test) / single_size);

fprintf('\nTow-pass method of creating testing/training set.');

if(~exist('TrainMean.mat'))
    fprintf('\nComputing image mean for training set... Total %d images', length(anno.list_train));
    TrainMean = double(zeros(ImgSize, ImgSize, 3));
    for i = 1:length(anno.list_train)
        fprintf('\nProcessing %d-th image', i);
        temp = imresize(imread(char(trainList(i))), [ ImgSize ImgSize ],'bilinear','AntiAliasing',true);
        if(size(temp,3)~=3)
            temp = cat(3, temp, temp, temp);%Grayscale to RGB
        end
        TrainMean = TrainMean + double(temp);
    end
    TrainMean = TrainMean./length(anno.list_train);
    TrainMean = single(TrainMean);
    save('TrainMean.mat','TrainMean');
else
    TrainMean = load('TrainMean.mat');
    TrainMean = TrainMean.TrainMean;
end

if(~exist('TestMean.mat'))
    fprintf('\nComputing image mean for testing set... Total %d images', length(anno.list_test));
    TestMean = double(zeros(ImgSize*2, ImgSize*2, 3));
    for i = 1:length(anno.list_test)
        fprintf('\nProcessing %d-th image', i);
        temp = imresize(imread(char(testList(i))), [ ImgSize*2 ImgSize*2 ],'bilinear','AntiAliasing',true);
        if(size(temp,3)~=3)
            temp = cat(3, temp, temp, temp);%Grayscale to RGB
        end
        TestMean = TestMean + double(temp);
    end
    TestMean = TestMean./length(anno.list_test);
    TestMean = single(TestMean);
    save('TestMean.mat','TestMean');
else
    TestMean = load('TestMean.mat');
    TestMean = TestMean.TestMean;
end

mkdir('train');
mkdir('test');

fprintf('\nstart to convert TrainImg.. total: %d segments\n ', trainNum);

parfor j = 1:trainNum
    %fprintf('Processing %d-th segment\n', j);
    filename = sprintf('train_%05d.hdf5', j);
    filename = fullfile('train',filename);
    in_range = ((j-1)*single_size)+1;
    if(j==trainNum)
        in_range = in_range:length(anno.list_train);
    else
        in_range = in_range:(in_range+single_size-1);
    end
    TrainLabel = anno.anno_train(:,in_range); % [number of labels, label ]
    TrainLabel = uint32(TrainLabel);%Convert to single
    TrainLabel = permute(TrainLabel, [3 4 1 2]);
    h5create(filename,'/data',[ImgSize*2 ImgSize*2 3 size(TrainLabel,4)], 'Datatype', 'single');
    k = 1;
    for i = in_range
        temp = imresize(imread(char(trainList(i))), [ ImgSize ImgSize ],'bilinear','AntiAliasing',true);
        if(size(temp,3)~=3)
            temp = cat(3, temp, temp, temp);%Grayscale to RGB
        end
        temp = single(temp);
        %temp = (temp - mean2(temp))./std2(temp);
        temp = temp-TrainMean;
        temp = cat(1,temp, flip(temp,1));
        temp = cat(2,temp, flip(temp,2));
        temp = permute(temp, [2 1 3 4]); %[col row channel num]
        h5write(filename, '/data', temp, [1 1 1 k], [ ImgSize*2 ImgSize*2 3 1 ]);
        k = k+1;
    end
end

fprintf('\nstart to convert TestImg.. total: %d segments\n ', testNum);
ImgSize = ImgSize*2;

parfor j = 1:testNum
    %fprintf('Processing %d-th segment\n', j);
    filename = sprintf('test_%05d.hdf5', j);
    filename = fullfile('test',filename);
    in_range = ((j-1)*single_size)+1;
    if(j==testNum)
        in_range = in_range:length(anno.list_test);
    else
        in_range = in_range:(in_range+single_size-1);
    end
    TestLabel = anno.anno_test(:,in_range); % [number of labels, label ]
    TestLabel = uint32(TestLabel);%Convert to single
    TestLabel = permute(TestLabel, [3 4 1 2]);
    h5create(filename,'/data',[ImgSize ImgSize 3 size(TestLabel,4)], 'Datatype', 'single');
    k = 1;
    for i = in_range
        temp = imresize(imread(char(testList(i))), [ ImgSize ImgSize ],'bilinear','AntiAliasing',true);
        if(size(temp,3)~=3)
            temp = cat(3, temp, temp, temp);%Grayscale to RGB
        end
        temp = single(temp);
        %temp = (temp - mean2(temp))./std2(temp);
        temp = temp-TestMean;
        temp = permute(temp, [2 1 3 4]); %[col row channel num]
        h5write(filename, '/data', temp, [1 1 1 k], [ ImgSize ImgSize 3 1 ]);
        k = k+1;
    end
    
end

mkdir(LabelPrefix);
mkdir(fullfile(LabelPrefix,'Train'));
mkdir(fullfile(LabelPrefix,'Test'));
num_class = length(anno.list_action);
for i = 1:num_class
    TrainLPath = fullfile(LabelPrefix,'Train', sprintf('Class-%d.hdf5',i));
    TestLPath = fullfile(LabelPrefix,'Test', sprintf('Class-%d.hdf5',i));
    h5create(TrainLPath, '/label', [1 1 1 length(anno.anno_train)], 'Datatype', 'uint8');%0, 1, 2
    h5create(TestLPath, '/label',  [1 1 1 length(anno.anno_test) ], 'Datatype', 'uint8');
    h5write(TrainLPath, '/label',  permute(uint8(anno.anno_train(i,:)), [3 4 1 2]));
    h5write(TestLPath, '/label',   permute(uint8(anno.anno_test(i,:)), [3 4 1 2]));
end

fprintf('\ndone\n');
