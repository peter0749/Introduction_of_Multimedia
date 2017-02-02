clear all;
addpath('~/caffe/matlab');
anno = load('img2hdf5/anno.mat');
caffe.set_mode_gpu();
caffe.set_device(0);
mkdir('features');
weight = 'models/caffe_alexnet_train_iter_170040.caffemodel';
model = 'extracter.prototxt';
ori = caffe.Net(model, weight, 'test');
H5List = table2array(readtable('train_loc.txt','Delimiter','/','ReadVariableNames',false));
k = 1;
for i=1:size(H5List,1)
    data = h5read(char(fullfile(H5List(i,1),H5List(i,2),H5List(i,3))), '/data');
    for j=1:size(data,4)
        net = ori;
        net.forward({data(:,:,:,j)});
        feat = single(net.blobs('fc7').get_data()');
        save(fullfile('features',strrep(anno.list_train{k},'.jpg','.mat')),'feat');
        k = k+1;
    end
    fprintf('%d / %d\n', i, size(H5List,1));
end


H5List = table2array(readtable('test_loc.txt','Delimiter','/','ReadVariableNames',false));
k = 1;
for i=1:size(H5List,1)
    data = h5read(char(fullfile(H5List(i,1),H5List(i,2),H5List(i,3))), '/data');
    for j=1:size(data,4)
        net = ori;
        net.forward({data(:,:,:,j)});
        feat = single(net.blobs('fc7').get_data()');
        save(fullfile('features',strrep(anno.list_test{k},'.jpg','.mat')),'feat');
        k = k+1;
    end
    fprintf('%d / %d\n', i, size(H5List,1));
end
