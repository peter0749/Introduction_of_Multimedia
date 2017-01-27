clear all;
addpath('~/caffe/matlab/');
weights = 'models/1/caffe_alexnet_train_iter_20000.caffemodel';
model = 'deploy.prototxt';
caffe.set_mode_cpu();
net = caffe.Net(model, weights, 'test');
data = h5read('img2hdf5/test/test_00002.hdf5','/data');
net.forward({data(:,:,:,285)});
res = net.blobs('prob').get_data();