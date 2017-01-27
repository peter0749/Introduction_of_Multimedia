# 多媒體期末Project
2016/01/17

---

# Caffe 的安裝

### 方法一：使用 Caffe 提供的 Dockerfile 編譯 Docker image

1. 首先，你要有 Docker...
2. 從GitHub 下載 Caffe 的原始碼
```shell
git clone https://github.com/BVLC/caffe
```
3. 開始編譯 Docker image

```shell=
cd caffe/docker #切換至 caffe 底下的 docker 目錄
docker build -t caffe:cpu standalone/cpu #For CPU
#or
docker build -t caffe:gpu standalone/gpu #For Nvidia GPU
```

但是官方的 Dockerfile 的CUDA版本為 7.5，若要使用CUDA 8.0，或是出現

```shell
Unsupported gpu architecture 'compute_60' #caffe dockerfile 支援的cuda 版本太低
```

必須自行編譯原始碼（參考方法二）

4. 使用 Caffe 的 Docker image

請參考 Caffe 的 [GitHub](https://github.com/BVLC/caffe/tree/master/docker)

5. 完成，咖啡泡好了！

### 方法二：使用 Caffe 的原始碼編譯

1. 先照[官網](http://caffe.berkeleyvision.org/install_apt.html)上安裝 Ubuntu 上的Dependency

\([http://caffe.berkeleyvision.org/install\_apt.html](http://caffe.berkeleyvision.org/install_apt.html)\)

若要使用 CUDA 、 cuDNN ，請至NVIDIA 官網上下載。

2. 從GitHub 下載 Caffe 的原始碼

```shell
git clone https://github.com/BVLC/caffe
```

3. 開始編譯 Caffe

```shell=
cd caffe
cp Makefile.config.example Makefile.config #複製 makefile 設定
```

4. （可選，編輯Makefile.config）

若要使用 CUDA 和cuDNN

```shell
USE_CUDNN := 1 #刪除這行的註解
```

若要支援 CUDA 8.0

```shell
-gencode arch=compute_50,code=compute_50
```

這行改成

```shell
-gencode arch=compute_50,code=compute_50 \
-gencode arch=compute_60,code=compute_60
```

若只要使用 CPU

```shell
CPU_ONLY := 1 #刪除這行的註解
```

若要使用Matlab

```shell
MATLAB_DIR := /usr/local #設成你的Matlab 安裝路徑
```

5. 處理找不到hdf5 問題

若此時直接 `make all`
可能會出現以下錯誤

```shell
src/caffe/net.cpp:8:18: fatal error: hdf5.h: 沒有此一檔案或目錄
```

在 Makefile.config 中，修改以下行

```shell=
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/
hdf5/serial
```

這會告訴Caffe hdf5 的標頭檔和函式庫在哪裡。
>注意：hdf5 不一定在範例中的路徑，要確認 hdf5 標頭檔和函式庫路徑，可以使用 find 指令找到，例如： "find /usr | grep hdf5"。然後去猜 hdf5 的路徑在哪裡。

6. 編譯

```shell=
make all -j2 #2 jobs 加速
```

7. 測試
```shell=
make test
make runtest
```
8. 完成，咖啡泡好了！

# 如何安裝 matcaffe

## 如何設定 matcaffe
1. 回到 Caffe 的建制目錄，執行 make matcaffe
2. 將matlab 的 $LD_PRELOAD 指定為 libstdc++.so.6
3. 在Matlab 下，addpath(matcaffe 的路徑)

# 如何將影像轉成 HDF5 檔
1. 將NaN 的 Label 設成 -2
2. 將影像轉成相同大小，並全部轉成RGB
3. 計算 Image mean 並正規化
4. 在訓練圖片上，進行水平、垂直翻轉
5. 將圖片、 Label 資料分批寫入 HDF5 檔

HDF5 用於 Caffe 的 Shape:

```
data                     Dataset {Number, Channel, Row, Col}
label                    Dataset {Number, Class, 1, 1}
```

在 Matlab 上的方向是相反的（由右向左）

Matlab: 
```
data                     Dataset {Col, Row, Channel, Number}
label                    Dataset {1, 1, Class, Number}
```
>需要注意的事：
>將全部圖片在記憶體內轉檔後輸出，不是好的作法，建議使用 Append 的方式寫入。
>另外，caffe 32-bit 對HDFS5 的限制，單檔不能超過2GB，必須切割。
```matlab=
clear all;
ImgSize=128; %As same as AlexNet
dirPrefix = 'images';
trainPrefix = 'train2015';
testPrefix = 'test2015';
anno = load('anno.mat');

anno.anno_test(isnan(anno.anno_test)) = -2;
anno.anno_test = single(anno.anno_test+2); %shift for caffe
anno.anno_test = anno.anno_test./3.0; %scale for sigmoid
anno.anno_test(anno.anno_test>1) = 1;
anno.anno_test(anno.anno_test<0) = 0;

anno.anno_train(isnan(anno.anno_train)) = -2;
anno.anno_train = single(anno.anno_train+2);
anno.anno_train = anno.anno_train./3.0;
anno.anno_train(anno.anno_train>1) = 1;
anno.anno_train(anno.anno_train<0) = 0;

single_size = 1024; %Number of image in every HDF5 file

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
    TrainLabel = single(TrainLabel);%Convert to single
    TrainLabel = permute(TrainLabel, [3 4 1 2]);
    h5create(filename,'/label',size(TrainLabel), 'Datatype', 'single');
    h5write(filename,'/label',TrainLabel);
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
        temp = cat(1,temp, flip(temp,1));%mirror
        temp = cat(2,temp, flip(temp,2));%mirror
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
    TestLabel = single(TestLabel);%Convert to single
    TestLabel = permute(TestLabel, [3 4 1 2]);
    h5create(filename,'/label',size(TestLabel), 'Datatype', 'single');
    h5write(filename,'/label',TestLabel);
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

fprintf('\ndone\n');
```

# 使用 Caffe 與 AlexNet
## 前置作業
修改 Caffe 中 AlexNet 的 Model 如下：

```
name: "AlexNet"
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  hdf5_data_param {
    source: "train_loc.txt" #裏面寫 HDF5 存放的位置
    batch_size: 16
  }
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  hdf5_data_param {
    source: "test_loc.txt"
    batch_size: 8
  }
}
//（中間略）
```
並拿掉
```
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
```

因為 `Accuracy Layer` 不支援 Multi-Label
另外，因為 SoftmaxWighLoss 在 `Multi-Label` 時有問題，我們改用 `EuclideanLoss`
```
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "fc8"
  bottom: "label"
  top: "loss"
}
```
有一個說法，若將AlexNet 的後面兩個Pooling 層使用Average Pooling，
可以降低Noise Data 造成的影響（主要是背景部份）。因此我們修改 AlexNet 後面兩個 Pooling 層成 AVE。

使用以下的 solver.prototxt
```
net: "train_val.prototxt" #Model 的位置
test_iter: 2383 # 測試資料/batch_size
test_interval: 1000 #每幾筆Test 一次
base_lr: 0.001 #Base learning rate
lr_policy: "step"
gamma: 0.1
stepsize: 100000
display: 20 #每幾筆顯示一次
max_iter: 450000 #最大迭代次數
momentum: 0.9
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "models/caffe_alexnet_train" #Snapshot 存放的位置
solver_mode: GPU #使用GPU 訓練
```

## 開始訓練
```
caffe train -solver="solver.prototxt" -gpu=0
```
會將 Caffe 的 weight 和 snapshot 存在 solver 指定的位置

## 抽取 DNN Features

### 使用 matcaffe
```matlab=
clear all;
addpath('~/caffe/matlab');
anno = load('img2hdf5/anno.mat');
caffe.set_mode_gpu();
caffe.set_device(0);
mkdir('features');
weight = 'models/caffe_alexnet_train_iter_95000.caffemodel';
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
```

會抽取 fc6 後的 fc7 全連接層的 Features，共 38116 個訓練Features，和 9658 個測試Features
使用的 Network 附在作業檔案中(extracter.prototxt)

# 使用 HICO Benchmark
將抽取完成的 DNN Features 交給HICO Benchmark 內附的 SVM 訓練
將 DNN Features 放在以下位置
```
hico_benchmark/data/precomputed_dnn_features/imagenet
```
並在 Matlab 上執行

```matlab
train_svm_vo
```
執行完成後，可以計算mAP
```matlab
eval_ko_run
```

計算得mAP `33.62`
似乎結果不盡理想，可能 Model 有誤，或訓練方法有誤，需要再進一步檢查，但時間有限，無法繼續追溯。

![](https://i.imgur.com/SEpMjaq.jpg)

# 如何執行程式
1. 下載 [HICO 資料](http://napoli18.eecs.umich.edu/public_html/data/hico_20150920.tar.gz)
2. 將 HICO 的images/ 和 anno.mat 放到img2hdf5 下
3. 執行作業中的img2hdf5/img2hdf5.m 轉成 HDF5 檔
4. 執行主目錄下的gen_hdf5_list.sh 取得 HDF5 檔路徑
5. caffe train -solver="solver.prototxt" 取得 Caffe Model
6. 執行 gen_res.m 取得 DNN Features
7. 將 features 內的資料放在 HICO Benchmark 主目錄下的 data/precomputed_dnn_features/imagenet
8. 設定好 HICO Benchmark ，執行 train_svm_ko
9. 執行 eval_ko_run 評估mAP 