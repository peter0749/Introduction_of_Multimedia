% 以下為開啟一個 RGB 的 Bitmap 圖，利用其 RGB Chenal 的 Pixel Value 轉為 YIQ Chenal 的範例
% 利用 Matlab 來寫影像的程式很容易，分為下列的步驟
% 1.開啟一個彩色影像檔（.bmp）（24 bit）
% 2.讀取此影像的 Pixel value
% 3.影像處理  
% 4.存取結果並顯示


clear

% Set Transfer Matrix
RGB2YIQ = [ 0.299 0.587 0.114 ; 0.596 -0.274 -0.322 ; 0.212 -0.523 0.311];

% Read File
X = imread ('Koala.bmp');
[H W B] = size(X);

if (B~=3)
   fprintf('Not RGB FIle!');
   return;
end

% Image Preocessing (RGB -> YIQ)
% 以這裡的例子是要把 RGB -> YIQ 最後取 Y （Gray Level）值顯示出來
fprintf('wait ......\n');
Y = double(reshape(X, W*H, B));
Y = Y * RGB2YIQ';
%Y = bsxfun(@times,RGB2YIQ,Y);
Y = uint8(round(reshape(Y, H, W, B)));
%for i = 1:H
%   for j = 1:W
%      OldPixel(1:B) = X(i,j,1:B);
%      NewPixel = RGB2YIQ*double(OldPixel');
%      Y(i,j,1:B) = uint8(round(NewPixel'));
%   end
%end
% 不過在 Matlab 中 for 迴圈的處理速度超級慢

% Eliminate I&Q Chanel 只取 Gray Level
Z = repmat(Y(:,:,1),1,1,B);

% 存檔
imwrite(Z,'lena_gray.bmp');

% 顯示結果
subplot(1,2,1);
image(X);
title ('Original Image');
subplot(1,2,2);
image(Z);
title ('Generated Image')
