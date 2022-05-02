A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
A
%%
B = A(:, 2);
B  % 取第二列，行
%%
B = A(1:2, 1:2); % 提取前两行中的前两列
B
%%
A, B = A([1, 3], [2, 3]) % 1,2;1,3;3,2;3,3
%%
f = imread('./workpace/pic/dipum_images_ch02/Fig0206(a)(rose-original).tif');
figure;
subplot(2, 2, 1);
imshow(f(end:-1:1, :)); %% 通过矩阵的索引进行垂直翻转
subplot(2, 2, 2);
imshow(f(257:768, 127:256)); %% 通过矩阵的索引进行取部分
subplot(2, 2, 3);
imshow(f(1:2:end, 1:2:end)); %% 通过矩阵的索引进行二次取样，隔一个取一次
subplot(2, 2, 4);
imshow(f(256, :)); %% 通过矩阵的索引水平扫描
%%
M = 20;
N = 10;

r = 0:M-1;
c = 0:N-1;

[C, R] = meshgrid(c, r);
C,R