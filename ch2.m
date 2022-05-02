A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
A
%%
B = A(:, 2);
B  % ȡ�ڶ��У���
%%
B = A(1:2, 1:2); % ��ȡǰ�����е�ǰ����
B
%%
A, B = A([1, 3], [2, 3]) % 1,2;1,3;3,2;3,3
%%
f = imread('./workpace/pic/dipum_images_ch02/Fig0206(a)(rose-original).tif');
figure;
subplot(2, 2, 1);
imshow(f(end:-1:1, :)); %% ͨ��������������д�ֱ��ת
subplot(2, 2, 2);
imshow(f(257:768, 127:256)); %% ͨ���������������ȡ����
subplot(2, 2, 3);
imshow(f(1:2:end, 1:2:end)); %% ͨ��������������ж���ȡ������һ��ȡһ��
subplot(2, 2, 4);
imshow(f(256, :)); %% ͨ�����������ˮƽɨ��
%%
M = 20;
N = 10;

r = 0:M-1;
c = 0:N-1;

[C, R] = meshgrid(c, r);
C,R