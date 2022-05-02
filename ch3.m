%% 
%% ���ö����任��������ӳ�䵽�߻Ҷ�
%%
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0305(a)(spectrum).tif');
imshow(f);
%%
figure
subplot(1, 2, 1), imshow(f);
subplot(1, 2, 2), imhist(f);
axis tight
%%
g = im2uint8(mat2gray(1 * log(1 + double(f))));
% g = im2uint8(mat2gray(f));
figure, subplot(1, 2, 1), imshow(g), subplot(1, 2, 2), imhist(g), axis tight
title('ʹ�ö����任��С��̬��Χ')
%% ���ɱ任
%%

%%
figure
imshow(f)
%%
%m_e = 2
m_e = 0.28 % < 1 ���ݴ�������仯����Ч��
m_c = 1
after_m = mat2gray(m_c * double(f) .^ m_e);
figure, subplot(1, 2, 1), imshow(after_m), subplot(1, 2, 2), imhist(after_m), axis tight
%% �Աȶ�����
%%

%%
m = 5;
e = 10;
multi = im2uint8(mat2gray(1./(1 + (m./double(f) + eps).^e))); % ���� eps ��ֹ 0 ֵ
figure
imshow(multi);
%% ��һ��ֱ��ͼ
%%
h = imhist(f, 2) / numel(f) % b = 2, �ֳ������Ҷȣ���������� 8bit ͼ��0~127 128 ~ 255 �ֳ�����
%% ����ֱ��ͼ
%%
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0308(a)(pollen).tif');
h = imhist(f);
h1 = h(1:10:256); % �������
horz = 1:10:256; % ˮƽ������
bar(horz, h1); % bar(horz, v, width) v �Ǳ����Ƶĵ�
% ���òο���Ϳ̶�Ҫ�ڻ�ͼ����֮��
axis([0 255 0 15000]);
set(gca, 'xtick', 0:50:255);
set(gca, 'ytick', 0:2000:15000); % gca ��õ�ǰ�ᣬ2000 Ϊ���
stem(horz, h1, 'r:d', 'filled'); % 'r:d' ��ֱ��ͼ���ߺ͵����ʽ
set(gca, 'xtick', 0:50:255);
set(gca, 'ytick', 0:2000:15000); % gca ��õ�ǰ�ᣬ2000 Ϊ���
plot(h)
set(gca, 'xtick', 0:50:255);
set(gca, 'ytick', 0:2000:15000); % gca ��õ�ǰ�ᣬ2000 Ϊ���
%% ֱ��ͼ���⻯
%%
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0308(a)(pollen).tif');
imshow(f);
figure, imhist(f); % �����ԻҶȾ�����һ���ϰ��ķ�Χ
ylim('auto')
g = histeq(f, 256); % histeq ����ֱ��ͼ���⣬ԭ������ɢ��������ķֲ��ɷ�ʽ, 256 �����㼶, ��ͼͳһ�㼶�Ļ���ֱ��ִ�о��⻯����
figure, imshow(g);
figure, imhist(g);
ylim('auto'); % ����ֱ��ͼ��Χ���ˣ��Աȶ�����
%% ֱ��ͼƥ��
%%
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0310(a)(Moon Phobos).tif');
figure
subplot(1, 2, 1), imshow(f), subplot(1, 2, 2), imhist(f);
figure;
h = histeq(f, 256);
subplot(1, 2, 1), imshow(h), subplot(1, 2, 2), imhist(h); 
% ֱ��ͼ���⻯���������ˣ����Ǽ����ڽ���һ�������ɫ��ԭͼ������ 0 �����������ۼƺ����ر���(��һ����ɢ�� �ֲ�����
% ���Խ�������ָ����һ��ֱ��ͼ��Ȼ����ֱ��ͼƥ�䷽��������ԭͼ��ֱ��ͼ
% ע�Ᵽ��ԭͼ���ԣ�����ʹ��˫���˹����
m1 = 0.15;
sig1 = 0.05;
m2 = 0.75;
sig2 = 0.05;
a1 = 1;
a2 = 0.07;
k = 0.002;
c1 = a1 * (1 / ((2 * pi) ^ 0.5) * sig1);
k1 = 2 * (sig1 ^ 2);
c2 = a2 * (1 / ((2 * pi) ^ 0.5) * sig2);
k2 = 2 * (sig2 ^ 2);
z = linspace(0, 1, 256);
p = k + c1 * exp(-((z - m1) .^ 2) ./ k1) + c2 * exp(-((z - m2) .^ 2) ./ k2);
p = p ./ sum(p(:));
figure;
plot(p);
xlim([0 255]);
% y = kx + b ����һ�»�ͼ
x = linspace(0, 256);
y = 2 * x + 2;
figure, plot(y), xlim([0 10]);

xlim([0.858 1.973])
ylim([1.18 6.75])
g = histeq(f, p); % ֱ��ͼƥ��
figure;
subplot(1, 2, 1), imshow(g), subplot(1, 2, 2), imhist(g) % �͸���˫��Ƚϣ������ƣ����һҶȱ���չ����Ϊ��ķ�Χ�ˣ��Աȶ���ǿ��
%% �ռ��˲�
%%
% ��Ҫ�����ĵ� + ������Ĵ���
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0315(a)(original_test_pattern).tif');
imshow(f);
%f = im2uint8(mat2gray(double(f)));
w = ones(31); % ����һ��ƽ���˲��������˲�
%w = w ./ 31^2 % ԭ��������д��
w, w ./ 32^2
% ���������� uint8 ����Ǻϣ�ԭ���� imfilter ������ͼ������ͬ�࣬����ͼ���� uint8�������ȻҲ��
% ����˲��������� 0 255 ��Χ�����Զ��ضϡ�������Ҫ��һ��ϵ������Ȼ��������н����ʾ
gd = imfilter(f, w), gd_ = imfilter(f, w ./ 31^2);
figure;
subplot(1, 2, 1), imshow(gd), subplot(1, 2, 2), imshow(gd_, [ ]); % �����Ա�Ե��� 0 �ᵼ�±�Եģ��
gr = imfilter(f, w ./ 31^2, 'replicate'); % ʹ���� replicate �����Եģ������
figure, imshow(gr, [ ]);
%% �������˲�
%%
% ��Ҫʹ�� colfilter���� colflter ��Ҫ�Ѿ�������ͼ�񣬹����˽� padarray
f = [1, 2; 3, 4];
fp1 = padarray(f, [3, 2], 'replicate', 'post');
fp2 = padarray(f, [3, 2], 'replicate', 'both');
fp1
fp2 % 1 2 3 4 �����ģ������������ҵ�һ�������һ����ʼ��䣬�����Ǹ��ơ�
size(fp2, 1)
size(fp2, 2)
prod(fp2, 1) % ���ŵ�һά�ȳ˻�
prod(fp2, 2)
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0310(a)(Moon Phobos).tif');
f1 = mat2gray(f);
f2 = f;
%f = [1, 2; 3, 4];
f1 = padarray(f1, [3 2], 'replicate'); 
f2 = padarray(f2, [3 2], 'replicate'); 
g1 = colfilt(f1, [3 2], 'sliding', @gmean);
g2 = colfilt(f2, [3 2], 'sliding', @gmean);
figure;
subplot(1, 2, 1), imshow(g1), subplot(1, 2, 2), imshow(g2);% ע���������ڣ�����û�н�����ֵӳ�䵽 0~1
%% ���� matlab ����ʵ��ģ����˲���
%% 
%%
% ���˲��� ������˹
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0316(a)(moon).tif')
w = (fspecial('laplacian', 0)); % ������˹���� ģ��
double(w)
g1 = imfilter(f, w, 'replicate'); % �Ҷ�ֵ�ᱻ�ü�
figure, imshow(g1, [])
f2 = mat2gray(f); % double ��������
f2
g2 = imfilter(f2, w, 'replicate'); % �˲�
figure, imshow(g2, []);
g = f2 - g2; % ������˹��ʽ f(x, y) + c * delta() ��û��Ӱ���ͼ�������ȥ��ע�������Ǽ�����Ϊģ�������� -4 ����
figure, imshow(g) % imshow Ĭ��ͼ���� [0 1] ֮��ĻҶȣ���������ͽض�Ϊ��ɫ(255)
% �ֹ�ָ���˲�������ǿ�����ıȽ�
f_d = mat2gray(f); % ע���
w4 = fspecial('laplacian', 0);
w8 = [1 1 1; 1 -8 1; 1 1 1];
gw4 = imfilter(f_d, w4, 'replicate');
gw8 = imfilter(f_d, w8, 'replicate');
figure;
subplot(1, 3, 1), imshow(f_d), subplot(1, 3, 2), imshow(f_d - gw4), subplot(1, 3, 3), imshow(f_d - gw8); 
% �����Կռ��˲� ͳ������
% ordfilt2(f, order, domain), order �����еڼ�������������� domain 0 1 ��ɾ���1 ��ʾ���λ������Ҫ������㣬ָ������
% ���� medfilt2 ��ֵ�˲�
f = imread('./DIP-Ex/pic/dipum_images_ch03/Fig0318(b)(ckt-board-slt-pep-both-0pt2).tif'); % ��������
figure,imshow(f);
% �����˲���
g1 = ordfilt2(f, median(1:3*3), ones(3, 3)); % �Զ�����ֵ�˲� ����ֵ������������
g2 = medfilt2(f); % ��ά��ֵ�˲�
figure;
subplot(1, 3, 1), imshow(f), subplot(1, 3, 2), imshow(g1), subplot(1, 3, 3), imshow(g2);
%% 
% 
%%
function v = gmean(A)
    mn = size(A, 1); % colfilt ��� A ��չ�������������� m * n https://blog.csdn.net/PanPan_1995/article/details/115390609
    v = prod(A, 1) .^ (1/mn);% �Ե�һά�������õ�һ���������������ǳ˻�
end