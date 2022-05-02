%% ��̬ѧͼ����
%% 9.2 ���ͺ͸�ʴ
%% 9.2.1 ����
% �� 9.1 ����Ӧ��
%%
A = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0906(a)(broken-text).tif');
figure;
imshow(A, []);
B = [0 1 0; 1 1 1; 0 1 0];
A2 = imdilate(A, B); % �� B ���� A
figure;
imshow(A2, []); % ���ͺ� e a ���Ž�
%% 9.2.2 �ṹԪ�طֽ�
% �ṹԪ������ܱ��ֽ�Ϊ�����ӽṹ�����ͽ����[����Ҫд���ʼ���]
% 
% 
%% 9.2.3 strel ���� SE
% �� 9.2 strel �ֽ�ṹԪ��

se = strel('diamond', 5);
se.Neighborhood
decomp = getsequence(se); % ��ȡ�����ֽ��еĵ����ṹԪ��
whos % ���֮ǰ��������Ϣ
% decomp size Ϊ 4 * 1��˵���ֽ�Ϊ 4 ���ṹ��
decomp(1).Neighborhood
decomp(2).Neighborhood
decomp(3).Neighborhood
decomp(4).Neighborhood
%% 9.2.4 ��ʴ
% �� 9.3 ��ʴ��˵��

A = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0908(a)(wirebond-mask).tif');
SE = strel('disk', 10);
A2 = imerode(A, SE);
figure;
imshow(A2, []); % ������ �뾶Ϊ 10 Բ�̸�ʴ
SE = strel('disk', 5);
A2 = imerode(A, SE);
figure;
imshow(A2, []); % ������ �뾶Ϊ 5 Բ�̸�ʴ���ֵ���û����ʴ��
SE = strel('disk', 20);
A2 = imerode(A, SE);
figure;
imshow(A2, []); % ������ �뾶Ϊ 20 Բ�̸�ʴ ֻ�����ķ��鱣��
%% 9.3 �����븯ʴ���
%% 9.3.1 �����������
% �� 9.4 imopen imclose
%%
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0910(a)(shapes).tif');
figure;
imshow(f, []);
se = strel('square', 20);
se.Neighborhood
fo = imopen(f, se); % ������
figure;
imshow(fo); % ����һ�¼��ν��ͣ���ͼ���ڲ�ƽ�ƣ���������͹�����Ӵ�Ҳ���Ͽ�
fc = imclose(f, se);
figure;
imshow(fc, []); % ����һ�¼��ν��ͣ���ͼ����Χƽ�ƣ��ǲ��ǻ������ڵĿ�϶����������͹��
foc = imclose(fo, se); % ��������ٱ����㣬����ƽ��Ŀ��ͼ��
figure;
imshow(foc, []);
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0911(a)(noisy-fingerprint).tif');
figure;
imshow(f, []); % ���Ǻ����ָ��ͼ
se = strel('square', 3);
fo = imopen(f, se);
figure;
imshow(fo, []);% ���ÿ�����ȥ��һЩë�����������������һЩ�ڲ��Ķ���(��һ�뼸�ν��ͣ��ṹԪ��ͼ���ڲ���
foc = imclose(fo, se); % ���ñ��������ȱ��
figure;
imshow(foc, []);
%% 9.3.2 ���� - �����б任
% �� 9.5 bwhitmiss
% 
% �˴�����ʽ�ڵ����棬���İ��и���һ�ָ������ġ�������ڴ�ʵ�����Ӳ��˽⣬����ͼ 9.12

B1 = strel([0 0 0; 0 1 1; 0 1 0]);
B2 = strel([1 1 1; 1 0 0; 1 0 0]);
B1.Neighborhood % Ҫ�ұߣ��±��������أ�ע�� B1 ��Ӧ�� 1 ��λ��
B2.Neighborhood % ��Ҫ����|����|����|�ϣ�ע�� B2 ��Ӧ�� 1 λ��
% B1 ���Ļ��У�B2 ���Ĳ����У�B1 B2 ���½Ƕ�Ϊ 0��˵�����ǲ����ĵ㡣
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0913(a)(small-squares).tif');
figure;
imshow(f, []);
g = bwhitmiss(f, B1, B2);
figure;
imshow(g, []); % ���ԶԱ�һ�»��� �� �����зֱ�������Ӷ��ﵽ��СĿ��ͼ���Ч��
%% 9.3.3 ʹ�ò��ұ�
% ע����ǰͷ˵�����ӣ���θ�һ����״����һ��Ψһ������

% ���������Ҷ˵�
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0914(a)(bone-skel).tif');
figure, imshow(f, []);
f_e = endpoints(f);
figure, imshow(f_e, []);
%% 9.3.4 bwmorph �������͸�ʴ���ұ����ʵ�ֲ���
% 

f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0916(a)(bone).tif');
figure;
imshow(f, []);
g1 = bwmorph(f, 'thin', 1); % ϸ������ 1 ��
g2 = bwmorph(f, 'thin', 2); 
g_inf = bwmorph(f, 'thin', inf); % �����Σ�ֱ��������ϸ�� 
figure, imshow(g1, []);
figure, imshow(g2, []);
figure, imshow(g_inf, []); % inf ϸ�������һЩ��ʧ�����ܺȹ�������ͬ
fs = bwmorph(f, 'skel', inf); % ֱ�ӽ��й�����
figure, imshow(fs, []);
%% 9.4 ��ע���ӷ���
%%
% 4\8 �ڽӣ�4\8 ����(���һ��·�� ��� ͼ 9.18
% ���ӷ���(����˵һ������)�������ڽӷ�ʽ�Ĳ�ͬ���仯 ͼ 9.19
%% 
% �� 9.7 ������ʾ���ӷ���������

f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0917(a)(ten-objects).tif');
figure, imshow(f);
[L, n] = bwlabel(f); % ������ͨ�����ĸ�����Ĭ�� 8 ���ӷ���, L Ϊ��Ǿ���
% figure;
L % ���Ǻü����������� 1 ��ʼ���
n % ������
% ͨ�����������˵���������ͨ���� 
[r, c] = find(L == 6); % ���Ϊ 6 �ķ���
[M, N] = size(f);
t_f    = zeros(M, N);
t_f(r, c) = f(r, c);
figure, imshow(t_f, []);
% ���ű������
figure;
imshow(f);
hold on; %�ú�ͷ�� plot ��������ǰһ��ͼ
for k = 1 : n
    [r, c] = find(L == k);
    rbar = mean(r);
    cbar = mean(c);
    % ���� plot ����ʱ�� x��y�� ������ΪʲôҪ��������(����ϵ����
    plot(cbar, rbar, 'Marker', 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor','k', 'MarkerSize', 10);
    plot(cbar, rbar, 'Marker', '*', 'MarkerEdgeColor', 'w');
end
hold off;
%% 9.5 ��̬ѧ�ع�
% f Ϊ��ǣ�g Ϊģ��, ���ϵ������¹�ʽ
% 
% $\left.h_{k+1} =\left(h_k \bigoplus f\right)\cap g\right)$��ֱ�� $h_{k+1} 
% =h_k$
%%
% imreconstruct(marker, mask)
%% 9.5.1 ���ع���������

% �� 9.8 ���ع���������
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0922(a)(book-text).tif');
figure;
imshow(f, []);
fe = imerode(f, ones(51, 1)); % �����߸�ʴ
% fe_hori = imerode(f, ones(1, 21));
figure, imshow(fe, []);
fo = imopen(f, ones(51, 1)); % �����㣬��ͼ���ڲ���������������
figure, imshow(fo);
fobr = imreconstruct(fe, f); % ����ʴ���ı��ͼ��ģ��ͼ��Ϊԭͼ
figure, imshow(fobr, []) % ���г�������ĸ����ԭ
% figure, imshow(imreconstruct(fe_hori, f)); ��ѡ
%% 9.5.2 ���׶�
% ���� imfill ����ʵ�֣�ע��������׶����㷨

% imfill
% 
%% 9.5.3 ����߽����
% ʹ�� imclearborder(f, conn), conn Ϊ 4 �� 8 �ڽ�
% 
% ��ͼ��߽������Ķ���������


%% 9.6 �Ҷ�ͼ����̬ѧ
%% 9.6.1 ��ʴ������
% ��ʵ���ǿ����� imerode, �������� strel ����ƽ̹��ƽ̹�ṹԪ
% 
% ƽ̹Ԫ�ĻҶȸ�ʴ�Ǿֲ���С���ӣ����Ի�ƫ�����Ҷ����ͻ�ƫ��
%%
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0923(a)(aerial).tif');
figure;

subplot(2, 2, 1);
imshow(f);

% b = strel([1 1 1], [1 2 1]); % ��ƽ̹�ṹԪ���Ҷ�ֵ��Ϊ xy ƽ��߶�
% f_b = imerode(f, b);
% subplot(2, 2, 2);
% imshow(f_b);
se = strel('square', 3);
gd = imdilate(f, se); % ƽ̹�ṹԪ����ͼ��
subplot(2, 2, 2);
imshow(gd);

ge = imerode(f, se);
subplot(2, 2, 3);
imshow(gd);

% ��̬ѧ�ݶ� - ���ͽ���͸�ʴ����Ĳ�ֵ
morph_grad = imsubtract(gd, ge); % ͼ����ֵ
subplot(2, 2, 4);
imshow(morph_grad); % ���Ա�Ե���ָ����ԣ���Ϊ�ݶȾ�����͹�Ա�Ե�ı仯�̶�
%% 9.6.2 ��������
% �ڻҶȿ��������У�ͬǰͷ˵�ģ�xy ƽ���ϵ�һ���߶�ֵ��������ֵ��������һ������������
% 
% �����㣬�ṹԪ��������£�ȥ���ȽṹԪС������ϸ�ڣ������㣬�ṹԪ�������ϡ�
% 
% ��� ͼ 9.24
% 
% 
% 
% �� 9.9 ��������̬ѧƽ��

f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0925(a)(dowels).tif');
figure, imshow(f, []);
se = strel('disk', 5);
fo = imopen(f, se);
figure, imshow(fo, [])
foc = imclose(fo, se);
figure, imshow(foc, []) % �ӻҶ�����Ƕ��룬��֪��Ϊʲôͼ�е�Բ���������
% ����һ�ֽ����˲���ʽ���ṹԪ��������
fasf = f;
for k = 2 : 5
    se = strel('disk', k);
    fasf = imclose(imopen(fasf, se), se); % �ȿ����
end

figure;
imshow(fasf) % ������������ƽ��
%% 
% �� 9.10 ��ñ�任

% ��ñ�� ������ȥ������ϸ��
f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0926(a)(rice).tif');
figure, imshow(f);
se = strel('disk', 10);
fo = imopen(f, se); % �ȿ�����
figure, imshow(fo);
f2 = imsubtract(f, fo); % ��ñ����
figure, imshow(f2); % ���� ��������ȡ������
%% 
% ��Ȼ���� imtophat Ҳ����

f2 = imtophat(f, se);
figure, imshow(f2)
%% 
% �õ�ñ�Ͷ�ñ��ǿ�Աȶ�
% 
% ��ñ����������İ�Ŀ��

se = strel('disk', 3);
g = imsubtract(imadd(f, imtophat(f, se)), imbothat(f, se));
figure, imshow(g)
%% 
% �� 9.11 ��������
% 
% ����
%% 9.6.3 �ع�
% �������ع�����ʴ��ͼ�������ͼ��ԭͼ����ģ��

f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0925(a)(dowels).tif');
se = strel('disk', 5);
fe = imerode(f, se);
fobr = imreconstruct(fe, f); 
figure, imshow(fobr);
%% 
% �ع�������

fobrc = imcomplement(fobr); % �� fobr ��
fobrce = imerode(fobrc, se);
fobrcbr = imcomplement(imreconstruct(fobrce, fobrc)); % ���ع��������󲹡��ع��������ģ����ԭͼ��
figure, imshow(fobrcbr);
%% 
% �� 9.12 ʹ���ع�ɾ�� ����ͼ�񱳾�
% 
% ~~������ͷ���ؽ������Ѿ��� imreconstruct ʵ�֣����ǻ��ڴ˵�һЩ���ɣ��ڱ�����չ�ֳ���~~
% 
% �����ع������㣬ǰͷ�Ѿ��ṩ�ˡ�
% 
% �磺ʲô�������ͼ��ʲô����ģ��ͼ��

f = imread('./DIP-Ex/pic/dipum_images_ch09/Fig0930(a)(calculator).tif');
figure, imshow(f);
%% 
% �������Ȱ����ϵ��ֶ����ó����ṹԪ��������
% 
% ��Ѱ�������ʴ��������ᱻ�������ͻ����Ӷ�����

f_obr = imreconstruct(imerode(f, ones(1, 71)), f); % �� f ���д�СΪ 1 ���ؽ�������
f_o   = imopen(f, ones(1, 71));
figure;
subplot(1, 2, 1), imshow(f_obr);
subplot(1, 2, 2), imshow(f_o); % �������ع��ȿ�����Ч������һЩ
%% 
% ��ñ�ع�

f_thr = imsubtract(f, f_obr);
figure, imshow(f_thr);
%% 
% ���ԣ���ֱ�⻹�Ǵ��ڵġ�
% 
% ���ö�ˮƽ�߽ṹԪ�ٽ��п������ع�
% 
% �ҶԱ��˸�ʴ�Ϳ������ؽ�����Ϊ�Ұ����߸�����ˣ�ֻ�����˸�ʴˮƽ����⣬�����˸�ʴ���ƻ����ࡣ

figure;
g_obr = imreconstruct(imerode(f_thr, ones(1, 11)), f_thr); % �Ѹ�ʴ����ͼ����Ϊ��ǣ�f_thr ��ģ��
subplot(1, 2, 1);
imshow(imerode(f_thr, ones(1, 11)));
subplot(1, 2, 2);
imshow(g_obr);
%% 
% ���Է��� i �� % ����ȱʧ����Ϊ������ֱ�ṹԪ�ұ���С������ʴ�ˡ�����Ҫ�ָ����ǡ�
% 
% �ö�ˮƽ������ g_obr��ʹ�ñ���ʴ�ַ��������ַ������ͣ�ʹ�ñ���ʴ�����ص�(��������ڴ���������أ�Ϊ�ؽ������
% 
% ����ñ�ؽ����ͼ������ģ��(�������ع������ָܻ���ȥ)������С��ͼ���������ͼ�񣬽����ؽ�

g_obrd = imdilate(g_obr, ones(1, 21));
f2 = imreconstruct(min(g_obrd, f_thr), f_thr);
figure, imshow(f2)
%% 
% 
% 
% 
%% �����Զ�����
%%
function g = endpoints(f)
    persistent lut % ����־ñ���
    if isempty(lut)
        lut = makelut(@endpoint_fcn, 3); % ����һ���б������õ�һ�����ұ�ԭ��鿴ʵ���� 9.3.3��
    end
    g = applylut(f, lut);
end

% �鿴�������Ƿ�Ϊ�˵�
function is_end_point = endpoint_fcn(nhood)
    % nhood �� 3 * 3 �Ķ�ֵ�����ʾ�����������Ԫ���Ƕ˵㣬�ͷ��� 1
    
%     [0 0 0
%      0 1 0
%      0 0 1]
%      [1 0 0
%      0 1 0
%      0 0 0]
%   �������ϵĶ���˵�
    is_end_point = nhood(2, 2) & (sum(nhood(:)) == 2);
end