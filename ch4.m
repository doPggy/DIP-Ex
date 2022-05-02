%% ������ Ƶ����
%% ���� DFT
%%
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0403(a)(image).tif');
imshow(f);
% Ҫ���и���Ҷ�任�ˣ����ǿ��ٸ���Ҷ
F = fft2(f);
S = abs(F); % ����Ƶ�ף�Ҳ����ʵ���鲿ƽ����
imshow(S, []); % ����ע�⵽Ƶ������������
% ����� Fc = fftshift(f); % �任ԭ���ƶ������ģ����� (-1)^(x+y), ��������Ǳ任���Ƶ��
% ���Կ������������
Fc = fftshift(F);
imshow(abs(Fc), []); % [] ����ʲô��˼ imshow(abs(Fc)) �ͻ������
% S2 = log(1 + mat2gray(Fc)); ���Ǵ���ģ���������ǿƵ��ͼ�����Ǹ���Ҷ�任Ƶ��
S2 = log(1 + abs(Fc));
figure;
imshow(S2, []);
% ����ifftshift �Ѿ��еߵ����������ڰ�����ת�ؾ������Ͻ�
% ifft2 ����Ҷ��任���������ԭʼ�ĸ���Ҷ�任
f1 = ifft2(F);
figure;
subplot(1, 3, 1), imshow(f1), subplot(1, 3, 2), imshow(ifft2(Fc)), subplot(1, 3, 3), imshow(real(ifft2(F)));
%% Ƶ���˲�
%%
% �ռ����� - Ƶ�������
% ��4.1 ��������˲�Ч��
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0405(a)(square_original).tif');
figure;
imshow(f); % 256 * 256
[M, N] = size(f);
F = fft2(f); %����Ҷ
% ��˹��ͨ�˲���
sig = 10;
H = lpfilter('gaussian', M, N, sig);
G = H .* F; % Ƶ�����˲�
fi = ifft2(G);
% ���Է��֣�δ���Ĵ�ֱ����û��ģ������ˮƽ����ģ����
% �˲���(�ռ��ʾ)����һ���������ڵ�ͼ���Ͼ������Ȼ��ģ��(����������Ϊ DFT���������������ǻ����ÿռ������)����ֱ���ֲ�̫���
% ���忴ͼ 4.32
imshow(real(fi), []); 
PQ = paddedsize(size(f)); % 512 * 512
Fp = fft2(f, PQ(1), PQ(2)); % �������
sig = 10;
Hp = lpfilter('gaussian', PQ(1), PQ(2), 2 * sig); % Ƶ������ʹ����䣬�˲�����ͼ���СҪ��ͬ PQ(1) * PQ(2)
Gp = Fp .* Hp;
gp = real(ifft2(Gp)); % ��任
gpc = gp(1:size(f, 1), 1:size(f, 2)); % �ü�����Ȼ�������Ҷ�2ͼ(���ƿռ��˲����
figure;
h = fspecial('gaussian', 15, 7);
gs = imfilter(f, h); % �ڿռ��˲�
subplot(1, 3, 1), imshow(gpc, []), subplot(1, 3, 2), imshow(gp, []), subplot(1, 3, 3), imshow(gs, []); % ˮƽ�ʹ�ֱ����ģ���߿�
%% DFT �˲���������
%%

%% �ռ��˲������Ƶ���˲���
%%
% �����������Կռ��˲������и���Ҷ H = fft2(h, PQ(1), PQ(2))
% �� 4.2
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0409(a)(bld).tif');
figure;
imshow(f);
F = fft2(f); % ���������, ����Ǻ�ɫ����ʱ���Ҫ�����ǲ���Ҫ��ǿƵ��ͼ�񣬶���Ҫ��Ҫ���У�
S = fftshift(log(1 + abs(F)));
figure;
imshow((S), []);
h = fspecial('sobel'); % Ĭ�Ϻ����sobel ���ӣ����Ժ�ͷ��ͼ�лᷢ�ֺ��Ե����ǿ
h
figure;
freqz2(h);
PQ = paddedsize(size(f));
H = freqz2(h, PQ(1), PQ(2)); % freqz2 ��������һ��Ƶ���˲�
H1 = ifftshift(H); % �������Ļ����ĸ���Ҷ�任����������Ļ�(ԭ��λ�����Ͻ�
figure;
% �������Ļ���
%subplot(1, 2, 1), freqz2(abs(H)), subplot(1, 2, 2), freqz2(abs(H1));
subplot(2, 2, 1), imshow(abs(H), []), subplot(2, 2, 2), imshow(abs(H1), []); 
% �ռ����˲���sobel ����
gs = imfilter(double(f), h);
% Ƶ�����˲���sobel ���Ӷ�Ӧ��Ƶ�ʺ���
gf = dftfilt(f, H1);
figure;
% ����һ��
subplot(1, 2, 1), imshow(gs, []), subplot(1, 2, 2), imshow(gf, []);
% ������ֵͼ�񣬿���Ե
figure;
imshow(abs(gs) > 0.2 * abs(max(gs(:))));
figure;
imshow(abs(gf) > 0.2 * abs(max(gf(:))));
%% ��Ƶ�����������˲���
%%
% ���� dftuv �����ã����Զ������룬ע�� meshgrid ���÷�
[U, V] = dftuv(8, 5);
U
V
D = U.^2 + V.^2;
D % ����������Ͻǵľ�������ˣ�������λ�þ��Ǿ�������
fftshift(D) % �����ԣ�������ת�Ƶ�����������
%% ��ͨƵ���˲���
%%
% �� 4.4
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0413(a)(original_test_pattern).tif');
figure;
imshow(f);
PQ = paddedsize(size(f));
[U, V] = dftuv(PQ(1), PQ(2));
D0 = 0.05 * PQ(2); % 
F = fft2(f, PQ(1), PQ(2)); % �任
H = exp(-(U .^ 2 + V .^ 2) ./ (2 * D0 ^ 2)); % ��˹��ͨ�˲���
g = dftfilt(f, H);
figure;
% H �����Ļ���, F �Ǹ���Ҷ�仯���ͼ��
subplot(2, 2, 1), imshow(g, []), subplot(2, 2, 2), imshow((H), []);
subplot(2, 2, 3), imshow(log(1 + abs(fftshift(F))), []);
% �Ƚ�һ�� D0 ��С���˲�Ч��
H1 = exp(-(U .^ 2 + V .^ 2) ./ (2 * 5 ^ 2));
H2 = exp(-(U .^ 2 + V .^ 2) ./ (2 * 20 ^ 2));
H3 = exp(-(U .^ 2 + V .^ 2) ./ (2 * 60 ^ 2));
H4 = exp(-(U .^ 2 + V .^ 2) ./ (2 * 160 ^ 2));
figure;
% D0 Խ��Ƶ����Խ��ͨ��Ƶ��Խ�࣬��Ƶ˥����С���ռ�˱�С��ģ��Ч��ԽС
subplot(2, 2, 1), imshow(dftfilt(f, H1), []), subplot(2, 2, 2), imshow(dftfilt(f, H2), []);
subplot(2, 2, 3), imshow(dftfilt(f, H3), []), subplot(2, 2, 4), imshow(dftfilt(f, H4), []);
%% �߿�ͼ�ͱ���ͼ
%%
% �� 4.5 �����߿�ͼ
H = fftshift(lpfilter('gaussian', 500, 500, 50));
figure;
mesh(H(1:50:500, 1:10:500));
figure;
mesh(H(1:50:500, 1:10:500));
colormap([0 0 0]);
% view(-25, 0); ����Ϊ 0
grid off;
axis off;
figure;
surf(H(1:50:500, 1:10:500)); % ����ͼ
colormap(gray);
shading interp; % ɾ��������
grid off, axis off;
% ����Ԫ����
[Y, X] = meshgrid(-2:0.1:2, -2:0.1:2);
Z = X .* (-X .^ 2 - Y .^ 2);
figure;
subplot(1, 2, 1), mesh(Z), subplot(1, 2, 2), surf(Z);
%% ��Ƶ���˲���
%%
% 1 - lp = hp ��� hpfilter
H = fftshift(hpfilter('ideal', 500, 500, 50));
figure;
mesh(H(1:10:500, 1:10:500));
colormap([0, 0, 0]);
view(-37, 30);
grid off, axis off;
H = fftshift(hpfilter('btw', 500, 500, 50, 2)); % 2 �װ�����˹�˲���
figure;
subplot(1, 2, 1);
mesh(H(1:10:500, 1:10:500));
colormap([0, 0, 0]);
view(-37, 30);
%grid off, axis off;
subplot(1, 2, 2);
subplot(1, 2, 2);
imshow(H, []);
% �� 4.7 ��ͨ�˲�
% PS ֮ǰѧϰ���ռ���õ�Ƶ������˲��������Դ��� dftfilt ��Ҫ��������Ҷ������������˲�������Ƶ�����ڵ��˲���
% ���Դ˴� dftfilt ֱ�Ӵ��� H ���ɣ�ע������
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0413(a)(original_test_pattern).tif');
PQ = paddedsize(size(f)); % Ҫ�����˲�
D0 = 0.05 * PQ(2);
H = hpfilter('gaussian', PQ(1), PQ(2), D0); % ���������Ͻǣ����Ǿ�������
g = dftfilt(f, H);
figure;
subplot(1, 2, 1);
imshow(fftshift(H), []); % �˲�����ͼ��
subplot(1, 2, 2);
imshow(g, []); % ����ʧȥɫ�����������ø�ͨǿ���˲�����
% ��ͨǿ���˲�
% �� 4.8 ��Ƶǿ���˲� + ֱ��ͼ����
f = imread('./DIP-Ex/pic/dipum_images_ch04/Fig0419(a)(chestXray_original).tif');
figure;
imshow(f);
PQ = paddedsize(size(f)); % Ҫ���
D0 = 0.05 * PQ(2);
% ��ͨ������˹ 2 ��
HBW = hpfilter('btw', PQ(1), PQ(2), D0, 2);
a = 0.5;
b = 2;
H = a + b * HBW; % ��ͨǿ���˲���ʽ
gbw = dftfilt(f, HBW); % ��ͨ������˹�˲������ڱ���ɫռ�࣬��Ч���ܲ�
ghf = dftfilt(f, H);
ghe = histeq(mat2gray(ghf), 256); % ghf ���Բ�̫���ԣ���Ҫ�Աȶ���ǿ��ֱ��ͼ����, ע��ӳ�䵽 0~1
figure;
subplot(2, 2, 1);
imshow(gbw, []);
subplot(2, 2, 2);
imshow(ghf, []);
subplot(2, 2, 3);
imshow(ghe, []);
%% ��������
%%
% fft ִ��ʱ��ȡ���� P Q ������������P Q Ϊ 2 �����ٶȸ���
% ����Ҷ�任����֧����䣬������Ҫһ���������ĺ���
%%
function PQ = paddedsize(AB, CD, PARAM) % AB �� [1, 2] ����������, ���ص�Ҳ�� [a, b] �� a b ������Ҫ���ĳ���
    if nargin == 1
        PQ = 2 * AB; % ����ͼ���� M * N ��Ȼ����Сֱ�ӳ� 2 �Ϳ��ԡ�
    % ˵�� AB, CD ������
    elseif nargin == 2 & ~ischar(CD)
        PQ = AB + CD - 1;
        PQ = 2 * ceil(PQ / 2); % PQ ��Ԫ������Ҫ��С�� AB + CD - 1��ԭ�����ڱ����۵����
    elseif nargin == 2
        m = max(AB);
        P = 2 ^ nextpow2(2 * m); % ������С�ı� 2m ��� 2 ��ָ����
        PQ = [P, P];
    end
end

% ��ͨ�˲���
function H = hpfilter(type, M, N, D0, n)
    if nargin == 4
        n = 1;
    end
    Hlp = lpfilter(type, M, N, D0, n);
    H = 1 - Hlp;
end

% ��ͨ�˲���
function [H, D] = lpfilter(type, M, N, D0, n)
    % LPFILTER Computes frequency domain lowpass filters
    %   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of a
    %   lowpass filter, H, of the specified TYPE and size (M-by-N). To view the
    %   filter as an image or mesh plot, it should be centered using H =
    %   fftshift(H)
    %   Valid value for TYPE, D0, and n are:
    %   'ideal' Ideal lowpass filter with cutoff frequency D0. n need not be
    %           supplied. D0 must be positive.
    %   'btw'   Butterworth lowpass filter of order n, and cutoff D0. The
    %           default value for n is 1.0. D0 must be positive.
    %   'gaussian'Gaussian lowpass filter with cutoff (standard deviation) D0.
    %           n need not be supplied. D0 must be positive.
    %
    % �õ�ָ�����͵ĵ�ͨ�˲���
    
    % Use function dftuv to set up the meshgrid arrays needed for computing the
    % required distances.
    [U, V] = dftuv(M, N);
    % Compute the distances D(U, V)
    D = sqrt(U.^2 + V.^2);
    % Begin filter computations
    switch type
        case 'ideal'
            H = double(D <= D0);
        % ������˹
        case 'btw'
            if nargin == 4
                n = 1;
            end
            H = 1 ./ (1 + (D ./ D0) .^ (2 * n));
        case 'gaussian'
            H = exp(-(D .^ 2) ./ (2 * (D0 ^ 2)));
        otherwise
            error('Unkown filter type.')
    end
end

function [U, V] = dftuv(M, N)
    % DFTUV Computes meshgrid frequency matrices.
    % [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and V. U and
    % V are useful for computing frequency-domain filter functions that can be
    % used with DFTFILT. U and V are both M-by-N.
    % more details to see the textbook Page 93
    %
    % [U��V] = DFTUV��M��N����������Ƶ�ʾ���U��V�� U��V���ڼ������DFTFILTһ��ʹ�õ�
    % Ƶ���˲������������á� U��V����M-by-N������ϸ�ڼ�������˹�̲�93ҳ
    
    % Set up range of variables.
    % ���ñ�����Χ
    u = 0 : (M - 1);
    v = 0 : (N - 1);
    
    % Compute the indices for use in meshgrid.
    % ������������������������ԭ��ת�Ƶ����Ͻǣ���ΪFFT����ʱ�任��ԭ�������Ͻǡ�
    idx = find(u > M / 2);
    u(idx) = u(idx) - M;
    idy = find(v > N / 2);
    v(idy) = v(idy) - N;
    
    % Compute the meshgrid arrays.
    % �����������
    [V, U] = meshgrid(v, u);
end


% �Զ���һ�� DFT �˲����� Ƶ�����˲�
function g = dftfilt(f, H) % f ͼ��H Ƶ���˲��� % �˲����� H ��Ҫ���л�(�ο�ǰͷ���Ľ�����Ƶ��
    % Ƶ������ʹ����䣬�˲�����ͼ���СҪ��ͬ
    F = fft2(f, size(H, 1), size(H, 2));
    g = real(ifft2(F .* H)); % ��任ȡʵ��
    g = g(1:size(f, 1), 1:size(f, 2)); % �ü���ԭͼ��С
end