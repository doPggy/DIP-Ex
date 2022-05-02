%% ͼ��ԭ
%% ����ģ��
%%
% ʹ�� imnoise �������
% imnoise(f, type, param)

% 5.2.2 ָ���ֲ������ռ��������
% �� 5.1 ������˿����ᵽ�������ֲ�������������Ĺ�ʽ��ô����(�ӷֲ��������������)
% ���Զ�Ӧ�ķֲ������Ǹ���Ȥ�����������ʽ����������� 5.1 ������
% rand ����Ԫ�ض��� 0,1 �ڵľ��ȷֲ�
A = rand(2, 2)
A = randn(2, 2) % ����Ԫ��Ϊ 0 ��ֵ����λ�������̬��
% find ����
A = [11 22 33; 44 55 66; 77 88 0];
I = find(A); % ���ؾ���Ԫ��������������ֱ�˾���
A(I); % �����ȱ��������Ƿ���Ԫ�ص�����
[r, c] = find(A); % ����Ԫ���к�������, Ҳ�������ȵ��߼�
[r, c, v] = find(A);
I = find(A < 88);
A(I) = 100;
A
% �� 5.2 imnoise2 ����ֱ��ͼ
figure;

r1 = imnoise2('gaussian', 10000, 1, 0, 1);
r2 = imnoise2('uniform', 10000, 1);
r3 = imnoise2('lognormal', 10000, 1);
r4 = imnoise2('rayleigh', 10000, 1);
r5 = imnoise2('exponential', 10000, 1);
r6 = imnoise2('erlang', 10000, 1);
% hist ����ֱ��ͼ������ imhist
subplot(3, 2, 1);
hist(r1, 50);
subplot(3, 2, 2);
hist(r2, 50);
subplot(3, 2, 3);
hist(r3, 50);
subplot(3, 2, 4);
hist(r4, 50);
subplot(3, 2, 5);
hist(r5, 50);
subplot(3, 2, 6);
hist(r6, 50);
% 5.2.3 ��������
% �� 5.3 ʹ�� imnoise3 ����

% 5.2.4 ������������
% ����ͼ����ƾ�ֵ�����ֱ��ͼ���ڹ������ a b(�� 5.1)
f = imread('./DIP-Ex/pic/dipum_images_ch05/Fig0504(a)(noisy_image).tif');
figure;
imshow(f, []);
figure;
[B, c, r] = roipoly(f); % ����ʽ���� mask
figure;
imshow(B);
[p, npix] = histroi(f, c, r);
figure, bar(p, 1); % ͨ�� roipoly ѡȡ����
[v, unv] = statmoments(p, 2);
v
unv % ��ֵ������
% �ù�������ģ�ͺ�����ͳ�Ƶõ��ľ�ֵ����õ������Ƶĸ�˹�ֲ�
X = imnoise2('gaussian', npix, 1, 147, 20); % ������ unv(2) �������޸�
figure, hist(X, 130); % ���� imhist��imhist ����ͼƬͳ�ƻҶ�ֵ����
axis([0 300 0 140]); 
%% ����������ԭ ���� �ռ��˲�
%%
% �� 5.5 
f = imread('./DIP-Ex/pic/dipum_images_ch05/Fig0504(a)(noisy_image).tif');
figure;
% ע��ʽ�ӵõ��Ĳ�� 1 �� 2 ʽ��һ�µġ�
% g = imfilter(0.1 * f, ones(3, 3), 'replicate');
% g = imfilter(f, 0.1 * ones(3, 3), 'replicate');
% g = imfilter(f, ones(3, 3), 'replicate');imshow(g);

[M, N] = size(f);
R = imnoise2('salt & pepper', M, N, 0.1, 0); % ���� 0.1 �Ľ�����
% ��Ⱦ���������ַ�
c = find(R == 0); % R �� 0 �Ķ��Ǵ���Ϊ��������
gp_0 = f;
gp_0(c) = 0; % �Ѷ�Ӧ�����ɫ�����ǽ������� 
imshow(gp_0);
% ��Ⱦ������
R = imnoise2('salt & pepper', M, N, 0, 0.1); % ���������� 0.1
c = find(R == 1);
gp_255 = f;
% ! gp(c) = 1; ������ˣ�����ҪŪ�ɰ�ɫ����ɫ�� 255������
gp_255(c) = 255;
figure;
imshow(gp_255, []);
% ���˽������� Q Ϊ���ķ������˲�
gp_0_chmean = spfilt(gp_0, 'chmean', 3, 3, 1.5); % q = 1.5
figure;
imshow(gp_0_chmean, []);
% Q Ϊ��ֵ�ķ������˲�����������
gp_255_chmean = spfilt(gp_255, 'chmean', 3, 3, -1.5); % q = 1.5
figure;
imshow(gp_255_chmean, []);
% �˲�����С��Ϊ 3 * 3
% �����ֵ�˲��Խ�����(�����ڽ������� 0��Ҫ�����Ĵ��棬���Կ���)
gp_0_max = spfilt(gp_0, 'max', 3, 3); % q = 1.5
% ͬ����Сֵ�˲�����������
gp_255_max = spfilt(gp_255, 'min', 3, 3); % q = 1.5

figure;
subplot(1, 2, 1), imshow(gp_0_max, []), subplot(1, 2, 2), imshow(gp_255_max, []);
% ����Ӧ�ռ��˲���
% ����Ӧ�ռ��˲����������㼶����ĺ��� �ؼ��� adpmedian ʵ�֣����� �� 5.6


%% �˻�������ģ
%%
f = checkerboard(8);
figure;
imshow(f, []);
% �� fspecial �����˶�ģ�����˻�ģ��֮һ
PSF = imfilter(f, fspecial('motion', 7, 45));
gb = imfilter(f, PSF, 'circular');
figure;
% fspecial('motion', len, angle) len ָ���˶��ĳ��ȣ�theta ����ʱ�뷽�����ָ���˶��ĽǶ� ���Ϸ���������
subplot(1, 2, 1), imshow(imfilter(f, fspecial('motion', 7, 45), 'circular'), []); % �߸����ص㣬˳ʱ�� 45��
subplot(1, 2, 2), imshow(imfilter(f, fspecial('motion', 7, -45), 'circular'), []);
noise = imnoise(zeros(size(f)), 'gaussian', 0, 0.001); % imnoise ����ֱ�ӽ�ĳ�����ӵ�ͼ���ϣ��˴��ǵ�����������
figure;
imshow(noise, []);
g = gb + noise; % �˻�ͼ��ģ�ͣ�H * G + noise(H * G �ڿռ������������ʹ�� imfilter)
figure;
imshow(pixeldup(g, 8), []);
% ���� pixeldup ���ƷŴ�
figure;
imshow(pixeldup(f, 8), []); % ���������� 8
%% ֱ�����˲���ԭͼ��
%%
% F = G / H �������Ҫ��������Ƶ������ʵ�֣�Ȼ�� F Ҫ�������˲�

%% ά���˲���ԭͼ�� 
%%
% ��ά���˲��У�������һ������ R = Na / fa ���������űȣ����� R �ý���ʽ��������
% g �����Ĺ�����˻�ͼ��
% �൱��ֱ���˲�
fr1 = deconvwnr(g, PSF); 
figure;
imshow(fr1, []);
%imshow(pixeldup(fr1, 8), []);
% Sn �����Ĺ�����
% Sf δ�˻���ͼ������
% Na ƽ����������
% fA ƽ��ͼ����
Sn = abs(fft2(noise)) .^ 2; % fft2 �õ��ģ���Ƶ�ʾ��Σ�ÿ���㶼�Ǿ��� DFT�� abs �õ�Ƶ��
NA = sum(Sn(:)) / prod(size(noise)); % 1/MN * sum{Sn(u, v)}
Sf = abs(fft2(f)) .^ 2;
fA = sum(Sf(:)) / prod(size(f));
R = NA / fA; % ���߽���ʽ���
fr2 = deconvwnr(g, PSF, R); % �� R ����
figure;
imshow(pixeldup(fr2, 8), []);
% F(u, v) ^ 2 = dft(f(x, y) �� f(x,y)) dft ������Ҷ�任���� �˴��������
% ���Թ����׵���任��������غ���
% ��غ;���Ĳ��� ��4.3
NCORR = fftshift(real(ifft2(Sn)));
ICORR = fftshift(real(ifft2(Sf)));
% ��ͼ�� I ���з���������� ncorr ������������غ�����icorr ��ԭʼͼ�������غ�����
%fr3 = deconvwnr(g, PSF, NCORR, ICORR);
fr3 = deconvwnr(g, PSF, real(ifft2(Sn)), real(ifft2(Sf)));
figure;
imshow(pixeldup(fr3, 8), []);
%% ��������
%%
% ά���˲���ԭ�У�Ϊʲô��ԭͼ�෴
%% �����Զ�����
%%
function R = imnoise2(type, M, N, a, b) % ��������ģ��
    if nargin == 1
        a = 0; b = 1;
        M = 1; N = 1;
    elseif nargin == 3
        a = 0; b = 1;
    end
    
    % ����� 5.1���Ƶ���ʽ���ֲ���������
    switch lower(type)
        case 'uniform'
            R = a + (b - a) * rand(M, N);
        case 'gaussian'
            R = a + b * randn(M, N); % Ϊ�������� ����
        case 'salt & pepper'
            if nargin <= 3
                a = 0.05; b = 0.05;
            end
            if (a + b > 1)
                error('Pa + Pb < 1');
            end
            R(1:M, 1:N) = 0.5; % 0.5 ��������0 ����ڽ���1 ����������
            X = rand(M, N);
            c = find(X <= a);
            R(c) = 0;
            u = a + b;
            c = find(X > a & X <= u); % Pb * (M * N) �ĵ�
            R(c) = 1;
        case 'lognormal'
            if nargin <= 3
                a = 1; b = 0.25;
            end
            R = a * exp(b * randn(M, N));
        % ������Ǵ����Ƶ�������
        case 'rayleigh'
            R = a + (-b * log(1 - rand(M, N))) .^ 0.5;
        % ָ��
        case 'exponential'
            if nargin <= 3
                a = 1;
            end
            if a <= 0
                error('cao')
            end
            k = -1/a;
            R = k * log(1 - rand(M, N));
        % 
        case 'erlang'
            if nargin <= 3
                a = 2; b = 5;
            end
            if (b ~= round(b) | b <= 0)
                error('positive integer');
            end
            k = -1 / a;
            R = zeros(M, N);
            for j = 1:b
                R = R + k * log(1 - rand(M, N))
            end
        otherwise
            error('unknow')
    end
    
    
    
end

% ����һ��ͼ���ڶ���������ڵ�ֱ��ͼ
% ע�� roipoly �÷�
function [p, npix] = histroi(f, c, r)
    B = roipoly(f, c, r); % �����Ǹ���Ȥ����Ķ�ֵͼ�񣬸���Ȥ����Ϊ 1
    p = imhist(f(B)); % ͳ�Ƹ���Ȥ����ֱ��ͼ��ȡ f �Ĳ�����������ֻ�� 0 1 �ľ��� mask
    if nargout > 1
        npix = sum(B(:));
    end
end




% ���� p �� n �׾�
function [v,unv] = statmoments(p, n)
%���������Ը�����˹��matlab�鼮
    Lp = length(p);
    if(Lp~=256) & (Lp~=65536)
        error('P must be a 256- or 65536_element vector.');
    end
    G=Lp-1;
    p=p/sum(p);
    p=p(:);
    z=0:G;
    z=z./G;
    m=z*p;
    z=z-m;
    v=zeros(1,n);
    v(1)=m;
    for j=2:n
        v(j)=(z.^j)*p;
    end
    if nargout > 1
        % compute the uncentralized moments.
        unv=zeros(1,n);
        unv(1)=m.*G;
    end
    for j=2:n
        unv(j)=((z*G).^j)*p;
    end
end

% ����ռ��˲��� m n �Ǻ˵Ĵ�С
function f = spfilt(g, type, m, n, parameter)
    if nargin == 2
        m = 3; n = 3; Q = 1.5; d = 2;
    elseif nargin ==  5
        Q = parameter; d = parameter;
    elseif nargin ==  4
        Q = 1.5; d = 2;
    else
        error('input error');
    end
    
    switch type
        case 'amean'
            w = fspecial('average', [m n]); % ����ƽ�����ռ��˲���
            f = imfilter(g, w, 'replicate');
        case 'gmean' % ����ƽ��
            f = gmean(g, m, n);
        case 'hmean' % ?
            f = harmean(g, m, n);
        case 'chmean' % ? ������
            f = charmean(g, m, n, Q);
        case 'median'
            f = medfilt2(g, [m n], 'symmetric'); % ��ֵ�˲�
        case 'max'
            f= ordfilt2(g, m * n, ones(m, n), 'symmetric'); % ���ֵ�˲�
        case 'min'
            f = ordfilt2(g, 1, ones(m, n), 'symmetric');
        case 'midpoint' % �е��˲� �� ��ֵ�˲���
            f1 = ordfilt2(g, 1, ones(m, n), 'symmetric');
            f2 = ordfilt2(g, m * n, ones(m, n), 'symmetric');
            f = imlincomb(0.5, f1, 0.5, f2);
        case 'atrimmed' % ?
            if (d < 0) | (d / 2 ~= round(d / 2))
                error('nonnegative, integer');
            end
            f = alphatrim(g, m, n, d); % ?
        otherwise
            error('fuck you');
    end
end

function f = gmean(g, m, n)
    inclass = class(g);
    g = im2double(g);
    warning off;
    f = exp(imfilter(log(g), ones(m, n), 'replicate')) .^ ( 1 / m / n);
    warning on;
    f = changeclass(inclass, f);
end

% ע��ʵ�ַ�ʽ����ԭͼȫ���ȴ���Ȼ���� ȫ 1 �ĺ����˲�
function f = harmean(g, m, n)
    inclass = class(g);
    
    g = im2double(g);
    % ע���˲�����ʽ��ʲô
    f = m * n ./ imfilter(1 ./ (g + eps), ones(m, n), 'replicate');
    f = changeclass(inclass, f);
end

% �����;�ֵ
function f = charmean(g, m, n, q)
    inclass = class(g);
    g = im2double(g);
    f = imfilter(g.^(q+1), ones(m, n), 'replicate'); % ? imfilter ��������ʽ
    f = f ./ (imfilter(g.^q, ones(m, n), 'replicate') + eps);
    f = changeclass(inclass, f);
end

% ˳�� - ƽ���ֵ�˲�(������������ֵ�˲�)
function f = alphatrim(g, m, n, d)
    % alpha
    inclass = class(g);
    g = im2double(g);
    f = imfilter(g, ones(m, n), 'symmetric');
    for k = 1:d/2
        % ?  imsubtract ������ X �е�ÿ��Ԫ���м�ȥ���� Y �еĶ�ӦԪ�� ���������뵽 0
        f = imsubtract(g, ones(m, n), 'symmetric');
    end
    for k = (m*n - (d/2) + 1): m*n
        f = imsubtract(f, ordfilt2(g, k, ones(m, n), 'symmetric'));
    end
    f = f / (m*n - d);
    f = changeclass(inclass, f);
end

function image = changeclass(class, varargin)
    %CHANGECLASS changes the storage class of an image.
    %   I2 = CHANGECLASS(CLASS, I);
    %   RGB2 = CHANGECLASS(CLASS, RGB);
    %   BW2 = CHANGECLASS(CLASS, BW);
    %   X2 = CHANAGECLASS(CLASS, X, 'indexed');
    %   Copyright 1993-2002 The MathWorks, Inc    
    %   $Revision: 1.2 $ $Date:2003/02/19 22:09:58 $

    switch class
    case 'uint8'
       image = im2uint8(varargin{:});
    case 'uint16'
       image = im2uint16(varargin{:});
    case 'double'
       image = im2double(varargin{:});
    otherwise
       error('Unsupported IPT data class.');
    end
end

function f = adpmedian(g, Smax)
    % ADPMEDIAN Perform adaptivev median filtering.
    %   F = ADPMEDIAN(G, SMAX) performs adaptive median filtering of image G.
    %   The median filter starts at size 3-by-3 and iterates up to size
    %   SMAX-by-SMAX. SMAX must be an odd integer greater than 1.
    
    % SMAX must be an odd, positive integer greater than 1.
    if (Smax <= 1)|(Smax/2 == round(Smax/2))|(Smax ~= round(Smax))
        error('Smax must be an odd integer >1.')
    end
    % Initial setup
    f = g;
    f(:) = 0;
    alreadyProcessed = false(size(g));
    % Begin filtering
    for k = 3:2:Smax
        zmin = ordfilt2(g, 1, ones(k,k), 'symmetric');
        zmax = ordfilt2(g, k*k, ones(k,k), 'symmetric');
        zmed = medfilt2(g, [k k], 'symmetric');
        processUsingLevelB = (zmed > zmin) & (zmax > zmed) &  ~alreadyProcessed;
        zB = (g > zmin) & (zmax > g);
        outputZxy = processUsingLevelB & zB;
        outputZmed = processUsingLevelB & ~zB;
        f(outputZxy) = g(outputZxy);
        f(outputZmed) = zmed(outputZmed);
        
        alreadyProcessed = alreadyProcessed | processUsingLevelB;
        if all(alreadyProcessed(:))
            break;
        end
    end
    % Output zmed for any remaining unprocessed pixels. Note that this zmed was
    % computed using a window of size Smax-by-Smax, which is the final value of
    % k in the loop.
    f(~alreadyProcessed) = zmed(~alreadyProcessed);
end

% 
function B = pixeldup(A, m, n)
    %PIXELDUP Duplicates pixels of an image in both directions.
    %   B = PIXELDUP(A, M, N) duplicates each pixel of A M times in the
    %   vertical direction and N times in the horizontal direction.
    %   Parameters M and N must be integers.  If N is not included, it
    %   defaults to M.
    
    %   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
    %   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
    %   $Revision: 1.5 $  $Date: 2005/01/03 00:01:28 $
    
    % Check inputs.
    if nargin < 2 
       error('At least two inputs are required.'); 
    end
    if nargin == 2 
       n = m; 
    end
    
    % Generate a vector with elements 1:size(A, 1).
    u = 1:size(A, 1);
    
    % Duplicate each element of the vector m times.
    m = round(m); % Protect against nonintegers.
    u = u(ones(1, m), :);
    u = u(:);
    
    % Now repeat for the other direction.
    v = 1:size(A, 2);
    n = round(n);
    v = v(ones(1, n), :);
    v = v(:);
    B = A(u, v);
end


% imnoise3 ����