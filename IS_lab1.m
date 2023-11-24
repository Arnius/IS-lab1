% Atliko: Arnoldas Gerulskis KTfm-23
clear
clc

% Paruošiamos duomenų matricos
x1 = zeros(1,13);
x2 = zeros(1,13);
d = zeros(1,13);
arraySize = size(x1,2);
fdArray = zeros(1,3);
dataFile = fopen('Data.txt','r');

% fscanf skaitymo formatas
formatspec = '%f %f %d';
sizeA = [1 3];
% Nuskaitomi failo eilutės duomenys iteratyviai
for iter = 1:arraySize
    fdArray = fscanf(dataFile, formatspec, sizeA);
    x1(iter) = fdArray(1);
    x2(iter) = fdArray(2);
    d(iter) = fdArray(3);
end

% Dalis duomenų paiimami, kaip mokymo duomenys
x1_m = [x1(1,3:5) x1(1,10:11)];
x2_m = [x2(1,3:5) x2(1,10:11)];
d_m = [d(1,3:5) d(1,10:11)];
arr_m_size = length(x1_m);

% Atsitiktiniu būdu parenkami svorio koeficientai
disp('Pradinės koeficientų vertės:')
w1 = randn(1);
w2 = randn(1);
b = randn(1);
n = 0.5;

% Sukuriami atsako, aktyvavimo, klaidos vektoriai
v = 1:arraySize;
y = 1:arraySize;
e = 1:arraySize;

% Skaičiuojama pirminio tinklo klaida
totalErr = 0;
for iter = 1:arr_m_size
    v(1,iter) = w1*x1_m(1,iter) + w2*x2_m(1,iter) + b;
    
    if v(1,iter) > 0
        y(1,iter) = 1;
    else 
        y(1,iter) = -1;
    end
    
    e(1,iter) = d_m(1,iter) - y(1,iter);

    totalErr = totalErr + abs(e(1,iter));
end

% Mokymosi algoritmas, kol e > 0
while totalErr ~= 0 

% Parametrų atnaujinimas
    for iter = 1:arr_m_size
        w1 = w1 + n*e(1,iter)*x1_m(1,iter);
        w2 = w2 + n*e(1,iter)*x2_m(1,iter);
        b = b +n*e(1,iter);
    end

    % Bandomi nauji parametrai. Skaičiuojamas atsakas, aktyvavimo funkcija
    % ir klaidos vertė
    totalErr = 0;
    for iter = 1:arr_m_size
        v(iter) = w1*x1_m(1,iter) + w2*x2_m(1,iter) + b;

        if v(1,iter) > 0
            y(1,iter) = 1;
        else 
            y(1,iter) = -1;
        end

        e(1,iter) = d_m(1,iter) - y(1,iter);

        totalErr = totalErr + abs(e(1,iter));
    end
end

% Bandoma su visais duomenimis
for indx=1:arraySize
    v(1,indx) = w1*x1(1,indx) + w2*x2(1,indx) + b;

    if v(1,indx) > 0
        y(1,indx) = 1;
    else 
        y(1,indx) = -1;
    end
end

disp('Galutinės koeficientų vertės:')
w1
w2
b
d
y

%% IS lab1 antra užduotis
% Naive Bayes klasifikatorius
clear
clc

% Nuskaitomi duomenys iš failo
x1 = zeros(1,13);
x2 = zeros(1,13);
d = zeros(1,13);
fdArray = zeros(1,3);
dataFile = fopen('Data.txt','r');
formatspec = '%f %f %d';
sizeA = [1 3];
arraySize = size(x1,2);

for iter = 1:arraySize
    fdArray = fscanf(dataFile, formatspec, sizeA);
    x1(iter) = fdArray(1);
    x2(iter) = fdArray(2);
    d(iter) = fdArray(3);
end

% Paskaičiuojama kiek yra kriaušiu ir obuolių iš d vektoriaus
o_sk = 0;
k_sk = 0;
for iter = 1:arraySize
    if d(iter) == 1
        o_sk = o_sk + 1;
    else
        k_sk = k_sk + 1;
    end
end

%Sukuriami nauji x1 ir x2 vektoriai, kur žinomi vaisių parametrai atskirti
x1_o = zeros(1, o_sk); % obuolio 1 parametro įejimo vektorius
x1_k = zeros(1, k_sk); % kriauses 1 parametro įejimo vektorius
x2_o = zeros(1, o_sk); % obuolio 2 parametro įejimo vektorius
x2_k = zeros(1, k_sk); % kriauses 2 parametro įejimo vektorius

% Išrusiuojami duomenys į naujus vektorius
for iter = 1:arraySize
    if d(iter) == 1
        x1_o(iter) = x1(iter);
        x2_o(iter) = x2(iter);
    else
        x1_k(iter) = x1(iter);
        x2_k(iter) = x2(iter);
    end
end

% Paskaičiuojamas vidurkis ir nuokrypis skirtingų vaisių parametrų
% vektoriams
% x1 skaiciavimai
% Obuoliams
x1_m_o = mean(x1_o); 
x1_d_o = std(x1_o);

% Kriaušėms
x1_m_k = mean(x1_k);
x1_d_k = std(x1_k);

% x2 skaiciavimai
% Obuoliams
x2_m_o = mean(x2_o);
x2_d_o = std(x2_o);

% Kriaušėms
x2_m_k = mean(x2_k);
x2_d_k = std(x2_k);

% Paskaičiuojama atsitiktinė tikimybė iš surinktų duomenų, kad bus obuolys (O_t) arba kriause (K_t)
O_t = o_sk / arraySize;
K_t = k_sk / arraySize;

% Paruošiami vektoriai bandymų rezultatams
v_o = zeros(1, arraySize);
v_k = zeros(1, arraySize);
y = zeros(1, arraySize);
e= zeros(1, arraySize);
totalErr = 0;

% Paruošiami tarpiniu skaiciavimu vektoriai
p_x1_o = zeros(1, arraySize);  % P(x1|Obuoliai)
p_x1_k = zeros(1, arraySize);  % P(x1|Kriauses)
p_x2_o = zeros(1, arraySize);  % P(x2|Obuoliai)
p_x2_k = zeros(1, arraySize);  % P(x2|Kriauses)

% Skaičiuojamos vertės žinomiems duomenims pagal Gauso skirstinį (funkcija
% - g_skirstinys.m faile
for iter = 1:arraySize
    p_x1_o(iter) = g_skirstinys(x1(iter), x1_m_o, x1_d_o);
    p_x1_k(iter) = g_skirstinys(x1(iter), x1_m_k, x1_d_k);
    p_x2_o(iter) = g_skirstinys(x2(iter), x2_m_o, x2_d_o);
    p_x2_k(iter) = g_skirstinys(x2(iter), x2_m_k, x2_d_k);
end

% Atliekami bandymai su žinomais x1 ir x2 parametrais
for iter=1:arraySize
    v_o(iter) = O_t * p_x1_o(iter) * p_x2_o(iter);
    v_k(iter) = K_t * p_x1_k(iter) * p_x2_k(iter);
    if v_o(iter) > v_k(iter)
        y(iter) = 1;
    else
        y(iter) = -1;
    end
    e(iter) = d(iter) - y(iter);
    totalErr = totalErr + abs(e(iter));
end
disp("Total error:")
totalErr

% Papildomas bandymas: Duodami atsitiktiniai x1 ir x2 kriaušės parametrai
% Kriauses duomenys
x_1 = 0.15931;
x_2 = 0.6897;

% Obuolio duomenys
x_3 = 0.4875
x_4 = 0.8562

% Gauso skirstinys P(x|Kriause)
% p_x_1_o = g_skirstinys(x_1, x1_m_o, x1_d_o);
% p_x_1_k = g_skirstinys(x_1, x1_m_k, x1_d_k);
% p_x_2_o = g_skirstinys(x_2, x2_m_o, x2_d_o);
% p_x_2_k = g_skirstinys(x_2, x2_m_k, x2_d_k);

% Gauso skirstinys P(x|Obuolys) 
p_x_1_o = g_skirstinys(x_3, x1_m_o, x1_d_o);
p_x_1_k = g_skirstinys(x_3, x1_m_k, x1_d_k);
p_x_2_o = g_skirstinys(x_4, x2_m_o, x2_d_o);
p_x_2_k = g_skirstinys(x_4, x2_m_k, x2_d_k);

v_oo = O_t * p_x_1_o * p_x_2_o;
v_kk = K_t * p_x_1_k * p_x_2_k;
if v_oo > v_kk
    yy = 1
else
    yy = -1
end
