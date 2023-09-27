% Atliko: Arnoldas Gerulskis KTfm-23
%% train single perceptron with two inputs and one output
clear
clc

% Read data from file
x1 = zeros(1,13);
x2 = zeros(1,13);
d = zeros(1,13);
arraySize = size(x1,2);
fdArray = zeros(1,3);
dataFile = fopen('Data.txt','r');

% fscanf parameters
formatspec = '%f %f %d';
sizeA = [1 3];
% Nuskaityti visus duomenis
for iter = 1:arraySize
    fdArray = fscanf(dataFile, formatspec, sizeA);
    x1(iter) = fdArray(1);
    x2(iter) = fdArray(2);
    d(iter) = fdArray(3);
end

% generate random initial values of w1, w2 and b
disp('Initial weight & bias values:')
w1 = randn(1)
w2 = randn(1)
b = randn(1)
n = 0.25;

v = 1:arraySize;
y = 1:arraySize;
e = 1:arraySize;

% calculate the output and total error for all inputs
totalErr = 0;
for iter = 1:arraySize
    v(iter) = w1*x1(iter) + w2*x2(iter) + b;
    
    if v(iter) > 0
        y(iter) = 1;
    else 
        y(iter) = -1;
    end
    
    e(iter) = d(iter) - y(iter);

    totalErr = totalErr + abs(e(iter));
end

disp('Total error value:')
totalErr

% write training algorithm
while totalErr ~= 0 % executes while the total error is not 0
disp('Re-learning!')
% here should be your code of parameter update
    for iter = 1:arraySize
        w1 = w1 + n*e(iter)*x1(iter);
        w2 = w2 + n*e(iter)*x2(iter);
        b = b +n*e(iter);
    end
% Parameter update end

%   Test how good are updated parameters (weights) on all examples used for training
%   calculate outputs and errors for all 5 examples using current values of the parameter set {w1, w2, b}
%   calculate 'v1', 'v2', 'v3',... 'v5'
    totalErr = 0;
    for iter = 1:arraySize
        v(iter) = w1*x1(iter) + w2*x2(iter) + b;
        
        if v(iter) > 0
            y(iter) = 1;
        else 
            y(iter) = -1;
        end

        e(iter) = d(iter) - y(iter);

        totalErr = totalErr + abs(e(iter));
    end
    
    disp('Total error:')
    totalErr
end

disp('Final weight and bias values:')
w1
w2
b
disp('Script finished!')

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
x_1 = 0.15931;
x_2 = 0.6897;

p_x_1_o = g_skirstinys(x_1, x1_m_o, x1_d_o);
p_x_1_k = g_skirstinys(x_1, x1_m_k, x1_d_k);
p_x_2_o = g_skirstinys(x_2, x2_m_o, x2_d_o);
p_x_2_k = g_skirstinys(x_2, x2_m_k, x2_d_k);

v_oo = O_t * p_x_1_o * p_x_2_o;
v_kk = K_t * p_x_1_k * p_x_2_k;
if v_oo > v_kk
    yy = 1
else
    yy = -1
end
