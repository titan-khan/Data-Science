opengl software;
rng('default')  % For reproducibility

hmax_cilacap = csvread('hmax_interim_cilacap.csv',1,1);
hmax_pangandaran = csvread('hmax_interim_pangandaran.csv',1,1);
hmax_indramayu = csvread('hmax_interim_indramayu.csv',1,1);
hmax_karangantu = csvread('hmax_interim_karangantu.csv',1,1);

%% Cilacap
figure(1)
y = hmax_cilacap;

h = cdfplot(y);
h.LineWidth = 2;
title('Empirical CDF Cilacap')
xlabel('Hmax [m]')
ylabel('Likelihood Occurance')
xlim([0, max(y)+1])

%% Pangandaran
figure(2)
y = hmax_pangandaran;

h = cdfplot(y);
h.LineWidth = 2;
title('Empirical CDF Pangandaran')
xlabel('Hmax [m]')
ylabel('Likelihood Occurance')
xlim([0, max(y)+1])

%% Indramayu
figure(3)
y = hmax_indramayu;

h = cdfplot(y);
h.LineWidth = 2;
title('Empirical CDF Indramayu')
xlabel('Hmax [m]')
ylabel('Likelihood Occurance')
xlim([0, max(y)+1])

%% Karangantu
figure(4)
y = hmax_karangantu;

h = cdfplot(y);
h.LineWidth = 2;
title('Empirical CDF Pelabuhan Karangantu')
xlabel('Hmax [m]')
ylabel('Likelihood Occurance')
xlim([0, max(y)+1])
