%% Load Raw Data From .CSV File %%
train = csvread('../Data/mitbih_train.csv');
test = csvread('../Data/mitbih_test.csv');
trainLabel = train(:,end);
testLabel = test(:,end);
train = train(:,1:end-1);
test = test(:,1:end-1);

%% Compute DFT Features %%
trDFT = abs(fft(train.')).';
teDFT = abs(fft(test.')).';

%% Compute Wavelet Features %%
% Wavelet (db4)
n_db4 = length(dwt(train(1,:), 'db4'));
trWav= zeros(size(train,1), n_db4);
teWav= zeros(size(test,1),  n_db4);
for i = 1:size(train,1)
    if (rem(i,10000)==0)
        fprintf('db4 training set: %.1f%% done\n', 100*i/size(train,1));
    end
    trWav(i,:) = dwt(train(i,:), 'db4');
end
for i = 1:size(test,1)
    if (rem(i,10000)==0)
        fprintf('db4 test set: %.1f%% done\n', 100*i/size(test,1));
    end
    teWav(i,:) = dwt(test(i,:), 'db4');
end 

% Wavelet (sym5)
n_S5 = length(dwt(train(1,:), 'sym5'));
trWavS5= zeros(size(train,1), n_S5);
teWavS5= zeros(size(test,1),  n_S5);
for i= 1:size(train,1)
    if (rem(i,10000)==0)
        fprintf('sym5 training set: %.1f%% done\n', 100*i/size(train,1));
    end
    trWavS5(i,:) = dwt(train(i,:), 'sym5');
end
for i= 1:size(test,1)
    if (rem(i,10000)==0)
        fprintf('sym5 test set: %.1f%% done\n', 100*i/size(test,1));
    end
    teWavS5(i,:) = dwt(test(i,:), 'sym5');
end

fprintf('All done!\n');

%% Save Features %%
save('all_features.m');
