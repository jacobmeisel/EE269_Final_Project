%% Load Data %%
load('../Data/all_features.mat');

%% KNN- Raw (zero mean, normalized) %%
train_norm = normalize(train, 2);
test_norm = normalize(test, 2);

disp("K=1");
idx = knnsearch(train_norm, test_norm, 'K',1,'Distance', 'euclidean');
tePred = trainLabel(idx);
teAccK1RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 1) for normalized raw data accuracy: %.2f%%\n', 100*teAccK1RawNorm);

disp("K=3");
idx = knnsearch(train_norm, test_norm, 'K',3,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK3RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 3) for normalized raw data accuracy: %.2f%%\n', 100*teAccK3RawNorm);


disp("K=5");
idx = knnsearch(train_norm, test_norm, 'K',5,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK5RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 5) for normalized raw data accuracy: %.2f%%\n', 100*teAccK5RawNorm);


%% KNN- DFT %%
disp("K=1");
idx = knnsearch(trDFT,teDFT,'K',1,'Distance', 'euclidean');
tePred = trainLabel(idx);
teAccK1FFT = sum(tePred == testLabel)./(length(testLabel));

disp("K=3");
idx = knnsearch(trDFT,teDFT,'K',3,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK3FFT = sum(tePred == testLabel)./(length(testLabel));

disp("K=5");
idx = knnsearch(trDFT,teDFT,'K',5,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK5FFT = sum(tePred == testLabel)./(length(testLabel));

%% KNN- DWT (db4) %%
disp("K=1");
idx2 = knnsearch(trWav,teWav,'K',1,'Distance', 'euclidean');
tePredW = trainLabel(idx2);
teAccK1DWT = sum(tePredW == testLabel)./(length(testLabel));

disp("K=3");
idx2 = knnsearch(trWav,teWav,'K',3,'Distance', 'euclidean');
tePredW = mode(trainLabel(idx2),2);
teAccK3DWT = sum(tePredW == testLabel)./(length(testLabel));

disp("K=5");
idx2 = knnsearch(trWav,teWav,'K',5,'Distance', 'euclidean');
tePredW = mode(trainLabel(idx2),2);
teAccK5DWT = sum(tePredW == testLabel)./(length(testLabel));

%% KNN- DWT (S5) %%
disp("K=1");
idx2 = knnsearch(trWavS5,teWavS5,'K',1,'Distance', 'euclidean');
tePredWS5 = trainLabel(idx2);
teAccK1DWTS5 = sum(tePredWS5 == testLabel)./(length(testLabel));

disp("K=3");
idx2 = knnsearch(trWavS5,teWavS5,'K',3,'Distance', 'euclidean');
tePredWS5 = mode(trainLabel(idx2),2);
teAccK3DWTS5 = sum(tePredWS5 == testLabel)./(length(testLabel));

disp("K=5");
idx2 = knnsearch(trWavS5,teWavS5,'K',5,'Distance', 'euclidean');
tePredWS5 = mode(trainLabel(idx2),2);
teAccK5DWTS5 = sum(tePredWS5 == testLabel)./(length(testLabel));

%% KNN- Raw (zero mean, normalized) + db4 DWT%%
train_normWav = [normalize(train, 2), normalize(trWav, 2)];
test_normWav =  [normalize(test, 2),  normalize(teWav, 2)];

disp("K=1");
idx = knnsearch(train_normWav, test_normWav, 'K', 1,'Distance', 'euclidean');
tePred = trainLabel(idx);
teAccK1RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 1) for DWT (db4) + normalized raw data accuracy: %.2f%%\n', 100*teAccK1RawNorm);

disp("K=3");
idx = knnsearch(train_normWav, test_normWav, 'K', 3,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK3RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 3) for DWT (db4) + normalized raw data accuracy: %.2f%%\n', 100*teAccK3RawNorm);


disp("K=5");
idx = knnsearch(train_normWav, test_normWav, 'K', 5,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK5RawNorm = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 5) for DWT (db4) + normalized raw data accuracy: %.2f%%\n', 100*teAccK5RawNorm);

%% KNN- Raw (zero mean, normalized) + sym5 DWT%%
train_normS5 = [normalize(train, 2), normalize(trWavS5, 2)];
test_normS5 =  [normalize(test, 2),  normalize(teWavS5, 2)];

disp("K=1");
idx = knnsearch(train_normS5 , test_normS5, 'K', 1,'Distance', 'euclidean');
tePred = trainLabel(idx);
teAccK1RawNormS5 = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 1) for DWT (sym5) + normalized raw data accuracy: %.2f%%\n', 100*teAccK1RawNormS5);

disp("K=3");
idx = knnsearch(train_normS5 , test_normS5, 'K', 3,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK3RawNormS5 = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 3) for DWT (sym5) + normalized raw data accuracy: %.2f%%\n', 100*teAccK3RawNormS5);


disp("K=5");
idx = knnsearch(train_normS5 , test_normS5, 'K', 5,'Distance', 'euclidean');
tePred = mode(trainLabel(idx),2);
teAccK5RawNormS5 = sum(tePred == testLabel)./(length(testLabel));
fprintf('KNN (K = 5) for DWT (sym5) + normalized raw data accuracy: %.2f%%\n', 100*teAccK5RawNormS5);

%% KNN- Raw Data (Max Normalized Crosscovariance Distance Metric) %%
disp("K=1");
idx3 = knnsearch(train,test,'K',1,'Distance', @max_normalized_crosscovariance_dist_metric);
tePredRawNCCD = trainLabel(idx3);
teAccK1RawNCCD = sum(tePredRawNCCD == testLabel)./(length(testLabel));

disp("K=3");
idx3 = knnsearch(train,test,'K',3,'Distance', @max_normalized_crosscovariance_dist_metric);
tePredRawNCCD = trainLabel(idx3);
teAccK3RawNCCD = sum(tePredRawNCCD == testLabel)./(length(testLabel));

disp("K=4");
idx3 = knnsearch(train,test,'K',5,'Distance', @max_normalized_crosscovariance_dist_metric);
tePredRawNCCD = trainLabel(idx3);
teAccK4RawNCCD = sum(tePredRawNCCD == testLabel)./(length(testLabel));

%% GMM with PCA- Raw Data %%
[pca_vectors, train_pca_raw, ~, ~, explained] = pca(train);
test_pca_raw = (test-mean(test,2)) * pinv(pca_vectors);

PCidxs = [1 2 5 20];
figure;
plot(1:30, explained(1:30), '-', PCidxs, explained(PCidxs), '*','LineWidth', 2);
title('Fraction of variance of raw data explained by each principal component');
ylabel('%');
xlabel('Component index');

rng(1);
options = statset('MaxIter',1000);
GMModels_raw = cell(size(PCidxs));
teAccGMMRaw = zeros(size(PCidxs));
for j = 1:length(PCidxs)
    GMModels_raw{j} = fitgmdist(train_pca_raw, PCidxs(j),           ...
                                    'CovarianceType', 'diagonal',   ...
                                    'Options', options);
    teAccGMMRaw(j) = sum(pdf(GMModels_raw{j}, test_pca_raw) == testLabel)/length(testLabel);
    fprintf('GMM with %d components (raw data) accuracy: %.2f%%\n', PCidxs(j), 100*teAccGMMRaw(j));
end

%% GMM with PCA- DFT Data %%
[pca_vectors, train_pca_DFT, ~, ~, explained] = pca(trDFT);
test_pca_DFT = (teDFT-mean(teDFT,2)) * pinv(pca_vectors);

PCidxs = [1 2 5 15];
figure;
plot(1:30, explained(1:30), '-', PCidxs, explained(PCidxs), '*','LineWidth', 2);
title('Fraction of variance of DFT data explained by each principal component');
ylabel('%');
xlabel('Component index');

rng(1);
options = statset('MaxIter',1000);
GMModels_DFT = cell(size(PCidxs));
teAccGMMDFT = zeros(size(PCidxs));
for j = 1:length(PCidxs)
    GMModels_DFT{j} = fitgmdist(train_pca_DFT, PCidxs(j),           ...
                                    'CovarianceType', 'diagonal',   ...
                                    'RegularizationValue', .001,    ...
                                    'Options', options);
    teAccGMMDFT(j) = sum(pdf(GMModels_DFT{j}, test_pca_DFT) == testLabel)/length(testLabel);
    fprintf('GMM with %d components (DFT data) accuracy: %.2f%%\n', PCidxs(j), 100*teAccGMMDFT(j));
end

%% GMM with PCA- db4 Data %%
[pca_vectors, train_pca_db4, ~, ~, explained] = pca(trWav);
test_pca_db4 = (teWav-mean(teWav,2)) * pinv(pca_vectors);

PCidxs = [1 2 6 20];
figure;
plot(1:30, explained(1:30), '-', PCidxs, explained(PCidxs), '*','LineWidth', 2);
title('Fraction of variance of db4 DWT data explained by each principal component');
ylabel('%');
xlabel('Component index');

rng(1);
options = statset('MaxIter',1000);
GMModels_db4 = cell(size(PCidxs));
teAccGMMdb4 = zeros(size(PCidxs));
for j = 1:length(PCidxs)
    GMModels_db4{j} = fitgmdist(train_pca_db4, PCidxs(j),           ...
                                    'CovarianceType', 'diagonal',   ...
                                    'RegularizationValue', .001,    ...
                                    'Options', options);
    teAccGMMdb4(j) = sum(pdf(GMModels_db4{j}, test_pca_db4) == testLabel)/length(testLabel);
    fprintf('GMM with %d components (db4 DWT data) accuracy: %.2f%%\n', PCidxs(j), 100*teAccGMMdb4(j));
end

%% GMM with PCA- sym5 Data %%
[pca_vectors, train_pca_S5, ~, ~, explained] = pca(trWavS5);
test_pca_S5 = (teWavS5-mean(teWavS5,2)) * pinv(pca_vectors);

PCidxs = [1 2 6 20];
figure;
plot(1:30, explained(1:30), '-', PCidxs, explained(PCidxs), '*','LineWidth', 2);
title('Fraction of variance of sym5 DWT data explained by each principal component');
ylabel('%');
xlabel('Component index');

rng(1);
options = statset('MaxIter',1000);
GMModels_S5 = cell(size(PCidxs));
teAccGMMS5 = zeros(size(PCidxs));
for j = 1:length(PCidxs)
    GMModels_S5{j} = fitgmdist(train_pca_S5, PCidxs(j),           ...
                                    'CovarianceType', 'diagonal',   ...
                                    'Options', options);
    teAccGMMS5(j) = sum(pdf(GMModels_S5{j}, test_pca_S5) == testLabel)/length(testLabel);
    fprintf('GMM with %d components (sym5 DWT data) accuracy: %.2f%%\n', PCidxs(j), 100*teAccGMMS5(j));
end

%% Load SVM models %%
load('../Models/allGeneratedSVMModels.mat');

%% Linear SVM %%

t = templateSVM('KernelFunction', 'linear', 'SaveSupportVectors',true);
Mdl = fitcecoc(train, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test);
teAccSVM = sum(tePredsvm==testLabel)./length(testLabel); %~92%

yPredqSVM_dft = lSVMModel1.predictFcn(teDFTT); %Linear SVM, DFT
teAccSVMDFT = mean(yPredqSVM_dft==testLabel); %89.26%

MdlWav = fitcecoc(trWav, trainLabel, 'Learners',t);
tePredsvmWav = predict(MdlWav, teWav);
teAccSVMWav = sum(tePredsvmWav==testLabel)./length(testLabel); %91.88

MdlWavS5 = fitcecoc(trWavS5, trainLabel, 'Learners',t);
tePredsvmWavS5 = predict(MdlWavS5, teWavS5);
teAccSVMWavS5 = sum(tePredsvmWavS5==testLabel)./length(testLabel); %91.88

%% RBF SVM (using 15 principal components)%%
t = templateSVM('KernelFunction', 'rbf', 'SaveSupportVectors', true, 'Standardize', true);
nPCA = 15;

% Raw data
[pca_vectors, train_pca_raw] = pca(train);
test_pca_raw = (test-mean(test,2)) * pinv(pca_vectors);
train_pca_raw = train_pca_raw(:,1:nPCA);
test_pca_raw  = test_pca_raw(:,1:nPCA);

Mdl = fitcecoc(train_pca_raw, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_raw);
teAccSVM_RBF_raw = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('Raw Data SVM (first %d principal components) with RBF Kernel accuracy: %.2f%%\n',nPCA, 100*teAccSVM_RBF_raw);

% DFT data
[pca_vectors, train_pca_DFT] = pca(trDFT);
test_pca_DFT = (teDFT-mean(teDFT,2)) * pinv(pca_vectors);
train_pca_DFT = train_pca_DFT(:,1:nPCA);
test_pca_DFT  = test_pca_DFT(:,1:nPCA);

Mdl = fitcecoc(train_pca_DFT, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_DFT);
teAccSVM_RBF_DFT = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('DFT Data SVM (first %d principal components) with RBF Kernel accuracy: %.2f%%\n',nPCA, 100*teAccSVM_RBF_DFT);

% db4 DWT data
[pca_vectors, train_pca_db4] = pca(trWav);
test_pca_db4 = (teWav-mean(teWav,2)) * pinv(pca_vectors);
train_pca_db4 = train_pca_db4(:,1:nPCA);
test_pca_db4  = test_pca_db4(:,1:nPCA);

Mdl = fitcecoc(train_pca_db4, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_db4);
teAccSVM_RBF_db4 = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('db4 DWT Data SVM (first %d principal components) with RBF Kernel accuracy: %.2f%%\n',nPCA, 100*teAccSVM_RBF_db4);

% sym5 DWT data
[pca_vectors, train_pca_S5] = pca(trWavS5);
test_pca_S5 = (teWavS5-mean(teWavS5,2)) * pinv(pca_vectors);
train_pca_S5 = train_pca_S5(:,1:nPCA);
test_pca_S5  = test_pca_S5(:,1:nPCA);

Mdl = fitcecoc(train_pca_S5, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_S5);
teAccSVM_RBF_S5 = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('sym5 DWT Data SVM (first %d principal components) with RBF Kernel accuracy: %.2f%%\n',nPCA, 100*teAccSVM_RBF_S5);

%% Quadratic SVM %%
varNames = qSVMModel1.RequiredVariables;
varNames{188}= 'e185';
testT.Properties.VariableNames = varNames;
yPredqSVM= qSVMModel1.predictFcn(testT); %Quad SVM, normal
qSVMAcc = mean(yPredqSVM==testLabel); %97.08%

yPredqSVMDFT = qSVMModel5.predictFcn(teDFTT); %Quad SVM, DFT
qSVMAccDFT = mean(yPredqSVMDFT==testLabel); %93.79%

yPredqSVMS5 = qSVMModel2.predictFcn(teWavS5T); %Quad SVM, wavelet
qSVMAccS5 = mean(yPredqSVMS5==testLabel); %97.02%

%% Cubic SVM %%
yPredqSVMS5_2 = qSVMModel3.predictFcn(teWavS5T); %Cubic SVM, wavelet data
qSVMAccS5_2 = mean(yPredqSVMS5_2==testLabel); %97.73%

yPredqSVM_DFT = qSVMModel6.predictFcn(teDFTT); %Cubic SVM, DFT
qSVMAcc_DFT2 = mean(yPredqSVM_DFT==testLabel); %96.22%

yPredqSVM_2 = qSVMModel4.predictFcn(testT); %Cubic SVM, reg data
qSVMAcc_2 = mean(yPredqSVM_2==testLabel); %97.67%

%% Cubic SVM (Raw + DWT features) %%
t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'SaveSupportVectors', true);
nPCA = 15;

% db4
train_normWav = [normalize(train, 2), normalize(trWav, 2)];
test_normWav =  [normalize(test, 2),  normalize(teWav, 2)];

% Raw data
[pca_vectors, train_pca_normWav] = pca(train_normWav);
test_pca_normWav = (test_normWav-mean(test_normWav,2)) * pinv(pca_vectors);
train_pca_normWav = train_pca_normWav(:,1:nPCA);
test_pca_normWav  = test_pca_normWav(:,1:nPCA);

Mdl = fitcecoc(train_pca_normWav, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_normWav);
teAccSVM_normWav = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('Raw + db4 DWT Cubic SVM (first %d principal components) accuracy: %.2f%%\n',nPCA, 100*teAccSVM_normWav);

% sym5
train_normWavS5 = [normalize(train, 2), normalize(trWavS5, 2)];
test_normWavS5 =  [normalize(test, 2),  normalize(teWavS5, 2)];

% Raw data
[pca_vectors, train_pca_normWavS5] = pcca(train_normWavS5);
test_pca_normWavS5 = (test_normWavS5-mean(test_normWavS5,2)) * pinv(pca_vectors);
train_pca_normWavS5 = train_pca_normWavS5(:,1:nPCA);
test_pca_normWavS5  = test_pca_normWavS5(:,1:nPCA);

Mdl = fitcecoc(train_pca_normWavS5, trainLabel, 'Learners',t);
tePredsvm = predict(Mdl, test_pca_normWavS5);
teAccSVM_normWavS5 = sum(tePredsvm==testLabel)./length(testLabel);
fprintf('Raw + sym5 DWT Cubic SVM (first %d principal components) accuracy: %.2f%%\n',nPCA, 100*teAccSVM_normWavS5);


%% Naive Bayes %%
Mdl = fitcnb(normalize(train), trainLabel, 'Crossval', 'on');
modelLosses = kfoldLoss(Mdl,'mode','individual');
[~, k]=min(modelLosses);
tePredNB= predict(Mdl.Trained{k}, normalize(test));
teAccNB = sum(tePredNB==testLabel)./length(testLabel); %~92%

MdlWav = fitcnb(trWav, trainLabel, 'Crossval', 'on');
modelLosses = kfoldLoss(MdlWav,'mode','individual');
[~, k]=min(modelLosses);
tePredWavNB= predict(MdlWav.Trained{k}, normalize(teWav));
teAccNBWav = sum(tePredWavNB==testLabel)./length(testLabel); %91.88

MdlWavS5 = fitcnb(trWavS5, trainLabel, 'Crossval', 'on');
modelLosses = kfoldLoss(MdlWavS5,'mode','individual');
[~, k]=min(modelLosses);
tePredWavS5NB = predict(MdlWavS5.Trained{k}, normalize(teWavS5));
teAccNBWavS5 = sum(tePredWavS5NB==testLabel)./length(testLabel);

%% Ensemble classification %%
MdlEns = fitcensemble(trWavS5, trainLabel);
tePredEns = predict(MdlEns, teWavS5);
teAccEnsWav = sum(tePredEns==testLabel)./length(testLabel);

MdlEns2 = fitcensemble(trWav, trainLabel);
tePredEns2 = predict(MdlEns2, teWav);
teAccEnsWavDB4 = sum(tePredEns2==testLabel)./length(testLabel);

MdlEns3 = fitcensemble(trDFT, trainLabel);
tePredEns3 = predict(MdlEns3, teDFT);
teAccEnsDFT = sum(tePredEns3==testLabel)./length(testLabel);

MdlEns4 = fitcensemble(train, trainLabel);
tePredEns4 = predict(MdlEns4, test);
teAccEns = sum(tePredEns4==testLabel)./length(testLabel);

%% CNN Architecture %%
% See cnn_classification.m      