% location where you want the data to be stored

% FOR TRAIN %
%folder_name = 'trainCWT/';
%data = train;
%labels = trainLabel;

% FOR TEST %
folder_name = 'testCWT/';
data = test;
labels = testLabel;

[~,signalLength] = size(data);

% create a filter
fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
r = size(data,1);

for ii = 1:size(data,1)
    % get coefficients after passing through the filter
    disp(ii);
    cfs = abs(fb.wt(data(ii,:)));
    % convert the coefficients into an image
    im = ind2rgb(im2uint8(rescale(cfs)),jet(128));
    % plot an image
    %     imagesc(im);
    % where you want file to be stored
    imgLoc = fullfile(folder_name);
    imFileName = strcat(char('test'),'_',num2str(ii),'.jpg');
    imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
end


