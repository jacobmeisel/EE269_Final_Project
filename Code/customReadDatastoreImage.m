function data = customReadDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    data = imread(filename);
    data = imresize3(data,[32 32,3]);
end