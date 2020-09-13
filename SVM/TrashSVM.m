% clear workspace
close all; clear;
% read data into dataset
trashDir='dataset-resized';
imds = imageDatastore(trashDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% spliting 85% of data to training and 15 % to test dataset
[train, test] = splitEachLabel(imds, 0.85, 'randomized');%splitEachLabel(imds, 0.85, 'randomized');
histogram(train.Labels)
%% OVersampling of data to improve the balance the classes
labels=train.Labels;
[G,classes] = findgroups(labels);
numObservations = splitapply(@numel,labels,G);
desiredNumObservationsPerClass = max(numObservations);
files = splitapply(@(x){randReplicateFiles(x,desiredNumObservationsPerClass)},train.Files,G);
files = vertcat(files{:});
labels=[];info=strfind(files,'\');
for i=1:numel(files)
    idx=info{i};
    dirName=files{i};
    targetStr=dirName(idx(end-1)+1:idx(end)-1);
    targetStr2=cellstr(targetStr);
    labels=[labels;categorical(targetStr2)];
end
train.Files = files;
train.Labels=labels;
labelCount_oversampled = countEachLabel(train);
histogram(train.Labels)

%% choicing  kernel size for HOG feature extraction
img = readimage(imds, 3);

% Extract HOG features and HOG visualization
[hog_32x32, vis32x32] = extractHOGFeatures(img,'CellSize',[32 32]);
[hog_64x64, vis64x64] = extractHOGFeatures(img,'CellSize',[64 64]);
[hog_128x128, vis128x128] = extractHOGFeatures(img,'CellSize',[128 128]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features 
subplot(2,3,4);  
plot(vis32x32); 
title({'CellSize = [32 32]'; ['Length = ' num2str(length(hog_32x32))]});

subplot(2,3,5);
plot(vis64x64); 
title({'CellSize = [64 64]'; ['Length = ' num2str(length(hog_64x64))]});

subplot(2,3,6);
plot(vis128x128); 
title({'CellSize = [128 128]'; ['Length = ' num2str(length(hog_128x128))]});

% 64 x 64 seem to capture more info of the images
cellSize = [64 64];%[128 128];
hogFeatureSize = length(hog_64x64);%length(hog_128x128);

% % Loop over the trainingSet  and extract HOG features from each image. A
% % similar procedure will be used to extract features from the testSet.
numImages = numel(imds.Files);
train_numImages=numel(train.Files);
test_numImages=numel(test.Files);
trainFeatures = zeros(train_numImages, hogFeatureSize, 'single');
testFeatures = zeros(test_numImages, hogFeatureSize, 'single');
for i = 1:train_numImages
    img = readimage(train, i);   
    trainFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
for i = 1:test_numImages
    img = readimage(test, i);   
    testFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
%% Creating SVM Classifier and predicting labels from test set
% Get labels for each image.
trainLabels = train.Labels;
Train_table=table(trainFeatures,trainLabels,'VariableNames',{'features','Classes'});
testLabels = test.Labels;
Test_table=table(testFeatures,testLabels,'VariableNames',{'features','Classes'});
% Classifer model
[trainedClassifier, validationAccuracy] = trainClassifierSVM(Train_table);
%Predicting
[yfit,scores]=trainedClassifier.predictFcn(Test_table);
% Displaying confusiong matrix 
figure;
plotconfusion(Test_table.Classes,yfit);

cm=confusionmat(Test_table.Classes,yfit,'order',{'cardboard','glass','metal','paper','plastic','trash'});
figure;
h=heatmap(cm./sum(cm,2),'CellLabelColor','none');colormap('hot');
h.XDisplayLabels = {'cardboard','glass','metal','paper','plastic','trash'};
h.YDisplayLabels = {'cardboard','glass','metal','paper','plastic','trash'};
caxis([0 1])
xlabel('Predicted Class');ylabel('True Class');




