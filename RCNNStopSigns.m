clc;clear
close all
%% Apply network and Detection training command
TrainNetwork=0;  % 1:Train and save network 0:load network
TrainDetector=0; % 1:Train and save Detector 0:load and use Detector 
n=5;             % Photo number to Show Lable
perm =2;         % Photo number to test the Detector
mod='max'; % 'all' 'max'

%% Design and Train CNN Network
if TrainNetwork==1
 ListDataSet = {'cifar-10Images','stopSignImages'};
 NameDataSet = ListDataSet{1};
 [imdsTrain,imdsTest] = LoadDataF(NameDataSet);
 layersCNN = CreateLayersRCNNF(imdsTrain);
 InitialLearnRate=0.100; 
 MaxEpochs= 15;    %Number of training rounds
 MiniBatchSize=32; %Size of the mini-batch to use for each training iteration
 Shuffle='once';   % 'never' 'every-epoch'
 obtionsCNN = TrainOptionsF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);
 network = trainNetwork(imdsTrain,layersCNN,obtionsCNN);
 % accuracy Network
  YPred = classify(network,imdsTest);
  YTest = imdsTest.Labels;
  accuracynetwork = sum(YPred == YTest)/numel(YTest);
 % save Network
  save('Network&Detector/networkRCNN' ,'network','layersCNN','imdsTrain','imdsTest');
else
 % load Network 
 load('Network&Detector/networkRCNN')
end

%% Load Images and Lables
load('Data&Image/stopSignsTable')
Address = fullfile('Data&Image');
stopSignsTable.imageFilename = fullfile(Address, stopSignsTable.imageFilename);
imdsDetector = imageDatastore(stopSignsTable.imageFilename);
bldsDetector = boxLabelDatastore(stopSignsTable(:,2:end));
ds = combine(imdsDetector, bldsDetector);

%% Display one of the images in the imdsDetector
ShowLableF(imdsDetector,bldsDetector,n)

%% Options for trayning Object Detections 
InitialLearnRate=0.001; 
MaxEpochs= 12;  %Number of training rounds
MiniBatchSize=32;%Size of the mini-batch to use for each training iteration
Shuffle='every-epoch'; % 'once' 'never' 'every-epoch'
optionsOD = TrainOptionsODF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Object Detections
if TrainDetector==1
 RCNNDetector = trainRCNNObjectDetector(stopSignsTable, ...
        network, optionsOD, 'NegativeOverlapRange', [0 0.3] , ... 
        'PositiveOverlapRange', [0.7 1]);
 % save Detector
 save('Network&Detector/rcnnODstopSigns' ,'RCNNDetector','imdsDetector','bldsDetector')
else
 % load Detector
 load('Network&Detector/rcnnODstopSigns')
end

%% Evaluate RCNNDetector 
EvaluateF(RCNNDetector,imdsDetector,bldsDetector);

%% Test RCNNDetector with Image
TestDetectorF(imdsDetector,RCNNDetector,perm,mod)
