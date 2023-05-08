clc;clear
close all
%% Apply network and Detection training command
TrainNetwork=0;   % 1:Train and save network 0:load network
TrainDetector=1;  % 1:Train and save Detector 0:load and use Detector 
data='stopSign';  % 'stopSign'  'vehicle'   
n=1;              % Photo number to Show Lable
perm =2;          % Photo number to test the Detector
mod='max';        % 'all' 'max'

%% Design and Train CNN Network
if TrainNetwork==1
 ListDataSet = {'cifar-10Images','stopSignImages'};
 NameDataSet = ListDataSet{1};
 [imdsTrain,imdsTest] = LoadDataF(NameDataSet);
 layersCNN = CreateLayersRCNNF(imdsTrain);
 InitialLearnRate=0.001; 
 MaxEpochs= 10;     %Number of training rounds
 MiniBatchSize=32;  %Size of the mini-batch to use for each training iteration
 Shuffle='once';    % 'never' 'every-epoch'
 obtionsCNN = TrainOptionsF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);
 network = trainNetwork(imdsTrain,layersCNN,obtionsCNN);
 % accuracy Network
  YPred = classify(network,imdsTest);
  YTest = imdsTest.Labels;
  accuracynetwork = sum(YPred == YTest)/numel(YTest);
 % save Network
  save('Network&Detector/networkFasterRCNN' ,'network','layersCNN','imdsTrain','imdsTest');
else
 % load Network 
 load('Network&Detector/networkFasterRCNN')
end

%% Load Images and Lables
switch data
 % load stopSign Data
 case 'stopSign'

  load('Data&Image/stopSignsTable')
  Address = fullfile('Data&Image');
  stopSignsTable.imageFilename = fullfile(Address, stopSignsTable.imageFilename);
  imdsDetector = imageDatastore(stopSignsTable.imageFilename);
  bldsDetector = boxLabelDatastore(stopSignsTable(:,2:end));
  ds = combine(imdsDetector, bldsDetector);
  % load vehicle Data
  case 'vehicle'

  load('Data&Image/vehicleTable')
  Address = fullfile('Data&Image');
  vehicleTable.imageFilename = fullfile(Address, vehicleTable.imageFilename);
  imdsDetector = imageDatastore(vehicleTable.imageFilename);
  bldsDetector = boxLabelDatastore(vehicleTable(:,2:end));
  ds = combine(imdsDetector, bldsDetector);
end

%% Display one of the images in the imdsDetector
ShowLableF(imdsDetector,bldsDetector,n)

%% Options for trayning Object Detections 
InitialLearnRate=0.001; 
MaxEpochs= 20;    %Number of training rounds
MiniBatchSize=1;  % for FastRCNN and FasterRCNN only 1 is Correct 
Shuffle='every-epoch';   % 'once' 'never' 'every-epoch'
optionsOD = TrainOptionsODF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Object Detections
if TrainDetector==1
 tic
 FasterRCNNDetector = trainFasterRCNNObjectDetector(vehicleTable, ...
        network, optionsOD, 'NegativeOverlapRange', [0 0.3] , ... 
        'PositiveOverlapRange', [0.5 1]);
 toc
 % save Detector
 save('Network&Detector/fasterrcnnODstopSigns' ,'FasterRCNNDetector','imdsDetector','bldsDetector')
 %save('Network&Detector/fasterrcnnODvehicle' ,'FasterRCNNDetector','imdsDetector','bldsDetector')
else
 % load Detector
 load('Network&Detector/fasterrcnnODstopSigns')
 %load('Network&Detector/fasterrcnnODvehicle')
end

%% Evaluate FasterRCNN Detector 
results=EvaluateF(FasterRCNNDetector,imdsDetector,bldsDetector);

%% Test FasterRCNN Detector with Image
TestDetectorF(imdsDetector,FasterRCNNDetector,perm,mod)