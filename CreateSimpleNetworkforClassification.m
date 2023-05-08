clc;clear
close all
%%  network training
TrainNetwork=0;  % 1:Train and save network 0:load network
perm =5;         % Photo number to test the Network
%% Load Image 
ListDataSet = {'MerchDataImage','DigitDatasetImage'};
NameDataSet = ListDataSet{1};
[imdsTrain,imdsTest] = LoadDataF(NameDataSet);
save('Data&Image/MerchDataTest&Train','imdsTrain','imdsTest')

load('Data&Image/MerchDataTest&Train')

%% Display some images
numImageTrain=numel(imdsTrain.Files);
figure;
n = randperm(numImageTrain,6); 
for i = 1:6
    subplot(2,3,i);
    imshow(imdsTrain.Files{n(i)});
end

%% Design CNN network layers
layersCNN = CreateLayersF(imdsTrain);

%% Options for Training Network 
InitialLearnRate=0.100; 
MaxEpochs= 4;  %Number of training rounds
MiniBatchSize=64;%Size of the mini-batch to use for each training iteration
Shuffle='once'; % 'never' 'every-epoch'
optionsCNN = TrainOptionsF(InitialLearnRate, ...
            MaxEpochs,MiniBatchSize,Shuffle);

%% Train Network
if TrainNetwork==1
network = trainNetwork(imdsTrain,layersCNN,optionsCNN);
 % save Network
  save('Network/networkMerchData' ,'network')
else
 % load Network   
 load('Network/networkMerchData')
end

%% accuracy Network
YPred = classify(network,imdsTest);
YTest = imdsTest.Labels;
accuracynetwork = sum(YPred == YTest)/numel(YTest);

%% Test network with Image
Imagetest = readimage(imdsTest,perm);
[label,scores] = classify(network,Imagetest);
figure
 imshow(Imagetest)
  title (string(label) + ", " + num2str(100*scores(label),3) + "%");