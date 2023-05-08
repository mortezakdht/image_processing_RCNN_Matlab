clc;clear
close all
%% Load Pretrained Network
ListNetwork = {'Squeezenet','Alexnet','Googlenet',...
               'Resnet18','Resnet50','resnet101',...
               'Vgg16','vgg19','Inceptionv3',...
               'Inceptionresnetv2'};    
NameNetwork = ListNetwork{5};
[network,inputSize] = SelectNetworkF(NameNetwork);

%% Read and Resize Image
I = imread('peppers.png'); % llama.jpg peppers.png
figure
 imshow(I)
size(I);  %N.N
I = imresize(I,inputSize); % 
figure
 imshow(I)
classNames = network.Layers(end).Classes; % N.N
numClasses = numel(classNames);              % N.N
disp(classNames(randperm(numClasses,10)));   % N.N

%% Classify Image
[label,scores] = classify(network,I);
figure
 imshow(I)
  title (string(label) + ", " + num2str(100*scores(label),3) + "%");

%% Display Top Predictions (N.N)
[~,idx] = sort(scores,'descend'); %Descending column numbers based on cell data
idx = idx(5:-1:1); % five predicted 
classNamesTop = network.Layers(end).Classes(idx);
scoresTop = scores(idx); % five predicted scores  
figure
 barh(scoresTop)
  xlim([0 1])
  xlabel('\fontsize{14} \fontname{Times New Roman}Probability')
  yticklabels(classNamesTop)
  title('\fontsize{16} \fontname{Times New Roman} Top 5 Predictions')