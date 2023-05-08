% imdsTrain:Image Data Store Train
% InputImageSize:Input image size.
% layers:Network layers.
function [layers] = CreateLayersRCNNF(imdsTrain)
%% Find Size Input Layer and numclass
numberClass = numel(categories(imdsTrain.Labels));
I = readimage(imdsTrain,1);
InputImageSize = size(I);
%% create Layer
layers = [
    imageInputLayer(InputImageSize)

    convolution2dLayer(3,8,'Padding',1,'WeightsInitializer','narrow-normal');
    batchNormalizationLayer;
    reluLayer;
    
  maxPooling2dLayer(2,'Stride',2);

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
  maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1,'WeightsInitializer','narrow-normal')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numberClass,'WeightsInitializer','narrow-normal')
    softmaxLayer
    classificationLayer];