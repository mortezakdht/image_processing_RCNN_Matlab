function [layers] = CreateLayersF(imdsTrain)
%% Find Size Input Layer and numclass
numberClass = numel(categories(imdsTrain.Labels));
I = readimage(imdsTrain,1);
InputImageSize = size(I);
%% create Layer
layers = [
    imageInputLayer(InputImageSize)

    convolution2dLayer(3,8,'Padding','same');
    batchNormalizationLayer;
    reluLayer;
    
  maxPooling2dLayer(2,'Stride',2);

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
  maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
  maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

  maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
            
  maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

  maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer 

  maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numberClass)
    softmaxLayer
    classificationLayer];