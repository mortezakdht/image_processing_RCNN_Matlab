% NameDataSet:It is the name of the data.
% imdsTest:Dataset for Testing.
% imdsValidation:Dataset for Validation.
% imdsTrain:Dataset for training.
% Address:The address is data.  
function [imdsTrain,imdsValidation,imdsTest] = LoadDataF(NameDataSet)
 Address = ['Data&Image/',NameDataSet];
 imds = imageDatastore(Address,'IncludeSubfolders',true,...
        'LabelSource','foldernames');  
 [imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.1,'randomized');
end