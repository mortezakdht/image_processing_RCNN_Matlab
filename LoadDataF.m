function [imdsTrain,imdsTest] = LoadDataF(NameDataSet)
 Address = ['Data&Image/',NameDataSet];
 imds = imageDatastore(Address,'IncludeSubfolders',true,...
        'LabelSource','foldernames');  
 [imdsTrain,imdsTest] = splitEachLabel(imds,0.7,0.3,'randomized');
end

