% bldsDetector:image Datastore of image file name.
% imdsDetector:box Label Datastore of image. 
% Detector:is Detector.
% outpot:plot Average Precision and average Miss Rate
function results=EvaluateF(Detector, imdsDetector,bldsDetector)
 %% Create a table to store the results
  numImages = numel(imdsDetector.Files);
  results = table('Size',[numImages 2],...
  'VariableTypes',{'cell','cell'},...
  'VariableNames',{'bbox','score'});
  for i = 1:numImages
     Img = imread(imdsDetector.Files{i});
     [bbox, score,label]= detect(Detector, Img);
     results.bbox{i} = bbox;
     results.score{i} = score;
     results.label{i} = label;
  end
 %% Evaluate Precision
  [ap, recall, precision] = evaluateDetectionPrecision(results, bldsDetector);
  figure;
   plot(recall, precision);
   grid on
   title(sprintf('Average precision = %.1f', ap))

 %% Evaluate Miss Rate
   [am, fppi, missRate] = evaluateDetectionMissRate(results, bldsDetector);
   figure;
    loglog(fppi, missRate);
    grid on
    title(sprintf('Log Average Miss Rate = %.1f', am))
end






