function TestDetectorF(imdsDetector,Detector,perm,mod)
 Image=imread(imdsDetector.Files{perm});
 [bbox, score, label] = detect(Detector, Image);
 switch mod
    case 'all'
      if numel(score)>5
         [score, idx] = sort(score,'descend');
         bbox = bbox(idx, :);
         score=score(1:5, :);
         bbox=bbox(1:5, :);
      end
      detectedImage = insertObjectAnnotation(Image, ...
        'rectangle', bbox,score,'LineWidth',4, ...
        'Color','green','FontSize',40);

    case 'max'
      [score, idx] = max(score);
      score=score*100;
      bbox = bbox(idx, :);
      annotation = sprintf('%s,%.3f%%', label(idx), score);
      detectedImage = insertObjectAnnotation(Image, ...
        'rectangle', bbox,annotation,'LineWidth',4, ...
        'Color','green','FontSize',40);  

 end
figure
imshow(detectedImage)
end