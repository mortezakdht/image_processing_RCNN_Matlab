function ShowLableF(imdsDetector,bldsDetector,n)
Image=imread(imdsDetector.Files{n});
bbox = bldsDetector.LabelData{n,1};
annotation = bldsDetector.LabelData{n,2};
Image = insertObjectAnnotation(Image, 'rectangle', bbox, ...
    annotation,'LineWidth',4,'Color','red','FontSize',40 );
figure
imshow(Image)
end

