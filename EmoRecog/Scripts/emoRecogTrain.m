function [score] = emoRecogTrain(imgLocation)
% EMOTIONAL RECOGNITION BY SUPERHEROES
%   Pass parameter image.
setDir = fullfile('C:\Users\lucas\Documents\MATLAB\pervasive_computing\project\EmoRecog\ImagesUncropped');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @faceDetection;
imds = shuffle(imds);
[trainingSet,testSet] = splitEachLabel(imds,0.8,'randomize');
bag = bagOfFeatures(trainingSet, 'VocabularySize', 1000, 'GridStep', [8 8]);
options = templateSVM('KernelFunction', 'gaussian'); % 'gaussian', 'linear', 'polynomial'
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag,'LearnerOptions',options);
img = faceDetection(fullfile('C:\Users\lucas\Documents\MATLAB\pervasive_computing\project\EmoRecog\Images\Test\', imgLocation));
[labelIdx, score] = predict(categoryClassifier,img);  % test it
label = categoryClassifier.Labels(labelIdx)           % test it
imshow(img);
confMatrix = evaluate(categoryClassifier,testSet);
mean(diag(confMatrix));
end