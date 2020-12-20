setDir = fullfile('C:\Users\lucas\Documents\MATLAB\pervasive_computing\project\EmoRecog\ImagesUncropped');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @faceDetection;
imds = shuffle(imds);
[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomize');
bag = bagOfFeatures(trainingSet, 'VocabularySize', 1000, 'GridStep', [8 8]);
options = templateSVM('KernelFunction', 'gaussian'); % 'gaussian', 'linear', 'polynomial'
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag,'LearnerOptions',options);
confMatrix = evaluate(categoryClassifier,testSet);
mean(diag(confMatrix));
imageTaker(categoryClassifier);