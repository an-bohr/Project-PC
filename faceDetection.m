function rightImg = faceDetection(inputImg)
%FACEDETECTION Function to detect a face and crop it so it can be used by
% our future algorithms to detect emotions
% Input: Image with a face
% Output: Cropped image of just the face with dimensions 256*256
faceDetector = vision.CascadeObjectDetector();  % Create the face detector object
faceDetector.MinSize = [100 100];
img = imread(fullfile(inputImg));               % Read the inputted image to a
                                                % to a temporary variable
bbox = step(faceDetector, img);                 % Create a box with 4 values that 
                                                % are around the face
croppedImg = imcrop(img, bbox);                 % Crop the image with the values
rightImg = imresize(croppedImg, [256 NaN]);
%rightImg = rgb2gray(rightImg);
end

