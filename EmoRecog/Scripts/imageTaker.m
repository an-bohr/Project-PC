function output = imageTaker(categoryClassifier)
fig = figure();
ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1   1]);
cam = webcam(1);
camsize = str2double( strsplit(cam.Resolution, 'x'));
im = image(zeros(camsize), 'Parent', ax);
preview(cam, im);     %pass the image object to preview!
axis image;
ButtonH=uicontrol('Parent',fig,'Style','pushbutton','String','Take image and predict','Units','normalized','Position',[0.0 0.0 1 0.2],'Visible','on');
ButtonH.Callback= {@evaluator, cam, categoryClassifier, ButtonH};
set(0, 'DefaultUIControlFontSize', 28);

function evaluator(src,event, cam, categoryClassifier, ButtonH)
    frame = snapshot(cam);
    faceImg =faceDetectionLive(frame);
    [labelIndex, score] = predict(categoryClassifier,faceImg);  % test it
    label = categoryClassifier.Labels(labelIndex)
    score
    set(ButtonH, 'String',sprintf('You are looking %s. Please press again to analyze your emotions.', label{:}));
end
end