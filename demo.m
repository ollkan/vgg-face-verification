
%load face models
config.paths.net_path = 'data/vgg_face.mat';
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
convNet = lib.face_feats.convNet(config.paths.net_path);
faces_file = load('data/faces.mat');
faces = faces_file.faces;

%prepare the video player for the face detection
faceDetector = vision.CascadeObjectDetector();
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
cam = webcam();
videoPlayer = vision.VideoPlayer('Position', [0, 0 ,720, 500]);

runLoop = true;
numPts = 0;
faceFound = false;

%detect the face
while runLoop && ~faceFound

    % Get the next frame.
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);

    if numPts < 10
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            oldPoints = xyPoints;
            bboxPoints = bbox2points(bbox(1, :));
            bboxPolygon = reshape(bboxPoints', 1, []);
         end

    else
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        numPts = size(visiblePoints, 1);

        if numPts >= 10
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            bboxPoints = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            faceFound = true;
        end

    end
    step(videoPlayer, videoFrame);
    runLoop = isOpen(videoPlayer);
end

% Clean up.
clear cam;
release(faceDetector);
release(pointTracker);
release(videoPlayer);
videoPlayer.hide;


%if you want to use existing image
%videoFrame = imread();


%get image and compare the results
det = faceDet.detect(videoFrame);
crop = lib.face_proc.faceCrop.crop(videoFrame,det(1:4,1));
result = convNet.simpleNN(crop);


%find closest distance average
faces_size = numel(faces);
min_idx = -1;
min_avg = -1;
for i=1:faces_size
    images_size = numel(faces{i}.images);
    sum = 0;
    for j=1:images_size
        image = faces{i}.images{j};
        distance = pdist([result';image'],'cosine');
        sum = sum + distance;
    end
    average = sum / images_size;
    if min_avg < 0 || average < min_avg
        min_avg = average;
        min_idx = i;
    end
end

%get image description
identity = faces{min_idx}.desc;
if min_avg >= 0.6
    identity = 'not found'; 
end

%show results
imshow(crop);
str = [sprintf('Distance: %f, Identity: ', min_avg) identity];
title(str);
