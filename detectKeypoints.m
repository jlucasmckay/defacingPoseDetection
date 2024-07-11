% detectKeypoints.m

% https://www.mathworks.com/help/deeplearning/ug/estimate-body-pose-using-deep-learning.html

clear all
close all
dataDir = "OpenPose"


trainedOpenPoseNet_url = "https://ssd.mathworks.com/supportfiles/"+ ...
    "vision/data/human-pose-estimation.zip";
downloadTrainedOpenPoseNet(trainedOpenPoseNet_url,dataDir)
unzip(fullfile(dataDir,"human-pose-estimation.zip"),dataDir);

modelfile = fullfile(dataDir,"human-pose-estimation.onnx");
net = importNetworkFromONNX(modelfile);

net = removeLayers(net,net.OutputNames);

inputSize = net.Layers(1).InputSize;
X = dlarray(rand(inputSize),"SSC");
net = initialize(net,X);


% im = imread("THREE-VIEW-V2_anonymized.png");
im = imread("THREE-VIEW-V2.png");
imshow(im)

netInput = im2single(im)-0.5;
netInput = netInput(:,:,[3 2 1]);
netInput = dlarray(netInput,"SSC");

[heatmaps,pafs] = predict(net,netInput);
heatmaps = extractdata(heatmaps);
montage(rescale(heatmaps),BackgroundColor="b",BorderSize=3)

heatmaps = heatmaps(:,:,1:end-1);

pafs = extractdata(pafs);

params = getBodyPoseParameters;

poses = getBodyPoses(heatmaps,pafs,params);

renderBodyPoses(im,poses,size(heatmaps,1),size(heatmaps,2),params);