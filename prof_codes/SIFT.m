Testfilename = 'fashion-mnist_test.csv';
MTest = readtable(Testfilename);
%size(M)
%array is label and rest 2-785 are pixel
%extract the label and image first
Testlabels = table2array(MTest(:, 1));
Testimages = table2array(MTest(:, 2:end)) ;
%since current image is 1D data, we need transform to 2D image
%from dataste descript and Size fucntion and see 784 pixel, 28*28 image
imageSize=[28,28];
TestnumSamples = size(Testimages, 1);
TestimageSet = permute(reshape(Testimages', [imageSize, TestnumSamples]), [2,1,3]);

% % for double check, check the figure
% figure;
% for i = 1:9 
%     subplot(3,3,i);
%     % Display the image
%     imPattern=  TestimageSet(:,:,i);
%     imPattern = uint8(imPattern);
%     imshow(imPattern);
%     title(['Label: ', num2str(Testlabels(i))]);
% end
% % the image is ok

% do the same for train set

Trianfilename = 'fashion-mnist_train.csv';
MTrain = readtable(Trianfilename);
Trainlabels = table2array(MTrain(:, 1));
Trainimages = table2array(MTrain(:, 2:end)) ;
TrainNumSamples = size(Trainimages, 1);
TrainimageSet = permute(reshape(Trainimages', [imageSize, TrainNumSamples]), [2,1,3]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Extract the second training image's SIFT features
% image = uint8(TrainimageSet(:,:,2));  % First image in the training set (or change the index)
% % Detect SIFT features
% points = detectSIFTFeatures(image);
% % Select the top 1 strongest keypoints
% strongest_points = points.selectStrongest(1);
% % Extract features from the selected keypoints
% [features, valid_points] = extractFeatures(image, strongest_points);
% disp(features);
% now feature is correct
%forget to conside the empty points case

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract SIFT features from all train& test images
train_num_images = size(TrainimageSet, 3);
test_num_images = size(TestimageSet, 3);
%create the equal number cell
train_sift_features = zeros(train_num_images, 128);
test_sift_features = zeros(test_num_images, 128);
% input the first SIFT feature to the cell
for i =1: train_num_images
    image = uint8(TrainimageSet(:,:,i));
    points = detectSIFTFeatures(image);
    % consider the empty point case 
    
    if ~isempty(points)
    % since zeros already initialize, we don't care 
    [features, ~] = extractFeatures(image, points);
    train_sift_features(i,:) = features(1,:);
    end
end 

%test_num_images

for i =1:test_num_images
    image = uint8(TestimageSet(:,:,i));
    points = detectSIFTFeatures(image);
    if ~isempty(points)
      [features,~] = extractFeatures(image, points);
      test_sift_features(i,:) = features(1,:);
    end     
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Trainlabels = reshape(Trainlabels, [], 1); 
Testlabels = reshape(Testlabels, [], 1);
Trainlabels = categorical(Trainlabels);
Testlabels = categorical(Testlabels);
 
% define MLP
layers = [     
    % forget to change from image input layer to feature input layer
    % i debug this tinny error for one day 
    featureInputLayer(128, 'Normalization', 'none')  % Input layer: 128 features per image
    fullyConnectedLayer(512)  % Fully connected layer with 512 neurons
    reluLayer                 % ReLU activation
    fullyConnectedLayer(128)  % Fully connected layer with 128 neurons
    reluLayer                 % ReLU activation
    fullyConnectedLayer(10)   % Output layer: 10 classes (for 10 fashion categories)
    softmaxLayer              % Softmax layer for classification probabilities
    classificationLayer       % Final classification layer
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 0.001);

net = trainNetwork(train_sift_features, Trainlabels, layers, options);

% Evaluate the trained network
PredTrain = classify(net, train_sift_features);
trainAccuracy = sum(PredTrain == Trainlabels) / numel(Trainlabels);
disp(['Training Accuracy: ', num2str(trainAccuracy * 100), '%']);

PredTest = classify(net, test_sift_features);
testAccuracy = sum(PredTest == Testlabels) / numel(Testlabels);
disp(['Testing Accuracy: ', num2str(testAccuracy * 100), '%']);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

