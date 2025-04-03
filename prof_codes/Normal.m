%% Load and Preprocess Data
Testfilename = 'fashion-mnist_test.csv';
MTest = readtable(Testfilename);
Testlabels = table2array(MTest(:, 1));
Testimages = table2array(MTest(:, 2:end)); 

Test = reshape(Testimages', 28, 28, 10000);
% Swap dimensions to ensure correct orientation
TestSet = permute(Test, [2, 1, 3]);


% Load training set
Trainfilename = 'fashion-mnist_train.csv';
MTrain = readtable(Trainfilename);
Trainlabels = table2array(MTrain(:, 1));
Trainimages = table2array(MTrain(:, 2:end)); 
% Transform 1D images into 2D (28x28)
Train = reshape(Trainimages', 28, 28, 60000);
% Swap dimensions to ensure correct orientation
TrainSet = permute(Train, [2, 1, 3]);


%this part is what i make wrong and why i cannot fit in the input layer
TrainSet = reshape(TrainSet,28,28,1,[]);
TestSet = reshape(TestSet,28,28,1,[]);


%% Ensure Labels Are Compatible
Trainlabels = reshape(Trainlabels, [], 1);  % Ensure column vector
Testlabels = reshape(Testlabels, [], 1);    % Ensure column vector

Trainlabels = categorical(Trainlabels);  % Convert labels to categorical
Testlabels = categorical(Testlabels);

%% Define MLP Network
layers = [
    imageInputLayer([28,28,1], 'Normalization', 'none')  % Feature input layer
    fullyConnectedLayer(512)  % Fully connected layer with 512 neurons
    reluLayer                 % ReLU activation
    fullyConnectedLayer(128)  % Fully connected layer with 128 neurons
    reluLayer                 % ReLU activation
    fullyConnectedLayer(10)   % Output layer: 10 classes
    softmaxLayer              % Softmax layer for classification probabilities
    classificationLayer       % Final classification layer
];

%% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 128, ...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 0.001);

%% Train the Network
net = trainNetwork(TrainSet, Trainlabels, layers, options);

%% Evaluate the Model
PredTrain = classify(net, TrainSet);
trainAccuracy = sum(PredTrain == Trainlabels) / numel(Trainlabels);
disp(['Training Accuracy: ', num2str(trainAccuracy * 100), '%']);

PredTest = classify(net, TestSet);
testAccuracy = sum(PredTest == Testlabels) / numel(Testlabels);
disp(['Testing Accuracy: ', num2str(testAccuracy * 100), '%']);
