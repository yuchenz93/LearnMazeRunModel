function net = ANNofR_Initialize(tlen,npos,varargin)
% this function initialize the reward policy of neural network
% inputs:  tlen, npos are memory length and number of position bins
% optional inputs:  hsize, hiddle layer size, adjust based on npos [50]
%                   tpct, percentage of train  [80]
% we will set the network by randomly generate one sample and train the
% network based on the sample

args.hsize = 100; % hidden layer size
args.tpct = 80; % training pct
args = parseArgs(varargin, args);

x = randi(npos,[tlen,5]);
t = rand(npos,5);

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = args.hsize; 
net = fitnet(hiddenLayerSize,trainFcn);
net = configure(net,x,t);
% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.trainParam.showWindow = 0;
% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = args.tpct/100;
tvpct = (100-args.tpct)/2;
net.divideParam.valRatio = tvpct/100;
net.divideParam.testRatio = tvpct/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,~] = train(net,x,t);




end
