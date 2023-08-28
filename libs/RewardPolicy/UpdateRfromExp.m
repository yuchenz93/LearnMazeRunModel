function net = UpdateRfromExp(TAct,TR,netori)
% this function train the network based on experience and actual reward
% inputs: TAct, is a matrix with columns with realizations, rows being
%                history of behavior states
%         TR,   is a matrix with columns with realizations, rows being
%                reward across spatial bins for each sample,
%         netori,  network from previous session, we use this to initialize
%                  network
%
% the row and column in TAct and TR are arranged in this way to match the
% default Matlab fitting neural network inputs and targets
% output: net, updated neural network

x = TAct;
t = TR;

% Train the Network
[net,tr] = train(netori,x,t);

% % Test the Network
% y = net(x);
% e = gsubtract(t,y);
% performance = perform(net,t,y)
% 
% % Recalculate Training, Validation and Test Performance
% trainTargets = t .* tr.trainMask{1};
% valTargets = t .* tr.valMask{1};
% testTargets = t .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y)
% valPerformance = perform(net,valTargets,y)
% testPerformance = perform(net,testTargets,y)


end