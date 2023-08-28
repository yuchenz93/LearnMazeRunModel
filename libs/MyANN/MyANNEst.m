function y = MyANNEst(net,x)
% MyANNEst this estimate outputs based on net and inputs, a sigmoid function
% was applied to each layer
% this might only support ANN with two layers as I am not sure about how
% Matlab organize net structure with more layers...

% inputs  net,  neural network with default Matlab struct, should contain
%               net.IW{} with net.IW{1} being weight matrix projecting from
%               input layer to hidden layer (num hidden * num inputs);
%               net.LW{} with net.LW{2,1} being weight matrix projecting 
%               from hidden layer to output layer (num outputs * num hidden)
%               net.b with net.b{1} being constant vector of hidden layer (num hidden * 1);
%               net.b with net.b{2} being constant vector of output layer (num outputs * 1);    
%          x,   inputs, columns being realizations and rows with inputs DOF
% output   y,   outputs, columns being realizations and rows with outputs DOF

% get hidden layer
hd = net.UW{1}*x + net.b{1};
% sigmoid 
hd = 1./(1+exp(-hd));

% get output layer
y = net.LW{2,1}*hd + net.b{2};
y = 1./(1+exp(-y));

end
