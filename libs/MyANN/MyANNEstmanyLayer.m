function y = MyANNEstmanyLayer(net,x)
% MyANNEst this estimate outputs based on net and inputs, a sigmoid function
% was applied to each layer
% this might only support ANN with two layers as I am not sure about how
% Matlab organize net structure with more layers...

% inputs  net,  neural network with net.W{} being weight matrix across
%               layers, and net.b{} being constant vector across layers 
%               at each layer, it takes a column vector and produce a
%               column vector (column matrix)
%          x,   inputs, columns being realizations and rows with inputs DOF
% output   y,   outputs, columns being realizations and rows with outputs DOF

ynow = x;
for ilayer = 1:length(net.W)
    ynow = sigmoid(net.W{ilayer}*ynow + net.b{ilayer});  
end

y = ynow;

end
