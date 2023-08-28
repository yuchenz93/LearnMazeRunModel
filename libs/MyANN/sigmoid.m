function sig = sigmoid(x)
% this function return the sigmoid function, which is 
% inputs can be scalar, vector, and matrix, derivatives are computed for
% each element

sig = 1./(1+exp(-x));


end