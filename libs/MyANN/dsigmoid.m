function y = dsigmoid(x)
% this function return the derivative of sigmoid function, which is 
% sig*(1-sig)
% inputs can be scalar, vector, and matrix, derivatives are computed for
% each element

sig = 1./(1+exp(-x));
y = sig.*(1-sig);

end