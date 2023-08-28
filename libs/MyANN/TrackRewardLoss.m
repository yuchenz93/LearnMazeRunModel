function [alldLdy,pctloss] = TrackRewardLoss(x,y,t)
% this function returns the derivative of loss function of estimated reward
% inputs   x,   inputs, columns being realizations and rows with inputs DOF
%          y,   outputs, columns being realizations and rows with outputs DOF
%          t,   targets, columns being realizations and rows with targets DOF
% specifically, rows of x should be a vector of states, with the first
% element being the current state (linear position)
% rows of t should be actual reward versus linear position
% we will use a ANN to estimate reward y = ANN(x) with rows of y being
% estimated reward
% output   alldLdy, a matrix with row being outputs and columns being
%                   realizations, the value if derivative of loss function
%                   with respect to that output
%                   in the case, each column only have one non-zero value
%                   as only the output at one position bin defined by
%                   current position of that realization contribute to loss
%                   function
%          pctloss, loss function formalized by true result
nout = size(t,1);
nsample = size(t,2);

alldLdy = zeros(nout,nsample); 
binidx = false(size(y));
% realization loop
for isample = 1:nsample
    % find the current state, this is the position bin we will look at
    statenow = x(1,isample); 
    % the contribution to loss function is (y(statenow,isample)-t(statenow,isample))^2
    % the derivative is 2*(y(statenow,isample)-t(statenow,isample))
    alldLdy(statenow,isample) = 2*(y(statenow,isample)-t(statenow,isample));
    binidx(statenow,isample) = true;
end
lall = t(binidx) - y(binidx);
pctloss = norm(lall)./sqrt(norm(t(binidx))^2+norm(y(binidx))^2);


end