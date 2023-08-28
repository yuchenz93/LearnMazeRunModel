function [alldLdy,pctloss] = FullTrackRewardLoss(x,y,t)
% this function returns the derivative of loss function of full track
% estimated reward
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
%          pctloss, loss function formalized by true result

alldLdy = 2*(y-t);
% this is the derivative of (y-t)^2 with respect to y
lall = t(:) - y(:);
pctloss = norm(lall)./sqrt(norm(t(:))^2+norm(y(:))^2);

end