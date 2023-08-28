function [alldLdy,pctloss,maxabsloss] = VCNTy_TrackRewardLoss_Dis(x,y,t,Dis,nbin)
% this function returns the derivative of loss function of estimated reward
% inputs   x,   inputs, columns being realizations and rows with inputs DOF
%          y,   outputs, columns being realizations and rows with outputs DOF
%          t,   targets, columns being realizations and rows with targets DOF
%          nbin, several bins close to current location can be used to
%                estimated loss
%          Dis,  Distance matrix
% specifically, rows of x should be a vector of states, with the first
% element being the current state (linear position)
% rows of t should be actual reward versus linear position
% we will use a ANN to estimate reward y = ANN(x) with rows of y being
% estimated reward
% output   alldLdy, a matrix with row being outputs and columns being
%                   realizations, the value if derivative of loss function
%                   with respect to that output
%                   in the case, each column only have several non-zero value
%                   as only the outputs at position bins close to
%                   current position of that realization contribute to loss
%                   function
%          pctloss, loss function formalized by true result
if nargin < 5
    npos = size(Dis,1);
    nbinrel = round(npos/5);
    nbin = max([nbinrel,8]);
    nbin = min([nbin,3]);
end

nout = size(t,1);
nsample = size(t,2);

alldLdy = zeros(nout,nsample); 
binidx = false(size(y));
% realization loop
for isample = 1:nsample
    % find the current state, this is the position bin we will look at
    statenow = x(1,isample); 
    Disnow = Dis(statenow,:);
    binrange = find(Disnow<=nbin);
    for ibin = binrange
        % the contribution to loss function is (y(statenow,isample)-t(statenow,isample))^2
        % the derivative is 2*(y(statenow,isample)-t(statenow,isample))
        alldLdy(ibin,isample) = 2*(y(ibin,isample)-t(ibin,isample));
        binidx(ibin,isample) = true;
    end
end
actrwd = t(binidx); actrwdl = actrwd(:);
estrwd = y(binidx); estrwdl = estrwd(:);
lall = actrwdl-estrwdl;
pctloss = norm(lall)./sqrt(norm(actrwdl)^2+norm(estrwdl)^2);
if isinf(pctloss)
   pctloss = 0.5; % in case that never see any reward, might lead to inf
end
maxabsloss = max(abs(lall));
end