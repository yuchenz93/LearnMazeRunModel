function Sh = ShbyTM(T,step)
% ShbyTM generates trajectory based on transition matrix T
% experience
% inputs:T   transition matrix, indicates probability from row to column,
%            the sum of row should equal to 1
%       step length of the trajectory
% output:Sh  generated trajectory

% derive parameters
npos = size(T,1);
% make sure the transition matrix row are normalized
for ir = 1:npos
    T(ir,:) = T(ir,:)/sum(T(ir,:));
end


% generate sequence
Sh = nan(1,step);
% generate the first step, randomly select position
Sh(1) = randn(npos,1);
if step >= 2
   for is = 2:step
       rnum = Sh(is-1); 
       Tnow = T(rnum,:);
       % get CDF, randomly generate a number of [0,1], compared with CDF to
       % get next state
       CDFnow = cumsum(Tnow);
       CDFnow = cat(0,CDFnow);
       ridx = rand(1);
       Sh(is) = discretize(ridx,CDFnow);
   end
end

end