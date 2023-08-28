function xnext = NextbyTM(T,xnow)
% NextbyTM generates next state randomly based on current state and
% transition matrix
% inputs:T   transition matrix, indicates probability from row to column,
%            the sum of row should equal to 1
%        xnow current state
% output:xnext  next state

% derive parameters
npos = size(T,1);
% make sure the transition matrix row are normalized
for ir = 1:npos
    T(ir,:) = T(ir,:)/sum(T(ir,:));
end


Tnow = T(xnow,:);
% get CDF, randomly generate a number of [0,1], compared with CDF to
% get next state
CDFnow = cumsum(Tnow);
CDFnow = cat(2,0,CDFnow);
ridx = rand(1);

xnext = discretize(ridx,CDFnow);

end