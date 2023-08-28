function T = RandomWalkerPolicy_withDis(Dis,gstd)
% this function generate a random walker policy based on number on distance
% matrix and the standard deviation of gaussian kernal
T = eye(size(Dis));

for ir = 1:size(Dis,1)
    % get distance to this current state from all other states
    Dnow = Dis(ir,:);
    % for each distance, find the probability based on Gaussian
    % distribution
    pb = normpdf(Dnow,0,gstd);
    % normalized the probability
    T(ir,:) = pb/sum(pb);
end

end