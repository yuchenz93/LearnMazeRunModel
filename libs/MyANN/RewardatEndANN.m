function net = RewardatEndANN(nt,npos)
% this is a manually designed ANN which should be accurately solve the
% reward problem with rewards placed at track ends
% inputs   nt,  memory length, or length of input vector
%          npos, position bins, or length of output vector
% output   net,  ANN network with input -> hid1 -> hid2 -> hid3 -> output
%                thus we will have net.W{1} net.W{2} net.W{3} net.W{4} 
%                and net.b{1} net.b{2} net.b{3} net.b{4} as the mapping
%                across those layers
% ANN design:    hidden layer 1 has 2*nt nodes, 1:nt is an index
%                representing whether the animal is at track start at each
%                time bin, nt+1:2*nt is an index representing whether
%                animal is at track end at each time bin.
%                hidden layer 2 has 2*nt nodes, 1:nt is an idex
%                representing whether this time bin is the first when
%                animal reach track start, and nt+1:2*nt is an idex
%                representing whether this time bin is the first when
%                animal reach track end.
%                hidden layer 3 has 2 nodes, each represent the time when
%                animal first reach track start or track end
% the input and output of this ANN should be column vectors


%% predefine weight and constant matrix
w1 = zeros(2*nt,nt);
w2 = zeros(2*nt,2*nt);
w3 = zeros(2,2*nt);
w4 = zeros(npos,2);

b1 = zeros(2*nt,1);
b2 = zeros(2*nt,1);
b3 = zeros(2,1);
b4 = zeros(npos,1);

%% design the mapping from input to hidden layer 1
% determine if animal is at track start
for ibin = 1:nt
    w1(ibin,ibin) = -100;
    b1(ibin,1) = 150;
end

% determine if animal is at track end
for ibin = nt+1:1:2*nt
    w1(ibin,ibin-nt) = 100;
    b1(ibin,1) = -100*npos+50;
end

%% design the mapping from hidden layer 1 to hidden layer 2
% determine if this is the first time bin when animal at track start
% we will skip first time bin (see ActRewardPolicy.m)
for ibin = 2:nt
    if ibin-1 >= 2
        for jbin = 2:ibin-1
            w2(ibin,jbin) = -100;
        end
    end
    w2(ibin,ibin) = 50;
end

for ibin = nt+2:1:2*nt
    if ibin-1 >= nt+2
        for jbin = nt+2:1:ibin-1
            w2(ibin,jbin) = -100;
        end
    end
    w2(ibin,ibin) = 50;
end

for ibin = 1:2*nt
    b2(ibin,1) = -25;
end
%% design the mapping from hidden layer 2 to hidden layer 3
% we will map to mild slope of sigmoid, only map to negative parts so we
% can distinguish the case where layer 2 are all 0s, if layers are all 0s,
% it means it possible that it reach track end at nt+1 (which is later than 1:nt cases)
for ibin = 2:nt
    w3(1,ibin) = 2*(ibin-nt-1)/nt;
end

for ibin = nt+2:1:2*nt
    w3(2,ibin) = 2*(ibin-2*nt-1)/nt;
end

% check the difference of mapping onto sigmoid
testt = 1:1:nt;
sigtest = sigmoid(4*(testt-nt/2)/nt);
mindiff = min(diff(sigtest));
% we need to amplify this difference to large enough number
amp = 100/mindiff;


%% design the mapping from hidden layer 3 to output layer
% we will map to mild slope of sigmoid
w4(1,1) = amp;
w4(1,2) = -amp;

w4(npos,1) = -amp;
w4(npos,2) = amp;

for ibin = 1:npos
    b4(ibin,1) = -50;
end
b4(1,1) = 50; 
% this is because if both of them never reach ends, we will has both
% elements in layer 3 equals to 0.5 then we put reward at start

net.W = {w1,w2,w3,w4};
net.b = {b1,b2,b3,b4};



end