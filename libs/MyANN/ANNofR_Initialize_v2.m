function net = ANNofR_Initialize_v2(tlen,npos,varargin)
% this function initialize the reward policy of neural network
% inputs:  tlen, npos are memory length and number of position bins
% optional inputs:  specify layers 
% we will set the network by randomly generate one sample and train the
% network based on the sample

args.layers = [tlen,2*tlen,2*tlen,6,npos];
args = parseArgs(varargin, args);

net = struct;
for ilayer = 1:length(args.layers) - 1
   net.W{ilayer} = (rand(args.layers(ilayer+1),args.layers(ilayer))-0.5)*10; 
   net.b{ilayer} = (rand(args.layers(ilayer+1),1)-0.5)*10; 
end



end
