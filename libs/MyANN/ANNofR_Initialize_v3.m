function net = ANNofR_Initialize_v3(xlen,npos,varargin)
% this function initialize the reward policy of neural network
% inputs:  tlen, npos are memory length and number of position bins
% optional inputs:  specify layers 
% we will set the network by randomly generate one sample and train the
% network based on the sample

args.layers = [xlen,2*xlen,2*xlen,8,npos];
args = parseArgs(varargin, args);

net = struct;
for ilayer = 1:length(args.layers) - 1
    numin = args.layers(ilayer);
    numout = args.layers(ilayer+1);
    factor = 2*sqrt(6/(numin+numout));
    net.W{ilayer} = (rand(args.layers(ilayer+1),args.layers(ilayer))-0.5)*factor;
    net.b{ilayer} = (rand(args.layers(ilayer+1),1)-0.5)*factor;
end



end
