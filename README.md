# LearnMazeRunModel
This is a model to explore how rats learn to run on a maze to collects rewards. The Code is written in Matlab 2017b

To run the code, 
first add entire package to the path;
then open the lastest main file (for example, main_RW2DR_inst_v4.m), 
         select the desired reward policy or create your reward policy in ActRewardPolicy_Dynamic.m
         edit the OutPath where the figures will be saved
         modify parameters if necessary. For example to speed up you can set runtest to 4 as if we run the experiment on 4 individual animals
         run the code
At this step, the sitmulation is slow, you may let it run over night.


Theory,
We assume animal is familar with the maze but don't know where are rewards.
During the experiment, we (experimentor) will design a reward policy such as alternatively put reward at track ends to encourage animal to run unidirectionally on tracks. In this case, the actual reward policy can be determined based on the behavior and reward history. We will sepcify the reward policy in ActRewardPolicy_Dynamic.m and run the experiment based on that.

However, the actual reward policy is never explicitly known to animal. Animal will have a hypothesized reward policy, and based on this conjecture, animal can estimated reward based on behavior and reward history. Then, based on estimated reward, animal will adapt a action policy to maxmize the time discounted reward in the future. The total reward is a balance between the collected reward and the effort (high speed run) to collect the reward. Without a clear idea of where is the reward, animal will use a random walker action policy. With a high confidence of where is the reward, animal will run towards that reward position. 

Then hypothesized reward policy is learnt based on actual experience. In the model, the animal predicts reward based on finite length behavior history and reward history. These are inputs to a three layer ANN, and the lost function is defined as difference of the expected rewards and actual rewards at the vicinity of the animal. The animal doesn't know the actual rewards at all position bins but only near the animal, the discrepancy of the measured loss and actual loss is the main challenge of the problem. 

In the model, one critial parameter is animal's confidence level of its hypothesized reward policy. The confidence level is high if the measured loss is low, and the confidence level is updated during exploration. The confidence level will impact the action policy and how animal updates its hypothesized reward policy. The actual action policy is a mix of random walker and the optimal action policy based on estimated reward. If the confidence level is low, the random walker policy will make higer contribution. Also, when confidence level is low, when the ANN is retrained, parameters are more randomly initialized rather than use previously trained parameters.
