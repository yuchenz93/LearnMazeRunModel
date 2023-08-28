function T = RandomWalkerPolicy(npos,gstd)
% this function generate a random walker policy based on number on position
% bins and gaussian smooth window length
T = eye(npos);
for ir = 1:npos
    T(ir,:) = smoothdata(T(ir,:),'gaussian',gstd);
    T(ir,:) = T(ir,:)/sum(T(ir,:));
end

end