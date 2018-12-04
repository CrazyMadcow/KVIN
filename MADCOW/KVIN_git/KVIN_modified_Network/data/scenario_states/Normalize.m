TotalSize = 100;
addpath('./new')
for i = 1:11
    fname = strcat('Gen',num2str(i),'.mat');
    load(fname)
    X = X + 50;
    Y = Y + 50;
    X = X./TotalSize .* 16;
    Y = Y./TotalSize .* 16;
    save(strcat('new/',fname),'X','Y','gamma','acc','Nsam','sam','time')
end