clear all;
clc;
path = 'D:\NCSU\ECE 765\Project02\NEW FINAL\test\water_test_mat\test_output\';
mat = dir('*.mat');
for q = 1:length(mat)
    load(mat(q).name)
    s = mat(q).name;
    s = strrep(s,'ProbMap','sample')
    save(strcat(path,s),'label_im','prob_map')
end    