%% set parameters
clear all;
clc;
% folderpath = '.\train_icvl\Depth\';
% filepath = '.\train_icvl\labels.txt';
% frameNum = 331006;

% folderpath = '.\test_icvl\Depth\';
% filepath = '.\test_icvl\icvl_test_list.txt';
% frameNum = 702+894;

folderpath = "/home/mahdi/HVR/original_datasets/ICVL/Depth/";
filepath = '/home/mahdi/HVR/original_datasets/ICVL/test_seq_1and2.txt';
frameNum = 702+894;



save_dir = '/home/mahdi/HVR/git_repos/A2J/data/icvl/test_seq_1and2_mat';

fp = fopen(filepath);
fid = 1;

tline = fgetl(fp);
while fid <= frameNum
    
    splitted = strsplit(tline);
    img_name = splitted{1}
%    img_name = strsplit(img_name, '/');
%    img_name = img_name{2}
    if exist(strcat(folderpath,img_name), 'file')
        img = imread(strcat(folderpath,img_name));
       
        fp_save = fopen(strcat(folderpath,img_name(1:size(img_name,2)-3),'bin'),'r');
        fwrite(fp_save,permute(img,[2,1,3]),'float');
        fclose(fp_save);
        
        save(fullfile(save_dir, strcat(num2str(fid), '.mat')),'img');
        
        %delete(strcat(folderpath,img_name));
    end
    fgetl(fp);
    tline = fgetl(fp);
    fid = fid + 1;
end

fclose(fp);







