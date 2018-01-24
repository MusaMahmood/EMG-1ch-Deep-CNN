% % PSD Extraction:
%% PSD Extraction:
clear; close all; clc;
Subject = 'csv\v\';
d = dir([Subject '*.csv']);
output_dir = ['output_dir\psd\' 'v\'];
mkdir(output_dir); PLOT = 1;
Fs = 250;
select_chs = 1:1; c_end = max(select_chs) + 1;
start = 1; whop = 12; wlen = 256;
relevant_data = [];
Y_all = [];
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread([Subject filename]);
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    P = zeros(length(wStart), length(select_chs), wlen/2);
    Y_file = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        Y_file(w) = str2double(filename(14));
        if mod(w,50) == 0
            disp(w)
        end
        selected_window = data(wStart(w):wEnd(w), :);
%             if wlen == 256
%                 temp_4 = tf_psd_rescale_w256(selected_window(:,select_chs));
%             elseif wlen == 384
%                 temp_4 = tf_psd_rescale_w384(selected_window(:,select_chs));
%             elseif wlen == 512
%                 temp_4 = tf_psd_rescale_w512(selected_window(:,select_chs));
%             end
        for ch = 1:length(select_chs)
            [P(w, ch, :), F] = welch_estimator_ORIG(selected_window(:,ch), Fs, hann(wlen)); %pass unfiltered
            P(w, ch, :) = rescale_minmax(P(w, ch, :)); % rescale on a per-channel basis
        end 
    end       
    relevant_data= [relevant_data; P];
    Y_all = [Y_all; Y_file];
end
mkdir([output_dir]);
Y = Y_all;
f_n = [output_dir, filename(1:end-4), '_psd_wlen_' num2str(wlen) '.mat'];
save(f_n, 'relevant_data', 'Y');
clear Y relevant_data

function [] = plot_imagesc(F, P)
    imagesc(1:2, F, P); 
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)');
end