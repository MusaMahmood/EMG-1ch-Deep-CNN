% % Raw Data (rescaled) Extraction:
clear; close all; clc;
V = 1
if V
    v = '\v\';
else
    v = '\';
end
Subject = ['csv\' v];
d = dir([Subject '*.csv']);
PLOT = 0;
Fs = 250;
select_chs = 1:1; c_end = max(select_chs) + 1;
start = 1; whop = 12; wlen = 128;
relevant_data = [];
Y_all = [];
[b, a] = butter(3, 1.5*2/Fs, 'high');
output_dir = ['output_dir\hpf_scaled_up_' num2str(wlen) v];
for i = 1:length(d)
    DATA{i} = csvread([Subject d(i).name]);
end
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread([Subject filename]);
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    file_data = zeros(length(wStart), length(select_chs), wlen);
    Y_file = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        Y_file(w) = str2double(filename(14));
        if mod(w,50) == 0
            disp(w)
        end
        selected_window = data(wStart(w):wEnd(w), :);
        for ch = 1:length(select_chs)
%             file_data(w, ch, :) = rescale_minmax(selected_window(:, ch)); % rescale on a per-channel basis
%             file_data(w, ch, :) = filtfilt(b,a,selected_window(:,ch)).*20;
            file_data(w, ch, :) = emg_hpf_upscale(selected_window(:,ch), 20);
            p2p(f*w) = peak2peak(squeeze(file_data(w, ch, :)));
        end
    end
    relevant_data = [relevant_data; file_data];
    Y_all = [Y_all; Y_file];
end
mkdir([output_dir]);
Y = Y_all;
f_n = [output_dir, filename(1:end-4), '_raw_wlen_' num2str(wlen) '.mat'];
save(f_n, 'relevant_data', 'Y');
clear Y relevant_data

function [] = plot_imagesc(F, P)
    imagesc(1:2, F, P); 
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)');
end