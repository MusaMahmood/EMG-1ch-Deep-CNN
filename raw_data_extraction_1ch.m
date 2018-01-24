% % Raw Data (rescaled) Extraction:
clear; close all; clc;
% d = dir([pwd '\S*.csv']);
Subject = 'data_1ch\';
% d = dir([Subject 'EMG_1CH_2017.10.30_10.47.35_250Hz*.csv']);
d = dir([Subject 'EMG_1CH_2017.10.29_17.21.26_250Hz*.csv']);
output_dir = ['output_dir\raw\' Subject];
mkdir(output_dir); PLOT = 0;
Fs = 250;
select_chs = 1:1; c_end = max(select_chs) + 1;
start = 1; whop = 12; wlen = 256;
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread([Subject filename]);
    wStart = start:whop:(15000-wlen); wEnd = wStart + wlen - 1;
    relevant_data = zeros(length(wStart), length(select_chs), wlen);
    Y = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        selected_window = data(wStart(w):wEnd(w), :);
        if sum(selected_window(:, c_end) == selected_window(1, c_end)) == wlen
            CLASS = selected_window(1, c_end)
            Y(w) = selected_window(1, c_end);
            for ch = 1:length(select_chs)
                relevant_data(w, ch, :) = rescale_minmax(selected_window(:, ch)); % rescale on a per-channel basis
            end
%             if (PLOT)
%                 figure(1); plot(squeeze(relevant_data(w, ch, :)));
%                 rgb = input('Continue? \n');
%             end
        end
    end         
    mkdir([output_dir]);
    f_n = [output_dir, filename(1:end-4), '_nofilt_raw_resc_wlen' num2str(wlen) '.mat'];
    save(f_n, 'relevant_data', 'Y');
    clear Y
end

function [] = plot_imagesc(F, P)
    imagesc(1:2, F, P); 
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)');
end