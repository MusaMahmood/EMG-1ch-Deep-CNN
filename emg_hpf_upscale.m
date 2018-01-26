function [ Y ] = emg_hpf_upscale( X, scale_factor )
%Filters EMG Signal; HPF; 1.5Hz; Fs =250; Butterworth:
b = [0.963000502799344,-2.88900150839803,2.88900150839803,-0.963000502799344];
a = [1,-2.92460623545070,2.85202781859389,-0.927369968350164];
Y = single(zeros(128, 1));
if (length(X) == 128)
    Y = single(filtfilt(b, a, X).*scale_factor);
end


end

