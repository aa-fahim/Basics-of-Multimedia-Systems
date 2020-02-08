%*****************************************************************
% ELE725 Lab1: Sampling and Quantizaion (Audio)
% Author: Bethany Santos, Ashif Fahim
% Date: January 17, 2019
%*****************************************************************
close ALL
clear

% Variables
N = 4
audioFile = 'ELE725_lab1.wav'
pf = 1
Mu = 100

%% Audio File Properties

[y,Fs] = audioread(audioFile)
audio_info = audioinfo(audioFile)

% file size = BitsPerSample*SampeRate*NumChannels*Duration
audiofile_size = (audio_info.BitsPerSample * audio_info.SampleRate * audio_info.NumChannels * audio_info.Duration)/8

%% Sampling
% plot original audio in time domain
ych1 = y(:,1) % take only one channel from .wav file
t = 0:1/Fs:length(ych1)*(1/Fs)-(1/Fs)
plot(t,ych1)
grid MINOR
xlabel('Time(s)')
ylabel('Signal Amplitude')
title('ELE725lab1.wav Audio in Time Domain')

% plot original adio in frequency domain
Y = abs(fftshift(fft(ych1)))
figure
plot(Y)
grid MINOR
xlabel('Frequency(Hz)')
ylabel('Signal Amplitude')
title('ELE725lab1.wav Audio in Frequency Domain')

% plot audio in freq domain after LPF
yLPF = lowpass(ych1,0.1)
YLPF = abs(fftshift(fft(yLPF)))
figure
plot(YLPF)
grid MINOR
axis ([-0.3 0.3 0 200])
xlabel('Frequency(Hz)')
ylabel('Signal Amplitude')
title('ELE725lab1.wav Audio in Frequency Domain (FILTERED)')

% downsample by a factor of N
DownSample(audioFile,'downSampled.wav',N,pf)

% plot interpolated and fft audio in freq domain
YInterp = abs(fftshift(fft(interp(ych1,N))))
figure
plot(YInterp)
grid MINOR
xlabel('Frequency(Hz)')
ylabel('Signal Amplitude')
title('ELE725lab1.wav Audio: Interpolated in Frequency Domain')

%% Quantization
%% Questions 1 and 2
[y, Fs] = audioread('ELE725_lab1.wav');
t=linspace(0,length(y)/Fs, length(y));

% Quantization signal for N = 2
y_quantized = UniformQuant('ELE725_lab1.wav', 'ELE725_lab1_quantized(N=2).wav', 2);
figure
subplot 311
stem(t,y_quantized)
title('Quantization at N = 2')
xlabel('Seconds [s]'); ylabel('Amplitude');
axis([0 2.6 -0.22 0.22])
grid minor

% Quantization signal for N = 4
y_quantized1 = UniformQuant('ELE725_lab1.wav', 'ELE725_lab1_quantized(N=4).wav', 4);
subplot 312
stem(t, y_quantized1)
title('Quantization at N = 4')
xlabel('Seconds [s]'); ylabel('Amplitude');
axis([0 2.6 -0.3 0.4])
grid minor 

% Quantization signal for N = 8
y_quantized2 = UniformQuant('ELE725_lab1.wav', 'ELE725_lab1_quantized(N=8).wav', 8);
subplot 313
stem(t, y_quantized2)
title('Quantization at N = 8')
xlabel('Seconds [s]'); ylabel('Amplitude');
axis([0 2.6 -0.3 0.3])
grid minor
print('UniformQuantN2-4-8', '-dpng')

% Printing the MSE for N=2,4,8
immse(y_quantized, y(:,1))
immse(y_quantized1, y(:,1))
immse(y_quantized2, y(:,1))

% Comparing the sound quality of different quantized signals
pause(3)
sound(y_quantized, Fs)
pause(3)
sound(y_quantized1, Fs)
pause(3)
sound(y_quantized2, Fs)

%% Question 3
% Mu Law Quantization signal for N = 2
y_expanded = MulawQuant('ELE725_lab1.wav', 'ELE725_lab1_expanded(N=2).wav', 2, 100);
figure
subplot 311
stem(y_expanded)
title('Mu Law Quantization at N = 2')
xlabel('Seconds'); ylabel('Amplitude');
axis([0 120000 -0.13 0.13])
grid minor

% Mu Law Quantization signal for N = 4
y_expanded1 = MulawQuant('ELE725_lab1.wav', 'ELE725_lab1_expanded(N=4).wav', 4, 100);
subplot 312
stem(y_expanded1)
title('Mu Law Quantization at N = 4')
xlabel('Seconds'); ylabel('Amplitude');
grid minor
print('MuLawQuantN4', '-dpng')

% Mu Law Quantization signal for N = 8
y_expanded2 = MulawQuant('ELE725_lab1.wav', 'ELE725_lab1_expanded(N=8).wav', 8, 100);
subplot 313
stem(y_expanded2)
title('Mu Law Quantization at N = 8')
xlabel('Seconds'); ylabel('Amplitude');
grid minor
print('MuLawQuantN2-4-8', '-dpng')

immse(y_expanded, y(:,1))
immse(y_expanded1, y(:,1))
immse(y_expanded2, y(:,1))

%% Question 4
% Plotting 200 samples of original, quantized and mu law quantized signals
figure
subplot 311
stem(y(:,1))
title('Original Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
subplot 312
stem(y_quantized)
title('Quantized Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
subplot 313
stem(y_expanded)
title('Mu Law Quantized Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
print('200samples', '-dpng')

% Plotting the same graph as above but overlapped
figure
stem(y(:,1), 'DisplayName', 'Original')
hold on
title('Original Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
stem(y_quantized, 'DisplayName', 'Quantized')
title('Quantized Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
stem(y_expanded, 'DisplayName', 'Mu Law Quantized')
hold off
title('Mu Law Quantized Signal')
axis([19000 19200 -0.2 0.2])
ylabel('Amplitude'); xlabel('Samples');
legend('Location', 'southwest')
print('200samples-overlapping', '-dpng')

% Finding the MSE for quantized signals under 0.15
custom_mse = MeanSquaredError(y_quantized, y, 0.15, "low")
custom_mse1 = MeanSquaredError(y_quantized1, y, 0.15, 'low')
custom_mse2 = MeanSquaredError(y_quantized2, y, 0.15, 'low')

% Finding the MSE for mu law quantized signal under 0.15
custom_mse3 = MeanSquaredError(y_expanded, y, 0.15, 'low')
custom_mse4 = MeanSquaredError(y_expanded1, y, 0.15, 'low')
custom_mse5 = MeanSquaredError(y_expanded2, y, 0.15, 'low')

% Finding the MSE for quantized signals over 0.15
custom_mse6 = MeanSquaredError(y_quantized, y, 0.15, 'high')
custom_mse7 = MeanSquaredError(y_quantized1, y, 0.15, 'high')
custom_mse8 = MeanSquaredError(y_quantized2, y, 0.15, 'high')

% Finding the MSE for mu law quantized signal over 0.15
custom_mse9 = MeanSquaredError(y_expanded, y, 0.15, 'high')
custom_mse10 = MeanSquaredError(y_expanded1, y, 0.15, 'high')
custom_mse11 = MeanSquaredError(y_expanded2, y, 0.15, 'high')

%% Functions
function DownSample(inFile, outFile, N, pf)
% DownSample function will take in an audio file, downsamples it, plays it
% back and saves the downsampled file as output.
% inFile -> audio file to be downsampled
% outFile -> Filename of the downsample audio to be saved
% N -> downsampling factor
% pf -> boolean flag indicating whether a pre filter should be used.
% *************************************************************************

    [y,fs] = audioread(inFile)
    ych1 = y(:,1)
    ych2 = y(:,2)

    if (pf == 1)
        ych1 = lowpass(ych1,.01)
    end

    yDownSampled = decimate(ych1,N)
    YDownSampled = abs(fftshift(fft(yDownSampled)))
    figure
    plot(YDownSampled)
    grid MINOR
    xlabel('Frequency(Hz)')
    ylabel('Signal Amplitude')
    title('ELE725lab1.wav Audio Signal Down Sampled')

    sound(y,fs)
    pause(2.5)
    sound(yDownSampled,fs/N)

    audiowrite(outFile,yDownSampled,round(fs/N))
end

function y_quantized = UniformQuant(inFile, outFile, N)
    
    % Reading input file
    [y, Fs] = audioread('ELE725_lab1.wav');
    
    % Finding the range and max of the signal
    y_max = max(y(:));
    y_min = min(y(:));
    
    % Finding the step size
    step_size = (y_max - y_min)/2^N
    
    % Quantizing the signal according to step size
    for i = 1:length(y)
        y_quantized(i,1) = (floor(y(i,1)/step_size) + 0.5) * step_size;
        %y_quantized(i,2) = (floor(y(i,2)/step_size) + 0.5) * step_size;
    end
    
    % Writing to output file
    audiowrite(outFile, y_quantized, Fs)
end

function y_expanded = MulawQuant(inFile, outFile, N, Mu)

    % Reading input file
    [y, Fs] = audioread('ELE725_lab1.wav');
    
    % Finding the range and max of the signal
    y_max = max(abs(y(:)));
    
    y_compressed = sign(y) .*log(1+(Mu*abs(y))) ./log(1+Mu);
    
    %Quantization Algorithm from UniformQuant()
    y_max_c = max(y_compressed(:));
    y_min_c = min(y_compressed(:));
    
    step_size = (y_max_c - y_min_c)/2^N 
    for i = 1:length(y)
        y_quantized(i,1) = (floor(y_compressed(i,1)/step_size) + 0.5) * step_size;
        %y_quantized(i,2) = (floor(y_compressed(i,2)/step_size) + 0.5) * step_size;
    end
    
    % Expanding quantized values
    y_max_q = max(abs(y_quantized(:)));
    y_expanded = sign(y_quantized) .*((1+Mu.^(abs(y_quantized)))-1) ./Mu;
    
    % Writing to output file
    audiowrite(outFile, y_expanded, Fs)
    
end

function custom_mse = MeanSquaredError(y_quan, y_original, amplitude, type)
    mse = 0;
    
    if (type == "low")
        for i = 1:length(y_quan)
            if (abs(y_original(i)) < amplitude)
                mse = mse + (y_quan(i) - y_original(i))^2;
            end
        end
    else
        for i = 1:length(y_quan)
            if (abs(y_original(i)) > amplitude)
                mse = mse + (y_quan(i) - y_original(i))^2;
            end
        end 
    end
    
    custom_mse = mse * (1/length(y_quan));
end