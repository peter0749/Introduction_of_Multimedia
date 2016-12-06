clear all;
wavfile = input('wave file? ','s');
startT = input('start(sec): ');
endT = input('end(sec): ');
%Determine time interval
if(startT < 0)
    startT = 0;
end
if(endT < startT)
    endT = startT+1;
end
[wave b] = audioread(wavfile,[1,1]);%test sample rate
[wave b] = audioread(wavfile,[ floor(1+b*startT) , floor(1+b*endT) ]);%Select time interval
wave = wave(:,1);%only take channel 1
N = length(wave(:,1));%Number of samples
timeline = (1:N)/b;%Time length
f=fft(wave(:,1));%f = wave after FFT
f=fftshift(f);%Move center to 0
fabs=abs(f);%Compute magnitude
colormap hot%Choose 'hot' colormap
subplot(3,1,1);
image = plot(timeline,wave(:,1));%time(sec.) -- Amp plot
xlabel('Time(s)');
ylabel('Amp');
subplot(3,1,2);
spectrogram(wave(:,1),128,120,128,b,'yaxis');%Sepctrogram
subplot(3,1,3);
plot((1:length(fabs))/length(fabs)*b-(b/2+1),fabs);%Freq. -- Magnitude plot
grid on;
xlabel('Frequency');
ylabel('Magnitude');
imagepath = input('image path? ','s');
saveas(image,imagepath);
y = input('play?','s');
if(y == 'y' || y == 'Y')
    p=audioplayer(wave(:,1),b);%play wavefile
    play(p);
end
