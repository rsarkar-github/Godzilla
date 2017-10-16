clc
clear all
close all

n = 500;
peak_freq = 5;
dt = 0.01;
delay_time = 0.2;
dfreq = 1 / (n * dt);

ricker = zeros(n,1);
for i = 0:n-1
    ricker(i+1,1) = (1 - 2* pi^2 * peak_freq^2 * (i*dt - delay_time)^2) * exp(- (pi* peak_freq * (i*dt - delay_time))^2);
end

ricker_fft = ifft(ricker);
ricker_fft_wrapped = ricker_fft;
ricker_fft_wrapped(n/2+1:n,1) = ricker_fft(1:n/2,1);
ricker_fft_wrapped(1:n/2,1) = ricker_fft(n/2+1:n,1);

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
plot(ricker,'-','LineWidth',2,'MarkerEdgeColor','r','MarkerSize',2)
grid on
ylim([-1.0 1.5]);
xlabel('t [s]','FontSize',18);
ylabel('Amplitude','FontSize',18);
marks_X = floor(n/10):floor(n/10):n;
labels_X = (marks_X - 1) * dt;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
print(h,'-depsc2','-painters','ricker-wavelet.eps')
close(h)

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
plot(abs(ricker_fft_wrapped),'-','LineWidth',2,'MarkerEdgeColor','r','MarkerSize',2)
grid on
xlabel('Frequency [Hz]','FontSize',18);
ylabel('Amplitude','FontSize',18);
marks_X = floor(n/10):floor(n/10):n;
labels_X = (marks_X - 1) * dfreq - n * dfreq / 2;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',sprintf('%3.1f|',labels_X))
set(gca,'YTickLabel',sprintf('%3.3f|',str2num(get(gca,'YTickLabel'))'));
print(h,'-depsc2','-painters','ricker-spectra.eps')
close(h)