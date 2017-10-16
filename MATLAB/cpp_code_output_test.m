clc
clear all
close all

N1 = 501;
N2 = 501;

file = fopen('real.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('imag.bin');
imag = fread(file, [N1 N2], 'double');
fclose(file);

real = real';
imag = imag';

lim = max(max(max(abs(real))), max(max(abs(imag))))/5;

h = figure(1);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(flipud(real));
pbaspect([1 1 1])
axis square;
caxis([-lim, lim]);
colorbar;
print(h,'-depsc2','-painters','real.eps')

% subplot(1,2,2);
% colormap(jet);
% imagesc(flipud(imag));
% pbaspect([1 1 1])
% axis square;
% caxis([-lim, lim]);
% colorbar;