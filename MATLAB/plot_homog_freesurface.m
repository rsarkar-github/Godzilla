clc
clear all
close all

N1 = 501;
N2 = 501;

file = fopen('real_homog_freesurface.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('imag_homog_freesurface.bin');
imag = fread(file, [N1 N2], 'double');
fclose(file);

real = real';
imag = imag';

lim = max(max(max(abs(real))), max(max(abs(imag))))/5;

h = figure();
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(flipud(real));
pbaspect([1 1 1])
axis square;
caxis([-lim, lim]);
colorbar;
%title('Real Component','FontSize',18);
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks = 50:50:500;
labels = 1:1:10;
set(gca,'XTick',marks)
set(gca,'XTickLabel',labels)
set(gca,'YTick',marks)
set(gca,'YTickLabel',labels)
print(h,'-depsc2','-painters','real-homog-freesurface.eps')
close(h)

h = figure();
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(flipud(imag));
pbaspect([1 1 1])
axis square;
caxis([-lim, lim]);
colorbar;
%title('Imaginary Component','FontSize',18);
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks = 50:50:500;
labels = 1:1:10;
set(gca,'XTick',marks)
set(gca,'XTickLabel',labels)
set(gca,'YTick',marks)
set(gca,'YTickLabel',labels)
print(h,'-depsc2','-painters','imag-homog-freesurface.eps')
close(h)
