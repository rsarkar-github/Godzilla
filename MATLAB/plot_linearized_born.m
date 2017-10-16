clc
clear all
close all

N1 = 501;
N2 = 501;

file = fopen('real_diff.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('imag_diff.bin');
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
print(h,'-depsc2','-painters','real-diff.eps')
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
print(h,'-depsc2','-painters','imag-diff.eps')
close(h)

%//////////////////////////////////////////////////////////////////////////
file = fopen('real_born.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('imag_born.bin');
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
print(h,'-depsc2','-painters','real-born.eps')
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
print(h,'-depsc2','-painters','imag-born.eps')
close(h)

%//////////////////////////////////////////////////////////////////////////
file = fopen('real_diff.bin');
real1 = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('imag_diff.bin');
imag1 = fread(file, [N1 N2], 'double');
fclose(file);

real1 = real1';
imag1 = imag1';

real = real - real1;
imag = imag - imag1;

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
print(h,'-depsc2','-painters','real-born-diff.eps')
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
print(h,'-depsc2','-painters','imag-born-diff.eps')
close(h)
