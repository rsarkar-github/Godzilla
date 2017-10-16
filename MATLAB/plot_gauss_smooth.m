clc
clear all
close all

N1 = 501;
N2 = 501;

file = fopen('gauss_smooth.bin');
smooth = fread(file, [N1 N2], 'double');
fclose(file);

file = fopen('gauss_pert_smooth.bin');
pert_smooth = fread(file, [N1 N2], 'double');
fclose(file);

smooth = smooth';
pert_smooth = pert_smooth';

lim = max(max(abs(smooth)));

h = figure();
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(flipud(smooth));
pbaspect([1 1 1])
axis square;
caxis([0, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks = 50:50:500;
labels = 1:1:10;
set(gca,'XTick',marks)
set(gca,'XTickLabel',labels)
set(gca,'YTick',marks)
set(gca,'YTickLabel',labels)
print(h,'-depsc2','-painters','gauss-smooth.eps')
close(h)

lim = max(max(abs(pert_smooth)));

h = figure();
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(flipud(pert_smooth));
pbaspect([1 1 1])
axis square;
caxis([0, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks = 50:50:500;
labels = 1:1:10;
set(gca,'XTick',marks)
set(gca,'XTickLabel',labels)
set(gca,'YTick',marks)
set(gca,'YTickLabel',labels)
print(h,'-depsc2','-painters','gauss-pert-smooth.eps')
% close(h)