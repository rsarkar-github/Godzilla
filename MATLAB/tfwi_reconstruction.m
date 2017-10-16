clc
clear all
close all

N1 = 250;
N2 = 501;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Background wave field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('real_reconstruction.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 50:50:500;
labels_X = 1:1:10;
marks_Y = 50:50:250;
labels_Y = 0.4:0.4:2.0;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',labels_Y)
print(h,'-depsc2','-painters','real-reconstruction.eps')
% close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Born wave field (Case 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('real_extBorn1_reconstruction.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 50:50:500;
labels_X = 1:1:10;
marks_Y = 50:50:250;
labels_Y = 0.4:0.4:2.0;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',labels_Y)
print(h,'-depsc2','-painters','real-extBorn1-reconstruction.eps')
% close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Born wave field (Case 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('real_extBorn2_reconstruction.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 50:50:500;
labels_X = 1:1:10;
marks_Y = 50:50:250;
labels_Y = 0.4:0.4:2.0;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',labels_Y)
print(h,'-depsc2','-painters','real-extBorn2-reconstruction.eps')
% close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Born wave field (Case 3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('real_extBorn3_reconstruction.bin');
real = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 50:50:500;
labels_X = 1:1:10;
marks_Y = 50:50:250;
labels_Y = 0.4:0.4:2.0;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',labels_Y)
print(h,'-depsc2','-painters','real-extBorn3-reconstruction.eps')
% close(h)