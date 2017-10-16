clc
clear all
close all

N1 = 500;
N2 = 601;
dx = 0.01;
dt = 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Background wave field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('tfwi_biondo_bkg.bin');
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
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dt;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));
print(h,'-depsc2','-painters','tfwi_biondo_bkg.eps')
% close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perturbed wave field
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('tfwi_biondo_pert.bin');
real1 = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real1)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real1);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dt;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));
print(h,'-depsc2','-painters','tfwi_biondo_pert.eps')
% close(h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Difference between perturbed and background wavefield
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
real2 = real1 - real;
lim = max(max(abs(real2)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real2);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dt;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));
print(h,'-depsc2','-painters','tfwi_biondo_diff.eps')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Born modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file = fopen('tfwi_biondo_extBorn.bin');
real3 = fread(file, [N1 N2], 'double');
fclose(file);

lim = max(max(abs(real3)))/2;

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(gray);
imagesc(real3);
grid on;
caxis([-lim, lim]);
colorbar;
xlabel('X [km]','FontSize',18);
ylabel('t [sec]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dt;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));
print(h,'-depsc2','-painters','tfwi_biondo_extBorn.eps')



