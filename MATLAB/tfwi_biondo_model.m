clc
clear all
close all

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Vel
% file = fopen('DATA_Vel0-anom.H@','r','b');
% vel = fread(file,'float','b');
% fclose(file);
% 
% file = fopen('vel0.bin','w','l');
% fwrite(file, vel,'float','l');
% fclose(file);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % DVel
% file = fopen('DATA_DVel-VerySlow-anom.H@','r','b');
% dvel = fread(file,'float','b');
% fclose(file);
% 
% file = fopen('dvel.bin','w','l');
% fwrite(file, dvel,'float','l');
% fclose(file);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Vel Plus DVel
% file = fopen('DATA_VelDat-VerySlow-anom.H@','r','b');
% velplusdvel = fread(file,'float','b');
% fclose(file);
% 
% file = fopen('velplusdvel.bin','w','l');
% fwrite(file, velplusdvel,'float','l');
% fclose(file);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Extended Model
% file = fopen('DATADecon-One-Array-VerySlow-anom.H@','r','b');
% velextended = fread(file,'float','b');
% fclose(file);
% 
% file = fopen('velextended.bin','w','l');
% fwrite(file, velextended,'float','l');
% fclose(file);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Extended Model
N1 = 300;
N2 = 601;
N3 = 65;
dx = 0.01;
dz = 0.01;
dt = 0.01;

file = fopen('velextended.bin','r','l');
velextended = fread(file,'float','l');
velextended = reshape(velextended,N1,N2,N3);
fclose(file);

slice = velextended(:,:,33);
h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(jet);
imagesc(slice);
grid on;
caxis([-7, 7]);
cb = colorbar;
ylabel(cb,'Velocity [km/s]','FontSize',18)
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dz - 0.8;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Vel
file = fopen('vel0.bin','r','l');
vel0 = fread(file,'float','l');
vel0 = reshape(vel0,N1,N2);
fclose(file);

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(jet);
imagesc(vel0);
grid on;
caxis([0.5, 1]);
cb = colorbar;
ylabel(cb,'Velocity [km/s]','FontSize',18)
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dz - 0.8;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read Vel + DVel
file = fopen('velplusdvel.bin','r','l');
velplusdvel = fread(file,'float','l');
velplusdvel = reshape(velplusdvel,N1,N2);
fclose(file);

h = figure('units','normalized','outerposition',[0 0 1 1]);
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
colormap(jet);
imagesc(velplusdvel);
grid on;
caxis([0.5, 1]);
cb = colorbar;
ylabel(cb,'Velocity [km/s]','FontSize',18)
xlabel('X [km]','FontSize',18);
ylabel('Z [km]','FontSize',18);
marks_X = 1:floor(N2/10):N2;
labels_X = (marks_X - 1) * dx - 3;
marks_Y = 1:floor(N1/10):N1;
labels_Y = (marks_Y - 1) * dz - 0.8;
set(gca,'XTick',marks_X)
set(gca,'XTickLabel',labels_X)
set(gca,'YTick',marks_Y)
set(gca,'YTickLabel',sprintf('%3.1f|',labels_Y));