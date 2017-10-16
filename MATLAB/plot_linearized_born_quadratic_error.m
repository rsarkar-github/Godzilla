clc
clear all
close all

percent = [1 2 3 4 5 6 7 8 9 10];

norm_error = [7.74261e-09 3.0523e-08 6.76972e-08 1.18656e-07 1.82823e-07 2.59651e-07 3.48622e-07 4.49244e-07 5.61052e-07 6.83601e-07] * 10^7;

h = figure();
iptsetpref('ImshowBorder','tight')
set(h,'Color','white')
plot(percent,norm_error,'-*r','LineWidth',2);
%title('Norm of error versus % perturbation','FontSize',18);
xlabel('% perturbation','FontSize',18);
ylabel('Norm of error (1e-7)','FontSize',18);
print(h,'-depsc2','-painters','norm-error.eps')
close(h)