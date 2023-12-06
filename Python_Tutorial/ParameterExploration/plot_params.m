
clear all; close all; clc
set(groot, 'defaulttextinterpreter', 'Latex')
set(groot, 'defaultAxesTickLabelInterpreter','Latex')
set(groot,'DefaultLegendInterpreter', 'Latex')
set(groot, 'DefaultLineLineWidth', 2)
set(groot, 'defaultAxesFontSize', 20)
set(groot, 'defaultLineMarkerSize', 10)

%% FTLE

figure;

epsilon = 0.1;
omega = 2*pi/10;
t = 20;

count = 1;

for A = [0.1 0.6 1.1]
    
    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,ftle_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=$\pi$/5, $t$=20'])
    
    oldcmap = colormap;
    colormap( flipud(oldcmap) );

    count = count + 3;

end


A = 0.1;
omega = 2*pi/10;

count = 2;

for epsilon = [0.1 0.6 1.1]

    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,ftle_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=$\pi$/5, $t$=20'])

    count = count + 3;

end

A = 0.1;
epsilon = 0.1;

count = 3;

for omega = [pi/5 3*pi/5 7*pi/5]

    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,ftle_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=',num2str(omega/(pi/5)), '$\pi$/5, $t$=20'])

    count = count + 3;

end


%% FTLE

figure;

epsilon = 0.1;
omega = 2*pi/10;
t = 20;

count = 1;

for A = [0.1 0.6 1.1]
    
    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,lavd_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=$\pi$/5, $t$=20'])

    count = count + 3;

end


A = 0.1;
omega = 2*pi/10;

count = 2;

for epsilon = [0.1 0.6 1.1]

    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,lavd_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=$\pi$/5, $t$=20'])

    count = count + 3;

end

A = 0.1;
epsilon = 0.1;

count = 3;

for omega = [pi/5 3*pi/5 7*pi/5]

    load(['fields_A',num2str(A),'_eps',num2str(epsilon),'_omega',num2str(round(omega,2)),'_t',num2str(t),'.mat'])

    subplot(3,3,count)
    h=pcolor(x,y,lavd_field);
    set(h, 'edgecolor', 'none')
    xlabel('$x$')
    ylabel('$y$')
    title(['$A$=',num2str(A),' $\epsilon$=', num2str(epsilon), ', $\omega$=',num2str(omega/(pi/5)), '$\pi$/5, $t$=20'])

    count = count + 3;

end

