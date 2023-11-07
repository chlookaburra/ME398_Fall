close all; clear all; clc

load('/Users/chloe/Documents/Stanford/ME398_Fall/Python_Tutorial/saveExplore/fields_A0.6_eps0.1_omega0.63_t20.mat')

xMin = 0.2; xMax = 0.5;
yMin = 0.2; yMax = 0.5;

[xF, yF, F] = getFrame(lavd_field, x, y, xMin, xMax, yMin, yMax);

function [xF, yF, F] = getFrame(lavd_field, x, y, xMin, xMax, yMin, yMax)

    xind1 = find(x >= xMin); xind2 = find(x<= xMax);
    xind = intersect(xind1, xind2);
    yind1 = find(y >= yMin); yind2 = find(y <= yMax);
    yind = intersect(yind1,yind2);

    xF = x(inds);
    yF = y(inds);
    F = lavd_field(inds);

end