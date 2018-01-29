%Seth Boren

%Numerical Integration
%Midpoint Composite Method

%Input an even integer for N
N = 2000 ;

% TO AVOID DIVISION BY ZERO, PROBLEM IS DIVIDED INTO LEFT AND RIGHT
%*******************************************************************
%Interval for x = [a,b]  for LEFT SIDE
a =   -5;
b =  -0.01;
% Build x column vector for each step h ON THE LEFT SIDE
h = (b-a) / (N + 2);
x = zeros(N+1,1);
for i = 1:(N+1)
    x(i,1) = a + (i*h);
end
%Calculate Area on Left Side
AREA_left = 0;
STEPS = (N/2) + 1;
for j = 1 : STEPS
    n    = ((j-1)*2) + 1;     %the 1 keeps n from equalling 0
    xn = x(n,1);
    %  Area equals function height times step width
    AREA_left = AREA_left + (2*h) * func_1(xn);
end
%******************************************************************
%Interval for x = [a,b]  for RIGHT SIDE
a =   0.01;
b =  5;
% Build x column vector for each step h ON THE RIGHT SIDE
h = (b-a) / (N + 2);
x = zeros(N+1,1);
for i = 1:(N+1)
    x(i,1) = a + (i*h);
end
%Calculate Area on Right Side
AREA_right = 0;
STEPS = (N/2) + 1;
for j = 1 : STEPS
    n    = ((j-1)*2) + 1;   %the 1 keeps n from equalling 0
    xn = x(n,1);
    %  Area equals function height times step width
    AREA_right = AREA_right + (2*h) * func_1(xn);
end
%*******************************************************************
% Add the Area on the left side to the Area on the right side
AREA = AREA_left + AREA_right;
AREA