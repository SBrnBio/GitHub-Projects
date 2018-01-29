%Seth Boren

%Nunerical Integration
%Romberg Integration Method (Starts by using Trapezoidal Composite)

%Input Number N
N = 512;

% TO AVOID DIVISION BY ZERO, PROBLEM IS DIVIDED INTO LEFT AND RIGHT
%*******************************************************************
%Interval for x = [a,b]  for LEFT SIDE
a = -5;
b =  -0.01;
% Build x column vector for each step h ON THE LEFT SIDE
h = (b-a) / (N);
x = zeros(N+1,1);
x(1,1) = a;
for i = 2:(N+1)
    x(i,1) =  h*(i-1);
end
%******************************************************************
%Calculate Area on Left Side
%Area of the 1st Half Space at x0
AREA_left = (1/2)*h*func_1(x(1,1));
% Area of Left Side Calculated
for i = 1 :(N+1)
   AREA_left = AREA_left + h*func_1(x(i,1));
end
%Add Area of last half space at xn
AREA_left = AREA_left + (1/2)*h*func_1(x(N+1,1));
%*******************************************************************
%Interval for x = [a,b]  for RIGHT SIDE
a = 0.01;
b = 5;
% Build x column vector for each step h ON THE RIGHT SIDE
h = (b-a) / (N);
x = zeros(N+1,1);
x(1,1) = a;
for i = 2:(N+1)
    x(i,1) =  h*(i-1);
end
%Calculate Area on RIGHT Side
%Area of the 1st Half Space at x0
AREA_right = (1/2)*h*func_1(x(1,1));
% Area of RIGHT Side Calculated
for i = 1 :(N+1)
   AREA_right = AREA_right + h*func_1(x(i,1));
end
%Add Area of last half space to xn
AREA_right = AREA_right + (1/2)*h*func_1(x(N+1,1));
%**************************************************************
% Add the Area on the left side to the Area on the right side
AREA = AREA_left+AREA_right;
AREA