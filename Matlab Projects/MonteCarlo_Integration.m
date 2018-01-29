%Seth Boren

%Numerical Integration  
%Monte Carlo Method

% TO AVOID DIVISION BY ZERO, PROBLEM IS DIVIDED INTO LEFT AND RIGHT
%*******************************************************************
%Interval for x = [a,b]  for LEFT SIDE
a = -5;
b = -0.01;
%Input value for N
N = 2000  ;
%Column Vector x to store random values of x for Left Side
x = zeros(N,1);
%Randomly Generate x values between -5 and 0
x = (b-a).*rand(N,1) + a;
%Calculate Area on Left Side
AREA_left = 0;
%**************************************************************
%Monte Carlo Formula; Area = (Sum of all f(x))*(b-a)/N
%**************************************************************
for i = 1:N
   AREA_left = AREA_left +  (b-a)*func_1(x(i,1));
end
%Divide by N because Area = average f(x) times width from b to a
 AreaLeftAverage = AREA_left/N  ;
%*******************************************************************
%Interval for x = [a,b]  for Right SIDE
a = 0.01;
b = 5;
%Column Vector x to store random values of x for Right Side
x = zeros(N,1);
%Randomly Geenerate x values between 0 and 5
x = (b-a).*rand(N,1)+a;
%Calculate Area on Right Side
AREA_right = 0;
%**************************************************************
%Monte Carlo Formula; Area = (Sum of all f(x))*(b-a)/N
%**************************************************************
for i = 1:N
   AREA_right = AREA_right + (b-a)*func_1(x(i,1));
end
%Divide by N because Area = average f(x) times width from b to a
AreaRightAverage = AREA_right/N ;
%**************************************************************
% Add the Area on the left side to the Area on the right side
AREA = AreaLeftAverage + AreaRightAverage;
AREA