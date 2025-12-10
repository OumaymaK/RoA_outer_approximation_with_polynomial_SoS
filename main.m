clear all
close all


%% Data

% Number of evaluation points
Nd = 50;

% Set X bound
xb = 1;

% Circle distribution
% radius = 1;
% theta = linspace(-pi,pi,Nd);
% xi = radius*[cos(theta(1:end-1)); sin(theta(1:end-1))];
% xi = [[0;0],xi];

% Random distribution
xi = [[0;0],2*xb*rand(2,Nd-1)-xb];

% Dynamics (supposed unknown, only generates data)
M = 2.5; ro = 0.5;
f = @(x) [x(1).*(sqrt(sum(x.^2))-ro); x(2).*(sqrt(sum(x.^2))-ro)];
for i = 1:length(xi)
    fi(:,i) = f(xi(:,i));
end


%% Optimisation

% Degree of polynomials
d = 10;

% Variables
t = sdpvar(1);
x = sdpvar(2,1);
y = sdpvar(2,1);

% Constraint set X = [-xb xb]
gx1 = xb^2 - x(1)^2;
gx2 = xb^2 - x(2)^2;

% Target set X = [-xT xT] (T=1s)
xT = 0.1;
gxT = (xT^2 - x'*x);

% Uncertainty set F_D 
gy = 0; con = []; var = [];
tic
for i = 1:length(xi)
    [sy,cy] = polynomial([t;x;y],d-2);
    var = [var; cy];
    gy = gy + (-sum((y-fi(:,i)).^2) + (M^2)*sum((x-xi(:,i)).^2)) * sy;
    con = [con; sos(sy)];
    fprintf("i = %d\n",i)
end
toc

% Polynomials
[w,cw] = polynomial(x,d);
[v,cv] = polynomial([t;x],d);
[s1,c1] = polynomial([t;x;y],d-2);
[s2,c2] = polynomial([t;x;y],d-2);
[st,ct] = polynomial([t;x;y],d-2);
[s3,c3] = polynomial(x,d-2);
[s4,c4] = polynomial(x,d-2);
[s5,c5] = polynomial(x,d-2);
[s6,c6] = polynomial(x,d-2);
[s7,c7] = polynomial(x,d-2);
var = [var; cw; cv; c1; c2; ct; c3; c4; c5; c6; c7];

% Moments on X
l = moments(d,[-xb -xb; xb xb]);

% Operator L
Lv = jacobian(v,t) + jacobian(v,x)*y;

% Constraints (Note that the dynamics was scaled by T, so there T = 1)
con = [con; sos((-Lv - gx1*s1 - gx2*s2 - gy + t*(t-1)*st)); sos(s1); sos(s2); sos(st)];   % Lv(t,x) <= 0 on [0,T] x Gamma_D
con = [con; sos(w - gx1*s6 - gx2*s7); sos(s6); sos(s7)];                                  % w(x) >= 0 on X
con = [con; sos(w - replace(v,t,0) - 1 - gx1*s4 - gx2*s5); sos(s4); sos(s5)];             % w(x) >= v(0,x)+1 on X
con = [con; sos(replace(v,t,1) - gxT*s3); sos(s3)];                                       % v(T,x) >= 0 on X_T

% Solve
solvesos(con,cw'*l,sdpsettings('solver','Mosek'),var);

% Coefficients of w and v
cw = double(cw);
cv = double(cv);


%% Plots

% Dataset
figure
scatter(xi(1,:), xi(2,:), 'MarkerEdgeColor', [0.5 0.4470 0.7410])
hold on
quiver(xi(1,:), xi(2,:), fi(1,:), fi(2,:), Color=[0.8500 0.3250 0.0980])
grid on

% Surface plot of w(x)
X1 = sdpvar(1); X2 = sdpvar(1);
p = vectorize(sdisplay(cw'*monolist([X1;X2],d))); % w(x)
[X1,X2] = meshgrid(-xb:0.05:xb,-xb:0.05:xb);
figure
surf(X1,X2,eval(p))
title('Polynomial w(x)')
xlabel('x_1') ; ylabel('x_2'); legend('w(x)')

% Obtained ROA
figure
X1 = sdpvar(1); X2 = sdpvar(1);
p = vectorize(sdisplay(replace(cv'*monolist([t;X1;X2],d),t,0)+1)); % v(0,x)+1
[X1,X2] = meshgrid(-xb:0.05:xb,-xb:0.05:xb);
surf(X1,X2,eval(p))
figure
contour(X1,X2,eval(p), [1 1], '-b', 'linewidth',2); hold on

% True RoA
viscircles([0 0],ro);
xlabel('x_1'); ylabel('x_2');
legend('Outer', 'True')
grid
title('ROA')