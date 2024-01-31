% Implementaci\'on del esquema GFDM para resolver el problema de
% Elder, el cual es un 'density-driven groundwater flows problem'
%% Las ecuaciones gobernantes son
% D2*Psi - Ra*Dx*C = 0
% D2*C - Dy*Psi*Dx*C + Dx*Psi*Dy*C = dC/dt
% donde
% D2 es una matriz de Laplaciano Discreto
% Dx es una matriz de diferenciaci\'on en x
% Dy es una matriz de diferenciaci\'on en y
%% Nombres de las figuras
clear all; close all; clc
% nombre = 'm81-41-t1239-';
% path = ['figuras/',nombre]; % carpeta
%% Par\'ametros
m = 5; % divisiones en el eje x (multiplo de 4 mas 1)
m2 = 4; % divisiones en el eje y
mm = m*m2;
H = 1;
H_L = 0.25; % H/L
L = H/H_L;
theta = 0;
Ra = 400; % N\'umero de Rayleigh
tfin = 1.239;
f = @(x,y) 0;
%% Discretizaci\'on del dominio
[x,y,tri,B,ba,bb,bc,bd] = malla_estructurada2(H,L,theta,m,m2,0);

% % % modificación de la malla en las singularidades
pot = 1.5;
% % % 0<x<1
for i = 2:(m-1)/4   % primera cuarta parte de la malla
    x(:,i) = 1-abs((x(1,i)-1)).^pot;
end
% 1<x<2
for i = (m-1)/4+2:(m-1)/2 % segunda cuarta parte
    x(:,i) = 1+abs((x(1,i)-1)).^pot;
end
% 2<x<3
for i = (m-1)/2+2:3*(m-1)/4 % tercera cuarta parte
    x(:,i) = 3-abs((x(1,i)-3)).^pot;
end
% 3<x<4
for i = 3*(m-1)/4+2:m-1 % cuarta \'ultima parte
    x(:,i) = 3+abs((x(1,i)-3)).^pot;
end

%% Condiciones de frontera
% Funci\'on de flujo Psi - Dirichlet
a_psi = @(y) 0;        % Izquierda
b_psi = @(y) 0;        % Derecha
c_psi = @(x) 0;        % Inferior
d_psi = @(x) 0;        % Superior
% Temperatura - Robin Superior e Inferior
a_T = @(y) 0;        % Izquierda
b_T = @(y) 0;        % Derecha
c_T = @(x) 0;        % Inferior
d_T = @(x) (x>=1).*(x<=3);        % Superior
% coeficientes condici\'on de Robin
aizq = @(y) 0;
bizq = @(y) 1;
ader = @(y) 0;
bder = @(y) 1;
ainf = @(x) 1;
binf = @(x) 0;
asup = @(x) (x>1).*(x<3);
bsup = @(x) (x<=1)+(x>=3);

%% Componentes del vector normal
theta=theta*pi/180;                     % Conversi\'on a radianes

%% Matrices de diferenciaci\'on para la funci\'on de flujo \Psi
    % Laplaciano
L2psi = [0 0 0 2 2 0]';
[D2psi,F2psi] = GFD_Dir2(x,y,L2psi,a_psi,b_psi,c_psi,d_psi,f,m,m2,ba,bb,bc,bd,B);
    % Diferencia en x
Lxpsi = [0 1 0 0 0 0]';
[Dxpsi,Fxpsi] = GFD_Dir2(x,y,Lxpsi,a_psi,b_psi,c_psi,d_psi,f,m,m2,ba,bb,bc,bd,B);
    % Diferencia en y
Lypsi = [0 0 1 0 0 0]';
[Dypsi,Fypsi] = GFD_Dir2(x,y,Lypsi,a_psi,b_psi,c_psi,d_psi,f,m,m2,ba,bb,bc,bd,B);

%% Matrices de diferenciaci\'on para la temperatura
    % Laplaciano
L2C = [0 0 0 2 2 0]';
[D2C,F2C] = GFD_RobM2(x,y,L2C,a_T,b_T,c_T,d_T,f,m,m2,ba,bb,bc,bd,B,theta,ainf,binf,asup,bsup,aizq,bizq,ader,bder);
    % Diferencia enx
LxC = [0 1 0 0 0 0]';
[DxC,FxC] = GFD_RobM2(x,y,LxC,a_T,b_T,c_T,d_T,f,m,m2,ba,bb,bc,bd,B,theta,ainf,binf,asup,bsup,aizq,bizq,ader,bder);
    % Diferencia en y
LyC = [0 0 1 0 0 0]';
[DyC,FyC] = GFD_RobM2(x,y,LyC,a_T,b_T,c_T,d_T,f,m,m2,ba,bb,bc,bd,B,theta,ainf,binf,asup,bsup,aizq,bizq,ader,bder);

%% modificaci\'on de las esquinas para cumplir condiciones de frontera
% Dirichlet
esq = [1 mm-m2+1];
cond = [0 0];
D2C(esq,esq) = speye(2);    F2C(esq) = cond;
DxC(esq,esq) = speye(2);    FxC(esq) = cond;
DyC(esq,esq) = speye(2);    FyC(esq) = cond;
% Neumann esquina m2
ii=m2;
i = [ii, ii+m2-1, ii+m2-1, ii-1, ii-2, ii+m2, ii+m2-2, ii+2*m2, ii+2*m2-1, ii+2*m2-2];   % \'Indices de los nodos soporte
h = x(i)-x(ii);   h = h';                 % Distancia horizontal del nodo central a los vecinos
k = y(i)-y(ii);   k = k';                 % Distancia vertical del nodo central a los vecinos
h(3) = - h(3); k(3) = -k(3);
M = zeros(6,10);                           % Inicializaci\'on de la matriz M
M(1,:) = 1;                               % Valores que aproximan u
M(2,:) = h;                               % Valores que aproximan u_x
M(3,:) = k;                               % Valores que aproximan u_y
M(4,:) = h.^2;                            % Valores que aproximan u_xx
M(5,:) = k.^2;                            % Valores que aproximan u_yy
M(6,:) = h.*k;                            % Valores que aproximan u_xy
Gamma = pinv(M)*L2C;                        % Soluci\'on para los pesos Gamma
D2C(ii,i(1)) = Gamma(1);
D2C(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
D2C(ii,i(4)) = Gamma(4);
D2C(ii,i(5)) = Gamma(5);
D2C(ii,i(6)) = Gamma(6);
D2C(ii,i(7)) = Gamma(7);
D2C(ii,i(8)) = Gamma(8);
D2C(ii,i(9)) = Gamma(9);
D2C(ii,i(10)) = Gamma(10);
F2C(ii) = f(x(ii),y(ii));
Gamma = pinv(M)*LxC;                        % Soluci\'on para los pesos Gamma
DxC(ii,i(1)) = Gamma(1);
DxC(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
DxC(ii,i(4)) = Gamma(4);
DxC(ii,i(5)) = Gamma(5);
DxC(ii,i(6)) = Gamma(6);
DxC(ii,i(7)) = Gamma(7);
DxC(ii,i(8)) = Gamma(8);
DxC(ii,i(9)) = Gamma(9);
DxC(ii,i(10)) = Gamma(10);
FxC(ii) = f(x(ii),y(ii));
Gamma = pinv(M)*LyC;                        % Soluci\'on para los pesos Gamma
DyC(ii,i(1)) = Gamma(1);
DyC(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
DyC(ii,i(4)) = Gamma(4);
DyC(ii,i(5)) = Gamma(5);
DyC(ii,i(6)) = Gamma(6);
DyC(ii,i(7)) = Gamma(7);
DyC(ii,i(8)) = Gamma(8);
DyC(ii,i(9)) = Gamma(9);
DyC(ii,i(10)) = Gamma(10);
FyC(ii) = f(x(ii),y(ii));

% Neumann esquina mm
ii=mm;
i = [ii, ii-m2-1, ii-m2-1, ii-1, ii-2, ii-m2, ii-m2-2, ii-2*m2, ii-2*m2-1, ii-2*m2-2];   % \'Indices de los nodos soporte
h = x(i)-x(ii);   h = h';                 % Distancia horizontal del nodo central a los vecinos
k = y(i)-y(ii);   k = k';                 % Distancia vertical del nodo central a los vecinos
h(3) = - h(3); k(3) = -k(3);
M = zeros(6,10);                           % Inicializaci\'on de la matriz M
M(1,:) = 1;                               % Valores que aproximan u
M(2,:) = h;                               % Valores que aproximan u_x
M(3,:) = k;                               % Valores que aproximan u_y
M(4,:) = h.^2;                            % Valores que aproximan u_xx
M(5,:) = k.^2;                            % Valores que aproximan u_yy
M(6,:) = h.*k;                            % Valores que aproximan u_xy
Gamma = pinv(M)*L2C;                        % Soluci\'on para los pesos Gamma
D2C(ii,i(1)) = Gamma(1);
D2C(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
D2C(ii,i(4)) = Gamma(4);
D2C(ii,i(5)) = Gamma(5);
D2C(ii,i(6)) = Gamma(6);
D2C(ii,i(7)) = Gamma(7);
D2C(ii,i(8)) = Gamma(8);
D2C(ii,i(9)) = Gamma(9);
D2C(ii,i(10)) = Gamma(10);
F2C(ii) = f(x(ii),y(ii));
Gamma = pinv(M)*LxC;                        % Soluci\'on para los pesos Gamma
DxC(ii,i(1)) = Gamma(1);
DxC(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
DxC(ii,i(4)) = Gamma(4);
DxC(ii,i(5)) = Gamma(5);
DxC(ii,i(6)) = Gamma(6);
DxC(ii,i(7)) = Gamma(7);
DxC(ii,i(8)) = Gamma(8);
DxC(ii,i(9)) = Gamma(9);
DxC(ii,i(10)) = Gamma(10);
FxC(ii) = f(x(ii),y(ii));
Gamma = pinv(M)*LyC;                        % Soluci\'on para los pesos Gamma
DyC(ii,i(1)) = Gamma(1);
DyC(ii,i(2)) = Gamma(2) + Gamma(3);
% K(ii,i(3)) = Gamma(3);
DyC(ii,i(4)) = Gamma(4);
DyC(ii,i(5)) = Gamma(5);
DyC(ii,i(6)) = Gamma(6);
DyC(ii,i(7)) = Gamma(7);
DyC(ii,i(8)) = Gamma(8);
DyC(ii,i(9)) = Gamma(9);
DyC(ii,i(10)) = Gamma(10);
FyC(ii) = f(x(ii),y(ii));

%% Sistema
% % D2*Psi - Ra*Dx*C = 0
% % D2*C - Dy*Psi*Dx*C + Dx*Psi*Dy*C = dC/dt  % Estacionario -> dC/dt = 0
Mceros = zeros(mm,mm);
Vceros = zeros(mm,1);

%
DxCpsi = DxC    ;   FxCpsi = FxC;
DxCpsi([bc bd ba bb],:) = 0;
FxCpsi([bc bd ba bb]) = 0;
A = [ D2psi  ,  -Ra*DxCpsi ;...
      Mceros ,  D2C    ];

% valores que pasan al lado derecho
Bder = [F2psi - Ra*FxCpsi ; Vceros + F2C] + ...              % parte lineal
       [Vceros     ; - Fxpsi.*FyC + Fypsi.*FxC];    % parte no lineal

% sistema acoplado (lineal + no lineal)
F = @(t,U) A*U - Bder + ...                  % parte lineal y lado derecho
    [Vceros; - (Dypsi*U(1:mm)).*(DxC*U(mm+1:2*mm)) + (Dxpsi*U(1:mm)).*(DyC*U(mm+1:2*mm))]; % parte no lineal


%% Resolver
% % aproximaciones iniciales
X = x(:); Y = y(:);
psi0 = 0*X;
C0 = Y .* ( (X<1).*X + (X>3).*(4-X) + (X>=1).*(X<=3) );
C0 = reshape(C0,m2,m); C0(2:end-1,2:end-1)=0; C0=C0(:);
U0 = [psi0; C0];

% figure; trisurf(tri,x,y,psi0); title('psi0')
% figure; trisurf(tri,x,y,C0); title('C0')
U0=U0';
tspan = [0 tfin];
[tode,U] = ode45(F,tspan,U0);

%% Gr\'afica de la soluci\'on no estacionaria
figure('Units','Normalized','OuterPosition',[0.025 0.15 0.45 0.8])
figure('Units','Normalized','OuterPosition',[0.5 0.15 0.45 0.8])
dt = tspan(end)/size(U,1);
curvaspsi = -21:3:21;
curvasc = -0.1:0.1:0.9;
ngraf_saltar = round(size(U,1)/10);
%%for i=1:ngraf_saltar:size(U,1)
%%    t = i*dt;
%%    figure(1)
%%    contourf(x,y,reshape(U(i,1:mm),m2,m),curvaspsi);
%%%     clabel(c,h,curvaspsi)
%%    axis([0 2 0 1])
%%    title(['\Psi , t = ', num2str(t)])
%%    figure(2)
%%    contourf(x,y,reshape(U(i,mm+1:2*mm),m2,m),curvasc);
%%%     clabel(c,h,curvasc)
%%    axis([0 2 0 1])
%%    title(['C , t = ', num2str(t)])
%%    pause(0.5)
%%end

figure(1)
[c,h]=contourf(x,y,reshape(U(end,1:mm),m2,m),curvaspsi);
clabel(c,h,curvaspsi)
axis([0 2 0 1])
title(['\Psi , t = ', num2str(tspan(end))])

figure(2)
[c,h]=contourf(x,y,reshape(U(end,mm+1:2*mm),m2,m),curvasc);
clabel(c,h,curvasc)
axis([0 2 0 1])
title(['C , t = ', num2str(tspan(end))])
% hold on
% quiver(x(:),y(:),Dypsi*U(end,1:mm)',-Dxpsi*U(end,1:mm)','Color','#c2c2c2')


%% gr\'afica de la soluci\'on estacionaria
psi = U(end,1:mm); C = U(end,mm+1:2*mm);

figure()
psi = reshape(psi,m2,m);
contour(x,y,psi,20)
title('Psi')
colorbar

figure()
C = reshape(C,m2,m);
contour(x,y,C,20)
title('C')
colorbar
hold on
quiver(x(:),y(:),Dypsi*psi(:),-Dxpsi*psi(:))

figure()
plot(x(:),y(:),".")
trimesh(tri,x,y)
title('Malla')
