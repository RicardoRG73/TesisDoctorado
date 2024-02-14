% Implementaci\'on del esquema GFDM para resolver el problema de
% Henry, el cual es un 'density-driven groundwater flows problem'
%% Las ecuaciones gobernantes son
% D2*Psi = 1/a * Dx*C
% D2*C - 1/b * ( Dy*Psi .* Dx*C  -  Dx*Psi .* Dy*C ) = dC/dt
% donde
% a es el parametro de descarga
% b es inverso del n\'umero de Peclet
% D2 es una matriz de Laplaciano Discreto
% Dx es una matriz de diferenciaci\'on en x
% Dy es una matriz de diferenciaci\'on en y
%% Las condiciones de frontera son
% frontera izquierda x=0
% C=0 , dPsi/dn = 0
% frontera derecha x=2
% C=1 , dPsi/dn = 0
% frontera inferior y=0
% dC/dn=0 , Psi=0
% frontera superior y=1
% dC/dn=0 , Psi=1
%% Definici\'on de par\'ametros
clear all; close all; clc
% constantes del problema
a = 0.2637;
b = 0.1;
tfin = 0.2;   % tiempo final
f = @(x,y) 0; % fuente
% par\'ametros para generar la malla
m = 21; % nodos en el eje x
m2 = 11; % nodos en el eje y
N = m*m2;
H = 1;
H_over_L = 0.5;
L = H / H_over_L;
theta = 0;

%% Generaci\on de la malla
[x,y,tri,B,ba,bb,bc,bd] = malla_estructurada2(H,L,theta,m,m2,1);

%% Condiciones de frontera
% considerando condiciones de robin con coeficientes a,b, tal que
% a*u + b*du/dn = g   ,   donde g es una funci\'on dada

% frontera izquierda x=0 (left)
% C=0 , dPsi/dn = 0
    % Concentraci\'on
    Cl=@(y) 0;
    Cla=@(y) 1; Clb=@(y) 0;     % coeficientes de robin
    % funci\'on de flujo
    Psil =@(y) 0;
    Psila=@(y) 0; Psilb=@(y) 1; % coeficientes de robin

% frontera derecha x=2 (right)
% C=1 , dPsi/dn = 0
    % concentraci\'on
    Cr=@(y) 1;
    Cra=@(y) 1; Crb=@(y) 0;     % coeficientes de robin
    % funci\'on de flujo
    Psir =@(y) 0;
    Psira=@(y) 0; Psirb=@(y) 1; % coeficientes de robin

% frontera inferior y=0 (down)
% dC/dn=0 , Psi=0
    % concentraci\'on
    Cd=@(x) 0;
    Cda=@(x) 0; Cdb=@(x) 1;     % coeficientes de robin
    % funci\'on de flujo
    Psid =@(x) 0;
    Psida=@(x) 1; Psidb=@(x) 0; % coeficientes de robin

% frontera superior y=1 (up)
% dC/dn=0 , Psi=1
    % concentraci\'on
    Cu=@(x) 0;
    Cua=@(x) 0; Cub=@(x) 1;     % coeficientes de robin
    % funci\'on de flujo
    Psiu =@(x) 1;
    Psiua=@(x) 1; Psiub=@(x) 0; % coeficientes de robin

%% Vectores de coeficientes L
L2 = [0 0 0 2 2 0]'; % laplaciano
Lx = [0 1 0 0 0 0]'; % diferencia en x
Ly = [0 0 1 0 0 0]'; % diferencia en y

%% Matrices de diferenciaci\'on D2,Dx,Dy, que cumplen las condiciones
%% de frontera, y valores que pasan al lado derecho F
% contenctraci\'on
[D2C,F2C] = GFD_RobM2(x,y,L2,Cl,Cr,Cd,Cu,f,m,m2,ba,bb,bc,bd,B,theta,Cda,Cdb,Cua,Cub,Cla,Clb,Cra,Crb);
[DxC,FxC] = GFD_RobM2(x,y,Lx,Cl,Cr,Cd,Cu,f,m,m2,ba,bb,bc,bd,B,theta,Cda,Cdb,Cua,Cub,Cla,Clb,Cra,Crb);
[DyC,FyC] = GFD_RobM2(x,y,Ly,Cl,Cr,Cd,Cu,f,m,m2,ba,bb,bc,bd,B,theta,Cda,Cdb,Cua,Cub,Cla,Clb,Cra,Crb);
% funci\'on de flujo
[D2Psi,F2Psi] = GFD_RobM2(x,y,L2,Psil,Psir,Psid,Psiu,f,m,m2,ba,bb,bc,bd,B,theta,Psida,Psidb,Psiua,Psiub,Psila,Psilb,Psira,Psirb);
[DxPsi,FxPsi] = GFD_RobM2(x,y,Lx,Psil,Psir,Psid,Psiu,f,m,m2,ba,bb,bc,bd,B,theta,Psida,Psidb,Psiua,Psiub,Psila,Psilb,Psira,Psirb);
[DyPsi,FyPsi] = GFD_RobM2(x,y,Ly,Psil,Psir,Psid,Psiu,f,m,m2,ba,bb,bc,bd,B,theta,Psida,Psidb,Psiua,Psiub,Psila,Psilb,Psira,Psirb);
% modificaci\'on de las 4 esquinas para condici\'on Dirichlet
esq = [1 m2 N N-m2+1];
condC = [Cl(0) Cl(H) Cr(0) Cr(H)];
D2C(esq,esq) = speye(4);    F2C(esq) = condC;
DxC(esq,esq) = speye(4);    FxC(esq) = condC;
DyC(esq,esq) = speye(4);    FyC(esq) = condC;
condPsi = [Psid(0) Psiu(0) Psiu(L) Psid(L)];
D2Psi(esq,esq) = speye(4);    F2Psi(esq) = condPsi;
DxPsi(esq,esq) = speye(4);    FxPsi(esq) = condPsi;
DyPsi(esq,esq) = speye(4);    FyPsi(esq) = condPsi;

%% Ensamble del sistema de ecuaciones
% D2*Psi = 1/a * Dx*C
% D2*C - 1/b * ( Dy*Psi .* Dx*C  -  Dx*Psi .* Dy*C ) = dC/dt
% definiendo un vector soluci\'on U=[Psi ; C];
% === Parte Lineal del sistema (Matriz A)===
Mceros = zeros(N,N);       % mismo tamano que D2,Dx,Dy
Vceros = zeros(N,1);        % mismo tamano que F2,Fx,Fy

DxCpsi = DxC; DxCpsi(B,:) = 0;  % mod de DxC para no afectar la frontera en psi
FxCpsi = FxC; FxCpsi(B) = 0;    % mod de FxC para no afectar la frontera en psi

DyPsiC = DyPsi; DyPsiC(B,:) = 0;
FyPsiC = FyPsi; FyPsiC(B) = 0;
DxPsiC = DxPsi; DxPsiC(B,:) = 0;
FxPsiC = FxPsi; FxPsiC(B) = 0;

A = [D2Psi , -1/a*DxCpsi   ;  Mceros , D2C];      % lineal
Ar = [F2Psi-1/a*FxCpsi   ;   Vceros+F2C];          % lado derecho

% === Parte Lineal y No lineal F(U) ===
uno = 1:N;         % primeros indices de U corresponden a Psi
dos = N+1:2*N;    % segundos indices de U corresponden a C

F =@(t,U) A*U - Ar ...
    +[Vceros ; - 1/b*(DyPsiC*U(uno)).*(DxC*U(dos)) + 1/b*(DxPsiC*U(uno)).*(DyC*U(dos))]...
    -[Vceros ; - 1/b* FyPsiC        .* FxC         + 1/b* FxPsiC        .* FyC];

% definiendo una aproximaci\'on inicial U0
Psi0=y(:);
% figure(); surf(x,y,reshape(Psi0,m2,m)); title('\Psi_0'); colorbar

C0=x(:)./L;
% figure(); surf(x,y,reshape(C0,m2,m)); title('C_0'); colorbar

%% Resolviendo problema estacionario
% % primera aproximaci\'on usando algoritmo predictor corrector
% tol = 1e-3;
% norma1=1; norma2=1; iter = 1; itermax = 300;
% while (norma1>tol && norma2>tol) && iter <= itermax
%     Psin = D2Psi\( +1/a*DxCpsi*C + F2Psi + 1/a*FxCpsi);
%     Cn = (D2C - 1/b*( (DyPsiC*Psin).*(DxC) - (DxPsiC*Psin).*(DyC) )) \ ( F2C - 1/b*( FyPsiC.*FxC - FxPsiC.*FyC ));
%     norma1 = norm(abs(Psin-Psi));
%     norma2 = norm(abs(Cn-C));
%     C = Cn; Psi = Psin;
%     fprintf('iter = %d  |  norm(Psin-Psi) = %2.4e  |  norm(Cn-C) = %2.4e \n',[iter,norma1,norma2])
%     iter = iter+1;
% end
% % funci\'on de flujo
% figure()
% % surf(x,y,reshape(Psi,m2,m)); title('\Psi')
% curvas = [-0.15 -0.1 -0.05 0:0.1:1.2];
% [c,h]=contour(x,y,reshape(Psi,m2,m),curvas); title('Psi')
% clabel(c,h,curvas)
%
% % concentraci\'on
% figure()
% % surf(x,y,reshape(C,m2,m)); title('C')
% curvas = 0.1:0.1:1;
% [c,h]=contour(x,y,reshape(C,m2,m),curvas); title('C')
% clabel(c,h,curvas)
%
% % Velodicad de flujo
% vx =  DyPsi*Psi; vx = reshape(vx,m2,m);
% vy = -DxPsi*Psi; vy = reshape(vy,m2,m);
% figure()
% quiver(x,y,vx,vy); title('Velocidad de flujo')
%
% % calculo de x-Toe
% figure()
% curvas = [0.5 0.5];
% [c,h]=contour(x,y,reshape(C,m2,m),curvas); title('x-Toe')
% clabel(c,h,curvas)
% xToe = c(:,end)';
% hold on
% plot(xToe(1),xToe(2),'*r')
%
% % segunda aproximaci\'on
% % U1 = [Psi; C];
% % opciones = optimset('Display','iter','MaxFunEvals',1e5);
% % [U,fval,exitflag,output] = fsolve(F,U1, opciones);
% %
% % %% Graficando resultados
% % Psi = U(1:N);
% % C = U(N+1:2*N);
% % % función de flujo
% % figure()
% % surf(x,y,reshape(Psi,m2,m)); title('\Psi')
% % % concentración
% % figure()
% % surf(x,y,reshape(C,m2,m)); title('C')


%% Resolviendo problema NO estacionario
Psi0=reshape(Psi0,m2,m);
C0=reshape(C0,m2,m);
Psi0(2:end-1,2:end-1)=0;
C0(2:end-1,2:end-1)=0;
U0 = [Psi0(:); C0(:)];
U0 = U0';
tspan = linspace(0,tfin,1000);
[t,U] = ode45(F,tspan,U0);

% graficando la soluci\'on

curvasc=0.2:0.1:0.8;
curvaspsi=[-0.15 , -0.05, -0.2:0.1:1];

%figure()
%figure()
%dt = tspan(end)/size(U,1);

%ngraf_saltar = round(size(U,1)/20);
%for i=1:ngraf_saltar:size(U,1)
%    tiempo = i*dt;
%    figure(1)
%    [c,h]=contour(x,y,reshape(U(i,1:N),m2,m),curvaspsi);
%    clabel(c,h,curvaspsi)
%    title(['\Psi , t = ', num2str(tiempo)])
%    figure(2)
%    [c,h]=contour(x,y,reshape(U(i,N+1:2*N),m2,m),curvasc);
%    clabel(c,h,curvasc)
%    title(['C , t = ', num2str(tiempo)])
%    pause(0.7)
%end

%% Soluciones al t final
Psifinal = U(end,1:N)'; Cfinal = U(end,N+1:2*N)';

% velocidad
figure()
quiver(x(:),y(:),DyPsi*Psifinal,-DxPsi*Psifinal)
axis([0 2 0 1])
title(['Velocidad , t = ', num2str(tspan(end))])

% funci\'on de flujo
figure()
[c,h]=contour(x,y,reshape(Psifinal,m2,m),curvaspsi);
clabel(c,h,curvaspsi)
title(['\Psi , t = ', num2str(tspan(end))])

% concentraci\'on
figure()
[c,h]=contour(x,y,reshape(Cfinal,m2,m),curvasc);
clabel(c,h,curvasc)
title(['C , t = ', num2str(tspan(end))])

% x-Toe
hold on
curvas = [0.5 0.5];
[c,h]=contour(x,y,reshape(Cfinal,m2,m),curvas);
xToe = c(:,2)'
plot(xToe(1), xToe(2), '*r', 'MarkerSize', 15)
clabel(c,h,curvasc)
