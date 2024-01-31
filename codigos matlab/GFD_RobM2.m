function [K,F]=GFD_RobM2(x,y,L,a,b,c,d,f,m,m2,ba,bb,bc,bd,B,theta,ainf,binf,asup,bsup,aizq,bizq,ader,bder)
% Implementaci\'on del esquema de GFDM para resolver 
% la ecuaci\'on de difusi\'on en 2D
% -div(grad(u)) = f
% Donde el dominio tiene forma de paralelogramo, inclinado respecto de la
% horizontal un \'angulo \theta
% Con condiciones de frontera:
%   Dirichlet en fronteras izquierda y derecha
%   u(0,y)=a(y)
%   u(1,y)=b(y)
%   Robin en fronteras inferior y superior
%   ar*u(x,0) + br*u_y(x,0) = c(x)
%   ar*u(x,1) + br*u_y(x,1) = d(x)

% distancia horizontal y vertical
dy = y(2)-y(1);
dx = x(m2+1)-x(1);
% componentes del vector normal
nx = cos(theta+pi/2);
ny = sin(theta+pi/2);

%
x = x(:);                       % x como vector columna
y = y(:);                       % y como vector columna
L = L(:);
mm = m*m2;                        % N\'umero de nodos en la malla (m en cada direcci\'on)

% \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi+m+1, Gi-m-1]
%                     1    2     3     4     5      6       7
%% Integraci\'on
K = sparse(mm,mm);
F = zeros(mm,1);
nodos = 1:mm;                                  % \'Indices de todos los nodos
interiores = setdiff(nodos,B);                % \'Indices de los nodos interiores

%% Nodos interiores
for j = interiores                            % Ciclo en los nodos interiores
    % Matriz M del sistema M\Gamma=L para encontrar los pesos \Gamma_j
    ii = j;                                   % \'Indice del nodo central
    i = [ii, ii+1, ii-1, ii+m2, ii-m2, ii+m2+1, ii-m2-1];   % \'Indices de los nodos soporte
    h = x(i)-x(ii);   h = h';                 % Distancia horizontal del nodo central a los vecinos
    k = y(i)-y(ii);   k = k';                 % Distancia vertical del nodo central a los vecinos
    M = zeros(6,7);                           % Inicializaci\'on de la matriz M
    M(1,:) = 1;                               % Valores que aproximan u
    M(2,:) = h;                               % Valores que aproximan u_x
    M(3,:) = k;                               % Valores que aproximan u_y
    M(4,:) = h.^2;                            % Valores que aproximan u_xx
    M(5,:) = k.^2;                            % Valores que aproximan u_yy
    M(6,:) = h.*k;                            % Valores que aproximan u_xy
    Gamma = pinv(M)*L;                        % Soluci\'on para los pesos Gamma
    
    % Ensamble de la matriz K y el vector F del sistema KU=F, usando los
    % pesos \Gamma_j
    K(ii,i(1)) = Gamma(1);
    K(ii,i(2)) = Gamma(2);
    K(ii,i(3)) = Gamma(3);
    K(ii,i(4)) = Gamma(4);
    K(ii,i(5)) = Gamma(5);
    K(ii,i(6)) = Gamma(6);
    K(ii,i(7)) = Gamma(7);
    F(ii) = f(x(ii),y(ii));
end

%% Frontera Izquierda Dirichlet u(0,y)=a(y)
% indice del nodo fantasma i-m
for j = ba
    ii = j;                                          % \'Indice del nodo central
    if bizq(y(ii)) == 0 || ii == ba(1) || ii == ba(m2)
        K(ii,ii) = 1;                                % Diagonal igual a 1 por condici\'on de Dirichlet
        F(ii) = a(y(ii));                            % F igual a la funci\'on dada
    else
        % Matriz M del sistema M\Gamma=L para encontrar los pesos \Gamma_j
        i = [ii, ii+1, ii-1, ii+m2, ii+m2, ii+m2+1, ii+m2-1]';   % \'Indices de los nodos soporte
        h = x(i)-x(ii);   h = h';                     % Distancia horizontal del nodo central a los vecinos
        h(5) = -h(5);                                 % nodo fantasma dist x
        k = y(i)-y(ii);   k = k';                     % Distancia vertical del nodo central a los vecinos
        k(5) = -k(5);                                 % nodo fantasma dist y
        M = zeros(6,7);                               % Inicializaci\'on de la matriz M
        M(1,:) = 1;                                   % Valores que aproximan u
        M(2,:) = h;                                   % Valores que aproximan u_x
        M(3,:) = k;                                   % Valores que aproximan u_y
        M(4,:) = h.^2;                                % Valores que aproximan u_xx
        M(5,:) = k.^2;                                % Valores que aproximan u_yy
        M(6,:) = h.*k;                                % Valores que aproximan u_xy
        Gamma = pinv(M)*L;                            % Soluci\'on para los pesos Gamma
        
        % Ensamble de la matriz K y el vector F del sistema KU=F
        % \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi+m+1, Gi+m-1]
        %                     1    2     3     4     5      6       7
        K(ii,ii)       = Gamma(1) - 2*dx * aizq(y(ii))/bizq(y(ii)) * Gamma(5);
        K(ii,ii+1)     = Gamma(2) + dx/dy * 1/bizq(y(ii)) * sin(theta) * Gamma(5);
        K(ii,ii-1)     = Gamma(3) - dx/dy * 1/bizq(y(ii)) * sin(theta) * Gamma(5);
        K(ii,ii+m2)     = Gamma(4) + Gamma(5);
        % K(ii,ii-m)   = 0; % Nodo fantasma
        K(ii,ii+m2+1)   = Gamma(6);
        K(ii,ii+m2-1)   = Gamma(7);
        % modificaci\'on de F en la frontera
        F(ii) = f(x(ii),y(ii)) + Gamma(5) * 2*dx * a(y(ii)) * 1/bizq(y(ii));
    end
end

%% Frontera Derecha Dirichlet u(1,y)=b(y)
% indice del nodo fantasma i+m
for j = bb
    ii = j;                                          % \'Indice del nodo central
    if bder(y(ii)) == 0 || ii == bb(1) || ii == bb(m2)
        K(ii,ii) = 1;                                    % Diagonal igual a 1 por condici\'on de Dirichlet
        F(ii) = b(y(ii));                                % F igual a la funci\'on dada
    else
        % Matriz M del sistema M\Gamma=L para encontrar los pesos \Gamma_j
        i = [ii, ii+1, ii-1, ii-m2, ii-m2, ii-m2+1, ii-m2-1]';   % \'Indices de los nodos soporte
        h = x(i)-x(ii);   h = h';                     % Distancia horizontal del nodo central a los vecinos
        h(4) = -h(4);   % nodo fantasma dist x
        k = y(i)-y(ii);   k = k';                     % Distancia vertical del nodo central a los vecinos
        k(4) = -k(4);   % nodo fantasma dist y
        M = zeros(6,7);                               % Inicializaci\'on de la matriz M
        M(1,:) = 1;                                   % Valores que aproximan u
        M(2,:) = h;                                   % Valores que aproximan u_x
        M(3,:) = k;                                   % Valores que aproximan u_y
        M(4,:) = h.^2;                                % Valores que aproximan u_xx
        M(5,:) = k.^2;                                % Valores que aproximan u_yy
        M(6,:) = h.*k;                                % Valores que aproximan u_xy
        Gamma = pinv(M)*L;                            % Soluci\'on para los pesos Gamma
        
        % Ensamble de la matriz K y el vector F del sistema KU=F
        % \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi-m+1, Gi-m-1]
        %                     1    2     3     4     5      6       7
        K(ii,ii)       = Gamma(1) - 2*dx * ader(y(ii))/bder(y(ii)) * Gamma(4);
        K(ii,ii+1)     = Gamma(2) + dx/dy * 1/bder(y(ii)) * sin(theta) * Gamma(4);
        K(ii,ii-1)     = Gamma(3) - dx/dy * 1/bder(y(ii)) * sin(theta) * Gamma(4);
%         K(ii,ii+m)   = 0; % Nodo fantasma
        K(ii,ii-m2)     = Gamma(5) + Gamma(4);
        K(ii,ii-m2+1)   = Gamma(6);
        K(ii,ii-m2-1)   = Gamma(7);
        % modificaci\'on de F en la frontera
        F(ii) = f(x(ii),y(ii)) - Gamma(4) * 2*dx * b(y(ii)) * 1/bder(y(ii));
    end
end

%% Frontera Inferior Robin a*u(x,0) + b*u(x,0) = c(x)
% el nodo fantasma tiene \'indice i-1
for j = bc
    ii = j;                                          % \'Indice del nodo central
    if binf(x(ii)) == 0
        K(ii,ii) = 1;
        F(ii) = c(x(ii))/ainf(x(ii));
    else
        i = [ii, ii+1, ii+1, ii+m2, ii-m2, ii+m2+1, ii-m2+1]';   % \'Indices de los nodos soporte
        h = x(i)-x(ii);   h = h';                     % Distancia horizontal del nodo central a los vecinos
        h(3) = -h(3);   % nodo fantasma dist x
        k = y(i)-y(ii);   k = k';                     % Distancia vertical del nodo central a los vecinos
        k(3) = -k(3);   % nodo fantasma dist y
        M = zeros(6,7);                               % Inicializaci\'on de la matriz M
        M(1,:) = 1;                                   % Valores que aproximan u
        M(2,:) = h;                                   % Valores que aproximan u_x
        M(3,:) = k;                                   % Valores que aproximan u_y
        M(4,:) = h.^2;                                % Valores que aproximan u_xx
        M(5,:) = k.^2;                                % Valores que aproximan u_yy
        M(6,:) = h.*k;                                % Valores que aproximan u_xy
        Gamma = pinv(M)*L;                            % Soluci\'on para los pesos Gamma
        
        
        % \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi+m+1, Gi-m+1]
        %                     1    2     3     4     5      6       7
        nymod = nx*sin(theta)-ny;
        K(ii,ii)       = Gamma(1) + 2*dy* ainf(x(ii))/binf(x(ii)) * 1/nymod * Gamma(3);
        K(ii,ii+1)     = Gamma(2)+Gamma(3);
        % K(ii,ii-1)   = 0;  % Nodo fantasma
        K(ii,ii+m2)     = Gamma(4) - Gamma(3) * dy/dx * nx/nymod;
        K(ii,ii-m2)     = Gamma(5) + Gamma(3) * dy/dx * nx/nymod;
        K(ii,ii+m2+1)   = Gamma(6);
        K(ii,ii-m2+1)   = Gamma(7);
        % modificaci\'on de F en la frontera
        F(ii) = f(x(ii),y(ii)) + Gamma(3) * 2*dy*c(x(ii)) * 1/binf(x(ii)) * 1/nymod;
    end
end

%% Frontera Superior Robin a*u(x,1) + b*u_y(x,1) = d(x)
% el nodo fantasma tiene \'indice i+1
for j = bd
    ii = j;                                          % \'Indice del nodo central
    if bsup(x(ii)) == 0
        K(ii,ii) = 1;
        F(ii) = d(x(ii))/asup(x(ii));
    else
        % Matriz M del sistema M\Gamma=L para encontrar los pesos \Gamma_j
        i = [ii, ii-1, ii-1, ii+m2, ii-m2, ii+m2-1, ii-m2-1]';   % \'Indices de los nodos soporte
        h = x(i)-x(ii);   h = h';                     % Distancia horizontal del nodo central a los vecinos
        h(2) = -h(2);   % nodo fantasma dist x
        k = y(i)-y(ii);   k = k';                     % Distancia vertical del nodo central a los vecinos
        k(2) = -k(2);   % nodo fantasma dist y
        M = zeros(6,7);                               % Inicializaci\'on de la matriz M
        M(1,:) = 1;                                   % Valores que aproximan u
        M(2,:) = h;                                   % Valores que aproximan u_x
        M(3,:) = k;                                   % Valores que aproximan u_y
        M(4,:) = h.^2;                                % Valores que aproximan u_xx
        M(5,:) = k.^2;                                % Valores que aproximan u_yy
        M(6,:) = h.*k;                                % Valores que aproximan u_xy
        Gamma = pinv(M)*L;                            % Soluci\'on para los pesos Gamma
        
        % Ensamble de la matriz K y el vector F del sistema KU=F
        % \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi+m-1, Gi-m-1]
        %                     1    2     3     4     5      6       7
        nymod = nx*sin(theta)-ny;
        K(ii,ii)     = Gamma(1) + 2*dy * asup(x(ii))/bsup(x(ii)) * 1/nymod * Gamma(2);
        % K(ii,ii+1)   = 0;   % Nodo fantasma
        K(ii,ii-1)   = Gamma(3) + Gamma(2);
        K(ii,ii+m2)   = Gamma(4) + Gamma(2) * dy/dx * nx/nymod;
        K(ii,ii-m2)   = Gamma(5) - Gamma(2) * dy/dx * nx/nymod;
        K(ii,ii-m2-1) = Gamma(7);
        K(ii,ii+m2-1) = Gamma(6);
        % modificaci\'on de F en la frontera
        F(ii) = f(x(ii),y(ii)) - Gamma(2) * 2*dy*d(x(ii)) * 1/bsup(x(ii)) * 1/nymod;
    end
end
end