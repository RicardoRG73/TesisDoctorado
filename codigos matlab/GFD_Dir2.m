function [K,F]=GFD_Dir2(x,y,L,a,b,c,d,f,m,m2,ba,bb,bc,bd,B)
% Implementaci\'on del esquema de GFDM para resolver 
% la ecuaci\'on de difusi\'on en 2D
% -div(grad(u)) = f
% Donde el dominio tiene forma de paralelogramo, inclinado respecto de la
% horizontal un \'angulo \theta
% Con condiciones de frontera:
%   Dirichlet en todas las fronteras
%   u(0,y)=a(y)
%   u(1,y)=b(y)
%   u(x,0)=c(x)
%   u(x,1)=d(x)
x = x(:);                       % x como vector columna
y = y(:);                       % y como vector columna
mm = m*m2;                        % N\'umero de nodos en la malla (m en cada direcci\'on)
%% Matriz M del sistema M\Gamma=L para encontrar los pesos \Gamma_j
ii = m2+2 ;                                    % \'Indice del nodo central
i = [ii, ii+1, ii-1, ii+m2, ii-m2, ii+m2+1, ii-m2-1];   % \'Indices de los nodos soporte
h = x(i)-x(ii);                        % Distancia horizontal del nodo central a los vecinos
k = y(i)-y(ii);                        % Distancia vertical del nodo central a los vecinos
M = zeros(6,7);                               % Inicializaci\'on de la matriz M
M(1,:) = 1;                                   % Valores que aproxima
M(2,:) = h;                                   % Valores que aproximan u_x
M(3,:) = k;                                   % Valores que aproximan u_y
M(4,:) = h.^2;                                % Valores que aproximan u_xx
M(5,:) = k.^2;                                % Valores que aproximan u_yy
M(6,:) = h.*k;                                % Valores que aproximan u_xy
Gamma = pinv(M)*L;                            % Soluci\'on para los pesos Gamma
% \'Indices Gamma = [Gi, Gi+1, Gi-1, Gi+m, Gi-m, Gi+m+1, Gi-m-1]
%                     1    2     3     4     5      6       7
%% Integraci\'on
K = sparse(mm,mm);
F = zeros(mm,1);
nodos = 1:mm;                                  % \'Indices de todos los nodos
interiores = setdiff(nodos,B);                % \'Indices de los nodos interiores
    %% Nodos interiores
for j = interiores                                   % Ciclo en los nodos interiores
    ii = j;                                          % \'Indice del nodo central
    i = [ii, ii+1, ii-1, ii+m2, ii-m2, ii+m2+1, ii-m2-1];% \'Indices de los nodos soporte
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
for j = ba
    ii = j;                                          % \'Indice del nodo central
    K(ii,ii) = 1;                                    % Diagonal igual a 1 por condici\'on de Dirichlet
    F(ii) = a(y(ii));                                % F igual a la funci\'on dada
end
    %% Frontera Derecha Dirichlet u(1,y)=b(y)
for j = bb
    ii = j;                                          % \'Indice del nodo central
    K(ii,ii) = 1;                                    % Diagonal igual a 1 por condici\'on de Dirichlet
    F(ii) = b(y(ii));                                % F igual a la funci\'on dada
end
    %% Frontera Inferior Dirichlet u(x,0)=c(x)
for j = bc
    ii = j;                                          % \'Indice del nodo central
    K(ii,ii) = 1;                                    % Diagonal igual a 1 por condici\'on de Dirichlet
    F(ii) = c(x(ii));                                % F igual a la funci\'on dada
end
    %% Frontera Superior Dirichlet u(x,1)=d(x)
for j = bd
    ii = j;                                          % \'Indice del nodo central
    K(ii,ii) = 1;                                    % Diagonal igual a 1 por condici\'on de Dirichlet
    F(ii) = d(x(ii));                                % F igual a la funci\'on dada
end
end