function [x,y,tri,B,ba,bb,bc,bd]=malla_estructurada2(H,L,theta,m,m2,grafica)
% Funci\'on que genera una malla con estructura regular en un dominio
% con forma de paralelogramo donde 
% H es la altura
% L la longitud,
% theta es la inclinaci\'on respecto a la horizontal en grados
% m es el numero de nodos por cada direcci\'on
% por lo que habr\'a m^2 nodos en la malla
%% Discretizaci\'on del dominio
theta=theta*pi/180;                 % \'Angulo en radianes
ca=L*cos(theta);                    % Cateto adyacente
x=linspace(0,ca,m);                 % Discretizaci\'on de x
y=linspace(0,H,m2);                  % Discretizaci\'on de y
s=tan(theta);                       % Pendiente del paralelogramo
r=s*x;                              % Factor para sumar a los valores de y
[x,y]=meshgrid(x,y);                % Malla rectangular
for i=1:m
    y(:,i)=y(:,i)+r(i);             % Sumando los factores r a las columnas de y
end
%% Obtenci\'on de los nodos de frontera
mm=m*m2;                              % N\'umero de nodos en la malla
ba=1:m2;                             % \'Indices de los nodos frontera izquierda
bb=mm-m2+1:mm;                         % \'Indices de los nodos frontera derecha
bc=m2+1:m2:mm-2*m2+1;                   % \'Indices de los nodos frontera inferior
bd=2*m2:m2:mm-m2;                       % \'Indices de los nodos frontera superior
B=[ba bb bc bd];                    % % \'Indices de totos los nodos frontera
%% Triangulaci\'on
tri=[[1:m2-1 ; m2+1:2*m2-1 ; m2+2:2*m2]' ;...
     [1:m2-1 ; m2+2:2*m2 ; 2:m2 ]' ];
tam=size(tri,1);
for i=1:m-2
    tri=[tri ; tri(1:tam,:) + m2*i];
end
%% Gr\'afica del dominio y nodos de frontera
if grafica == 1
    figure('units','Normalized','OuterPosition',[0 0 1 1])
    triplot(tri,x,y,'Color','blue')
    hold on
    plot(x(ba),y(ba),'*r','markersize',10)
    plot(x(bb),y(bb),'*g','markersize',10)
    plot(x(bc),y(bc),'db','markersize',10)
    plot(x(bd),y(bd),'ok','markersize',10)
    % dando formato a la gr\'afica
    grid on
    axis equal
    legend('Malla','ba', 'bb', 'bc', 'bd','Location','NorthWest')
    set(gca,'fontsize',20)
    xlabel('x','fontsize',24)
    ylabel('y','fontsize',24)
end
end