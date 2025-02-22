gm = importGeometry('BlueTubeFinAlignment.stl');
pdegplot(gm);
hold on;
x = [0, 10];
y = [2, 30];
plot(x, y, 'linewidth', 2);
hold off;
