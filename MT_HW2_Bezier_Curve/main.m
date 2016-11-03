clear classes;
prefix = 'images';
%controls = [8,0;10,5;0,6;9.5,2];
renDim = 100;
controls = [];%Randomly generate four points in space
for i = 1:4
    controls = [ controls; rand()*renDim, rand()*renDim];
end
mkdir(prefix);
for i = 1:13
    t = bezier_c(controls, i, 0);
    image = plot(t(:,1),t(:,2));hold on;
    image = scatter(t(:,1),t(:,2));hold off;
    path = fullfile(prefix,sprintf('illustrate_%d.jpg',i));
    saveas(image,path);
end
image = plot(t(:,1),t(:,2));
saveas(image,fullfile(prefix,'Final.png'));