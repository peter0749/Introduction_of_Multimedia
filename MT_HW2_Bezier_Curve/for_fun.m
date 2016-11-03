clear classes;
prefix = 'fun';
mkdir(prefix);
%controls = [8,0;10,5;0,6;9.5,2];
renDim = 100;
linNum = 16;
thHOLD = 13;
for j = 1:linNum
    
controls = [ renDim/2, 0 ];%Randomly generate four points in space
for i = 1:3
    controls = [ controls; rand()*renDim, rand()*renDim];
end
hold on;
    t = bezier_c(controls, thHOLD, 0);
    image = plot(t(:,1),t(:,2),'color',rand(1,3));
end
saveas(image,fullfile(prefix,'Beautiful.png'));