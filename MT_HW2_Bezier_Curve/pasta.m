clear classes;
prefix = 'pasta';
mkdir(prefix);
%controls = [8,0;10,5;0,6;9.5,2];
renDim = 100;
linNum = 55;
thHOLD = 14;
for j = 1:linNum
    
controls = [ 0,renDim/2 ];%Randomly generate four points in space
for i = 1:2
    controls = [ controls; rand()*renDim, rand()*renDim];
end
controls = [controls; renDim,renDim/2 ];
hold on;
    t = bezier_c(controls, thHOLD, 0);
    image = plot(t(:,1),t(:,2),'LineWidth',4,'color',rand(1,3));
end
saveas(image,fullfile(prefix,'BeautifulPasta.png'));