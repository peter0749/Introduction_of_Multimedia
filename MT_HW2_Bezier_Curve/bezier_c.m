function b = bezier_c(P,max_depth,depth)
P = double(P)
if(depth>=max_depth || size(P,1)<4 || size(P,2)<2)
   return;
end
p = [];
for i = 1:3
    p = [ p;(P(i,:)+P(i+1,:))/2 ];
end
p0112 = ( p(1,:)+p(2,:) )/2;
p1223 = ( p(2,:)+p(3,:) )/2;
pm = (p0112+p1223)/2;
[ P(1,:); p(1,:); p0112; pm ]
bezier_c( [ P(1,:); p(1,:); p0112; pm ], max_depth, depth+1 );
bezier_c( [pm; p1223; p(3,:); P(4,:)], max_depth,depth+1 );

end