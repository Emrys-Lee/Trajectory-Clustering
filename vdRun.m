close all
clear

load vdData.mat
load color.mat

dt = 0; % data type, 0:bboxes, 1:splines
if dt==0
    Traj=DataBboxes;
else
    Traj=DataSplines;
    for i=1:length(Traj)
        Traj(i).data = Traj(i).data*20+500;
    end
end
Traj1 = Traj;
Traj2 = Traj;
Traj3 = Traj;
Traj4 = Traj;

if dt==0
    Iter = 20;
    rmax = 100;%100 is best for bbox;
    rmin = 1;%1 is best for bbox
else
    Iter = 10;
    rmax = 50;%50 is best for spline
    rmin = 1;%1 is best for spline
end
wr=rmax:(rmin-rmax)/(Iter-1):rmin;
lambda=0.5;
lr=5;

flag=1;
flagf=0;
gr=3;
D=30;
[Traj1, ~, ~] = ExtractFeature(Traj1, D, flag, flagf, gr);
[Traj2, ~, ~] = ExtractFeature(Traj2, D, flag, flagf, gr);
[Traj3, ~, ~] = ExtractFeature(Traj3, D, flag, flagf, gr);
[Traj4, ~, ~] = ExtractFeature(Traj4, D, flag, flagf, gr);

figure
subplot(2,3,1)
if dt==0
    img1=imread('black.jpg');
    %img1=imread('IMG_0121.jpg');
else
    img1=imread('black.jpg');
end
image(img1)
hold on
for i=1:length(Traj)
    traj=Traj(i).data;
    label=Traj(i).label;
    plot(traj(:,1),traj(:,2),'color',color(label,:));
end
hold off
axis tight

for k=1:Iter
    k
    if k==1  
        [Traj1, Map] = MeanShift( Traj1, wr(k) );
        Traj2 = MBMSFast( Traj2, wr(k), lr, Map );
        [~, R, C] = ParaConfig( Traj3 );
        [Traj3,~] = FastAMKS( Traj3, wr(k), 0, R, C, Map );
        [~, R, C] = ParaConfig( Traj4 );
        [Traj4,~] = FastAMKS( Traj4, wr(k), lambda, R, C, Map );
    else
        Traj1 = MeanShiftFast( Traj1, wr(k), Map );
        Traj2 = MBMSFast( Traj2, wr(k), lr, Map );
        [~, R, C] = ParaConfig( Traj3 );
        [Traj3,~] = FastAMKS( Traj3, wr(k), 0, R, C, Map );
        [~, R, C] = ParaConfig( Traj4 );
        [Traj4,~] = FastAMKS( Traj4, wr(k), lambda, R, C, Map );
    end
end


subplot(2,3,2)
image(img1)
hold on
for i=1:length(Traj1)
    traj=Traj1(i).data;
    label=Traj1(i).label;
    plot(traj(:,1),traj(:,2),'color',color(label,:));
end
hold off
axis tight
title('MeanShift');

subplot(2,3,3)
image(img1)
hold on
for i=1:length(Traj2)
    traj=Traj2(i).data;
    label=Traj2(i).label;
    plot(traj(:,1),traj(:,2),'color',color(label,:));
end
hold off
axis tight
title('MBMS');

subplot(2,3,4)
image(img1)
hold on
for i=1:length(Traj3)
    traj=Traj3(i).data;
    label=Traj3(i).label;
    plot(traj(:,1),traj(:,2),'color',color(label,:));
end
hold off
axis tight
title('Without Speed Regularization')

subplot(2,3,5)
image(img1)
hold on
for i=1:length(Traj4)
    traj=Traj4(i).data;
    label=Traj4(i).label;    
    if label==1
        continue
    end
    plot(traj(:,1),traj(:,2),'color',color(label,:));
end
hold off
axis tight
title('Proposed FastAMKS')
