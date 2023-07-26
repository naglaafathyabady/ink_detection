%%%model 1 n.e=10,minpatch=100,rate.005  rms     aacuracy 98.87
%%%model 1 n.e=10,minpatch=100,rate.005  adam     aacuracy 99.17
%%%model 1 n.e=10,minpatch=100,rate.005  sgdm     aacuracy 98.42
%%%model 1 n.e=10,minpatch=100,rate.005  rms     aacuracy 98.32
%%%model 1 n.e=10,minpatch=100,rate.005  adam     aacuracy 98.91
%%%model 1 n.e=10,minpatch=300,rate.005  sgdm     aacuracy 98.64
%%%model 1 n.e=10,minpatch=300,rate.0005  rms     aacuracy 98.91
%%%model 1 n.e=10,minpatch=300,rate.0005  adam     aacuracy 98.95
%%%model 1 n.e=10,minpatch=300,rate.0005  sgdm     aacuracy 98.36
%%%model 2 n.e=10,minpatch=300,rate.05  rms     aacuracy 98.78
%%%model 2 n.e=10,minpatch=300,rate.05  adam     aacuracy 98.17
%%%model 2 n.e=10,minpatch=300,rate.05  sgdm     aacuracy 98.89
%%%model 3 n.e=10,minpatch=300,rate.05  rms     aacuracy 96.83
%%%model 3 n.e=10,minpatch=300,rate.05  adam     aacuracy 99.1
%%%model 3 n.e=10,minpatch=300,rate.05  sgdm     aacuracy 99.1
%%%model 3 n.e=10,minpatch=300,rate.05  rms     aacuracy 99.2
%%%model 3 n.e=10,minpatch=300,rate.05  adam     aacuracy 98.95
%%%model 3 n.e=10,minpatch=300,rate.001  sgdm     aacuracy 98.63
%%%model 2 4 conv e=15,minpatch=300 rate=0.005 sgdm  accuracy=99.1
clc
close all
clear
tic
root_path = 'new/'; norm_path = 'new/';
ink_name = {'Blue', 'Black'}; numInks = length(ink_name);
a=load([root_path ink_name{1} sprintf('/train/spectral_%02d.mat',1)]);

b=load([root_path ink_name{1} sprintf('/train/spectral_%02d.mat',2)]);
c=load([root_path ink_name{1} sprintf('/train/spectral_%02d.mat',3)]);
d=load([root_path ink_name{1} sprintf('/train/spectral_%02d.mat',4)]);
e=load([root_path ink_name{1} sprintf('/train/spectral_%02d.mat',5)]);
train=[a.spectral;b.spectral;c.spectral;d.spectral;e.spectral];
s=size(a.spectral,1);
s1=size(b.spectral,1);
s2=size(c.spectral,1);
s3=size(d.spectral,1);

s4=size(e.spectral,1);
for i=1:s
    train_Lable(i)=1;
end
for i=s+1:s+s1
     train_Lable(i)=2;
end
for i=s+s1+1:s+s1+s2
     train_Lable(i)=3;
end
for i=s+s1+s2+1:s+s1+s2+s3
     train_Lable(i)=4;
end
for i=s+s1+s2+s3+1:s+s1+s2+s3+s4
     train_Lable(i)=5;
end
train_Lable=train_Lable';
train_Lable = categorical(train_Lable);
 train(:,34:36)=0;
 trainimg=[]

 for i=1:size(train,1)
z=reshape(train(i,:), [6 6 1 size(train(1,:),1) ]);
z=z';
trainimg(:,:,1,i)=z;
 end
%   trainimg=(reshape(train, [6 6  1 size(train,1)  ]));
 
% img_train=[];
% trainimg = zeros([6 6 length(train)]);
% %%%%append three zero
% for i=1:length(train)
%     img_train(i,:)=[train(i,:) 0 0 0 ];
%     trainimg(:,:,i)=reshape(img_train(i,:),[6 6 1 1 ]);
%     trainimg(:,:,i)=trainimg(:,:,i)';
% end 
% %%%%load and process test data
a1=load([root_path ink_name{1} sprintf('/test/spectral_%02d.mat',1)]);
b1=load([root_path ink_name{1} sprintf('/test/spectral_%02d.mat',2)]);
c1=load([root_path ink_name{1} sprintf('/test/spectral_%02d.mat',3)]);
d1=load([root_path ink_name{1} sprintf('/test/spectral_%02d.mat',4)]);
e1=load([root_path ink_name{1} sprintf('/test/spectral_%02d.mat',5)]);
test=[a1.x;b1.x;c1.x;d1.x;e1.x];
m=size(a1.x,1);
m1=size(b1.x,1);
m2=size(c1.x,1);
m3=size(d1.x,1);
m4=size(e1.x,1);
for i=1:m
    test_Lable(i)=1;
end
for i=m+1:m+m1
     test_Lable(i)=2;
end
for i=m+m1+1:m+m1+m2
     test_Lable(i)=3;
end
for i=m+m1+m2+1:m+m1+m2+m3
     test_Lable(i)=4;
end
for i=m+m1+m2+m3+1:m+m1+m2+m3+m4
     test_Lable(i)=5;
end
test_Lable=test_Lable';
test_Lable = categorical(test_Lable);
test(:,34:36)=0;
for i=1:size(test,1)
z=reshape(test(i,:), [6 6 1 size(test(1,:),1) ]);
z=z';
testimg(:,:,1,i)=z;
 end
%  testimg=(reshape(test, [6 6 1 size(test,1) ]));
%  XVal=(reshape(val_X,[ 6 6 1 size(val_X,1) ]));
% img_test=[];
% testimg = zeros([6 6 length(test)]);
% %%%%append three zero
% for i=1:length(test)
%     img_test(i,:)=[test(i,:) 0 0 0 ];
%     testimg(:,:,i)=reshape(img_test(i,:),[6 6 1 1 ]);
%     testimg(:,:,i)=testimg(:,:,i)'; 
% end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model 1
%%%%%%%%%%%%%%%% model 1  sgdm -99.5  adam-99.7 Rmsprop99.8
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
%      batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%      maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model2
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    model II %
%%%%% model 2 sgdm=99.89, adam=99.9,rmsprop=99.89)
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%       maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%   model 3   III
%%%%%% model 3 (sgdm=99.91  , adam =99.95  ,Rmsprop =99.92 ) 

layers = [
    imageInputLayer([6 6 1],"Name","imageinput")
    convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([3 3],24,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
      maxPooling2dLayer([2 2],"Name","maxpool1","Stride",[2,2],"Padding","same")
     dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(5,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% %%%%%%%%%%%%%%%   model 4     
%%%%%%%%%%%%%% model 4 sgdm=99.95  adam=99.87   , rmsprop=99.92)
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],24,"Name","conv3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],32,"Name","conv4","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_4")
%     reluLayer("Name","relu_4")
%     maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% %%%%%%%%%%%%%%%%%%%%%   model5
%%%%%%%%%%%%% model sgdm=99.94   ,adam =99.91  ,rmsprop=99.88)
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],24,"Name","conv3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],32,"Name","conv4","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_4")
%     reluLayer("Name","relu_4")
%     convolution2dLayer([3 3],40,"Name","conv5","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_5")
%     reluLayer("Name","relu_5")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% % %%%%%%%%%%%%%%%%%%%%%%%%% model 6
%%%%%%%% model 6 ( sgdm=99.5 ,adam=99.87   rmsprop=99.48 )

% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],8,"Name","conv1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],16,"Name","conv2","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],24,"Name","conv3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],32,"Name","conv4","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_4")
%     reluLayer("Name","relu_4")
%     convolution2dLayer([3 3],40,"Name","conv5","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_5")
%     reluLayer("Name","relu_5")
%     convolution2dLayer([3 3],48,"Name","conv6","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_6")
%     reluLayer("Name","relu_6")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%% model1
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],12,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],24,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%model2
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],64,"Name","conv1","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],64,"Name","conv2","Padding","same","Stride",[2 2])
%     batchNormalizationLayer("Name","batchnorm_2")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],128,"Name","conv3","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_3")
%     reluLayer("Name","relu_3")
%     convolution2dLayer([3 3],128,"Name","conv4","Padding","same","Stride",[2 2])
%     batchNormalizationLayer("Name","batchnorm_4")
%     reluLayer("Name","relu_4")
%     convolution2dLayer([3 3],256,"Name","conv5","Padding","same")
%     batchNormalizationLayer("Name","batchnorm_5")
%     reluLayer("Name","relu_5")
%     convolution2dLayer([3 3],256,"Name","conv6","Padding","same","Stride",[2 2])
%     batchNormalizationLayer("Name","batchnorm_6")
%     reluLayer("Name","relu_6")
%    dropoutLayer
%    fullyConnectedLayer(5,'Name','fc')
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%%%%%%%%%%%%%%%model3
%  layers = [
%     imageInputLayer([6 6 1])
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(5)
%     softmaxLayer
%     classificationLayer];
%%%%%%%%%%%%%%%%  model 4
% layers = [imageInputLayer([6 6 1]);
%           convolution2dLayer(3,6);
%           reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
%           fullyConnectedLayer(5);
%           softmaxLayer();
%           classificationLayer()];

% options=trainingOptions('rmsprop',...
%     'MiniBatchSize',300, ...
%     'MaxEpochs',10, ...
%     'InitialLearnRate',.05, ...
%     'Shuffle','every-epoch', ...
%     'LearnRateDropPeriod',5,...
%     'LearnRateDropFactor',0.2,...
%     'Verbose',false, ...
%     'Plots','training-progress');

options=trainingOptions('sgdm', ...
    'MiniBatchSize',300, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',0.005, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(trainimg,train_Lable,layers,options);
testpreds = classify(net,testimg);
  
        accuracy = mean(testpreds == test_Lable);
        toc
        % Tabulate the results using a confusion matrix.
 confMat =  confusionmat(test_Lable,testpreds);
% 
% % Convert confusion matrix into percentage form
% confMat = bsxfun(@rdivide,confMat,sum(confMat,2));


% net1 = trainNetwork(trainimg,train_Lable,layers,options);
% testpreds1 = classify(net1,testimg);
%   
%         accuracy1 = mean(testpreds1 == test_Lable);
%         net2 = trainNetwork(trainimg,train_Lable,layers,options);
% testpreds2 = classify(net2,testimg);
%   
%         accuracy2 = mean(testpreds2 == test_Lable);
%         net3 = trainNetwork(trainimg,train_Lable,layers,options);
% testpreds3 = classify(net3,testimg);
%   
%         accuracy3 = mean(testpreds3 == test_Lable);
%         net4 = trainNetwork(trainimg,train_Lable,layers,options);
% testpreds4 = classify(net4,testimg);
%   
%         accuracy4 = mean(testpreds4 == test_Lable);