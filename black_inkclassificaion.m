%%% model 3 with 4 conv e=15 minpatch=300, rate.005 sgdm accuracy=82.27

clc
close all
clear
root_path = 'new/'; norm_path = 'new/';
ink_name = {'Blue', 'Black'}; numInks = length(ink_name);
a=load([root_path ink_name{2} sprintf('/train/spectral_%02d.mat',1)]);

b=load([root_path ink_name{2} sprintf('/train/spectral_%02d.mat',2)]);
c=load([root_path ink_name{2} sprintf('/train/spectral_%02d.mat',3)]);
d=load([root_path ink_name{2} sprintf('/train/spectral_%02d.mat',4)]);
e=load([root_path ink_name{2} sprintf('/train/spectral_%02d.mat',5)]);
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
a1=load([root_path ink_name{2} sprintf('/test/spectral_%02d.mat',1)]);
b1=load([root_path ink_name{2} sprintf('/test/spectral_%02d.mat',2)]);
c1=load([root_path ink_name{2} sprintf('/test/spectral_%02d.mat',3)]);
d1=load([root_path ink_name{2} sprintf('/test/spectral_%02d.mat',4)]);
e1=load([root_path ink_name{2} sprintf('/test/spectral_%02d.mat',5)]);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model 1 (I)  95.5%
%%%%%%%%%%%%%%%% model 1 93.9 sgdm - 94.4 adam-93.2 Rmsprop
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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model2
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    model II 97.3%
%%%%% model 2 sgdm=96.7, adam=97 ,rmsprop=96.6
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

% %%%%%%%%%%%%%%%%%%%%%%%%%%%   model 3   III (96.7 or 97.3)
%%%%%% model 3 (sgdm=  97.4, adam =97.3  ,Rmsprop =97.7 ) 
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
% %%%%%%%%%%%%%%%   model 4     97.7
%%%%%%%%%%%%%% model 4 sgdm=97.7  adam= 96.8  , rmsprop=97.8)
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
%     maxPooling2dLayer([2 2],"Name","maxpool1","Stride",[2,2],"Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% %%%%%%%%%%%%%%%%%%%%%   model5
%%%%%%%%%%%%% model sgdm=97.6   ,adam =97.4  ,rmsprop=96.6)
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
%     maxPooling2dLayer([2 2],"Name","maxpool","Stride",[2,2],"Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
% % %%%%%%%%%%%%%%%%%%%%%%%%% model 6
%%%%%%%% model 6 ( sgdm=97.3 ,adam=97.6 rmsprop= 97.3)

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
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% model 1conv without batch +94.9%
%  layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%       batchNormalizationLayer("Name","batchnorm_1")
%     reluLayer("Name","relu_1")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
% %      fullyConnectedLayer(50,"Name","fc1")
%     fullyConnectedLayer(5,"Name","fc2")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%% model  2conv  with out batch +97.3%
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model  3conv without batch 97.8%
%  layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%     convolution2dLayer([3 3],96,"Name","conv_3","Padding","same")
%     reluLayer("Name","relu_3")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%      dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model 4conv without batch+97.4
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%      maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     convolution2dLayer([3 3],96,"Name","conv_3","Padding","same")
%     reluLayer("Name","relu_3")
%    
% 
%     convolution2dLayer([3 3],128,"Name","conv_4","Padding","same")
%     reluLayer("Name","relu_4")
%     maxPooling2dLayer([2 2],"Name","maxpool2","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model 5conv without batch97.5
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%      maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same")
%     convolution2dLayer([3 3],96,"Name","conv_3","Padding","same")
%     reluLayer("Name","relu_3")
%    
% maxPooling2dLayer([2 2],"Name","maxpool3","Padding","same")
%     convolution2dLayer([3 3],128,"Name","conv_4","Padding","same")
%     reluLayer("Name","relu_4")
%     maxPooling2dLayer([2 2],"Name","maxpool4","Padding","same")
%     convolution2dLayer([3 3],160,"Name","conv_5","Padding","same")
%     reluLayer("Name","relu_5")
%     maxPooling2dLayer([2 2],"Name","maxpool5","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  model 6 +6conv without patch 97.6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  or 97
% layers = [
%     imageInputLayer([6 6 1],"Name","imageinput")
%     convolution2dLayer([3 3],32,"Name","conv_1","Padding","same")
%     reluLayer("Name","relu_1")
%     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
%     convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
%     reluLayer("Name","relu_2")
%      maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same")
%     convolution2dLayer([3 3],96,"Name","conv_3","Padding","same")
%     reluLayer("Name","relu_3")
%    
% maxPooling2dLayer([2 2],"Name","maxpool3","Padding","same")
%     convolution2dLayer([3 3],128,"Name","conv_4","Padding","same")
%     reluLayer("Name","relu_4")
%     maxPooling2dLayer([2 2],"Name","maxpool4","Padding","same")
%     convolution2dLayer([3 3],160,"Name","conv_5","Padding","same")
%     reluLayer("Name","relu_5")
%     maxPooling2dLayer([2 2],"Name","maxpool5","Padding","same")
%      convolution2dLayer([3 3],160,"Name","conv_6","Padding","same")
%     reluLayer("Name","relu_6")
%     maxPooling2dLayer([2 2],"Name","maxpool6","Padding","same")
%     dropoutLayer(0.5,"Name","dropout")
%     fullyConnectedLayer(5,"Name","fc")
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classoutput")];


%%%model2
%%%%%%%%%%%%%blackmodelWithHE+0.5GAMMA+gaussian_2_97.1
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
%%%%%%%%%%%%%%%%%%%%model3
%  layers = [
%     imageInputLayer([6 6 1])
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,128,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(5)
%     softmaxLayer
%     classificationLayer];
%%%%%%%%%%%%%%%%  model 4
%blackmodelWithHE+GAMMA0.5gaussian_2_91.8
% layers = [imageInputLayer([6 6 1]);
%           convolution2dLayer(3,32);
%           reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
%           fullyConnectedLayer(5);
%           softmaxLayer();
%           classificationLayer()];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     layers = [
%     imageInputLayer([6 6 1])
%     
%     convolution2dLayer(3,6,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,18,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,36,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer
%     fullyConnectedLayer(5)
%     softmaxLayer
%     classificationLayer];
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
 [net,info] = trainNetwork(trainimg,train_Lable,layers,options);
testpreds = classify(net,testimg);
  
        accuracy = mean(testpreds == test_Lable);
        % Tabulate the results using a confusion matrix.
confMat =  confusionmat(test_Lable,testpreds);

% Convert confusion matrix into percentage form
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% clc
% close all
% clear
% a=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\train\spectral_01.mat');
% 
% b=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\train\spectral_02.mat');
% c=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\train\spectral_03.mat');
% d=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\train\spectral_04.mat');
% e=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\train\spectral_05.mat');
% train=[a.spectral_1;b.spectral_2;c.spectral_3;d.spectral_4;e.spectral_5];
% s=size(a.spectral_1,1);
% s1=size(b.spectral_2,1);
% 
% s2=size(c.spectral_3,1);
% 
% s3=size(d.spectral_4,1);
% 
% s4=size(e.spectral_5,1);
% for i=1:s
%     train_Lable(i)=1;
% end
% for i=s+1:s+s1
%      train_Lable(i)=2;
% end
% for i=s+s1+1:s+s1+s2
%      train_Lable(i)=3;
% end
% for i=s+s1+s2+1:s+s1+s2+s3
%      train_Lable(i)=4;
% end
% for i=s+s1+s2+s3+1:s+s1+s2+s3+s4
%      train_Lable(i)=5;
% end
% train_Lable=train_Lable';
% train_Lable = categorical(train_Lable);
% %  trn  = zeros(size(train,1),36);
% %  trn(:,4:36)=train;
%  train(:,34:36)=0;
%  trainimg=[]
% 
%  for i=1:size(train,1)
% z=reshape(train(i,:), [6 6 1 size(train(1,:),1) ]);
% z=z';
% %%edge detection 
% % zz=PointDetection (z);
% % %% normalization %77.5
% %      fim=mat2gray(z);
% %      lnfim=localnormalize(fim,4,4);
% %      Iout=mat2gray(lnfim);
% %      I=normalize(Iout);
% trainimg(:,:,1,i)=z;
%  end
% %  for i=1:size(train,1)
% %      newtrain(i,1)=mat2cell(trainimg(:,:,1,i),6);
% %  end
% 
% % `1q    trainimg=(reshape(train, [6 6  1 size(train,1)  ]));
% %  
% % % img_train=[];
% % % trainimg = zeros([6 6 length(train)]);
% % % %%%%append three zero
% % % for i=1:length(train)
% % %     img_train(i,:)=[train(i,:) 0 0 0 ];
% % %     trainimg(:,:,i)=reshape(img_train(i,:),[6 6 1 1 ]);
% % %     trainimg(:,:,i)=trainimg(:,:,i)';
% % % end 
% % % %%%%load and process test data
% a1=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\test\spectral_01.mat');
% b1=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\test\spectral_02.mat');
% c1=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\test\spectral_03.mat');
% d1=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\test\spectral_04.mat');
% e1=load('E:\naglaa\ink\gaussian\gaussianinsecond\new\Black\test\spectral_05.mat');
% test=[a1.result;b1.result;c1.result;d1.result;e1.result];
% m=size(a1.result,1);
% m1=size(b1.result,1);
% m2=size(c1.result,1);
% m3=size(d1.result,1);
% m4=size(e1.result,1);
% for i=1:m
%     test_Lable(i)=1;
% end
% for i=m+1:m+m1
%      test_Lable(i)=2;
% end
% for i=m+m1+1:m+m1+m2
%      test_Lable(i)=3;
% end
% for i=m+m1+m2+1:m+m1+m2+m3
%      test_Lable(i)=4;
% end
% for i=m+m1+m2+m3+1:m+m1+m2+m3+m4
%      test_Lable(i)=5;
% end
% test_Lable=test_Lable';
% test_Lable = categorical(test_Lable);
%  tst  = zeros(size(test,1),36);
% %   tst(:,4:36)=test;
% test(:,34:36)=0;
% testimg=[];
% for i=1:size(test,1)
% z=reshape(test(i,:), [6 6 1 size(test(1,:),1) ]);
% z=z';
% % se=strel('disk',5);
% % z = imdilate(z,se);
% %%edge detection 
% % zz=PointDetection (z);
%  %% normalization %77.5
% %      fim=mat2gray(z);
% %      lnfim=localnormalize(fim,4,4);
% %      Iout=mat2gray(lnfim);
% %      I=normalize(Iout);
% testimg(:,:,1,i)=z;
% end
% %  for i=1:size(test,1)
% %      newtest(i,1)=mat2cell(testimg(:,:,1,i),6);
% %  end
% %  testimg=(reshape(test, [6 6 1 size(test,1) ]));
% %  XVal=(reshape(val_X,[ 6 6 1 size(val_X,1) ]));
% % img_test=[];
% % testimg = zeros([6 6 length(test)]);
% % %%%%append three zero
% % for i=1:length(test)
% %     img_test(i,:)=[test(i,:) 0 0 0 ];
% %     testimg(:,:,i)=reshape(img_test(i,:),[6 6 1 1 ]);
% %     testimg(:,:,i)=testimg(:,:,i)'; 
% % end 
% %%%%%%%% model1
% % layers = [
% %     imageInputLayer([6 6 1],"Name","imageinput")
% %     convolution2dLayer([3 3],256,"Name","conv_1","Padding","same")
% %     reluLayer("Name","relu_1")
% %     convolution2dLayer([3 3],256,"Name","conv_2","Padding","same")
% %     reluLayer("Name","relu_2")
% %     maxPooling2dLayer([2 2],"Name","maxpool","Padding","same")
% %     fullyConnectedLayer(5,"Name","fc")
% %     softmaxLayer("Name","softmax")
% %     classificationLayer("Name","classoutput")];
% %%%model2
% % layers = [
% %     imageInputLayer([6 6 1],"Name","imageinput")
% %     convolution2dLayer([3 3],96,"Name","conv1","Padding","same")
% %     batchNormalizationLayer("Name","batchnorm_1")
% %     reluLayer("Name","relu_1")
% %     convolution2dLayer([3 3],96,"Name","conv2","Padding","same","Stride",[2 2])
% %     batchNormalizationLayer("Name","batchnorm_2")
% %     reluLayer("Name","relu_2")
% %     convolution2dLayer([3 3],128,"Name","conv3","Padding","same")
% %     batchNormalizationLayer("Name","batchnorm_3")
% %     reluLayer("Name","relu_3")
% %     convolution2dLayer([3 3],128,"Name","conv4","Padding","same","Stride",[2 2])
% %     batchNormalizationLayer("Name","batchnorm_4")
% %     reluLayer("Name","relu_4")
% %     convolution2dLayer([3 3],128,"Name","conv5","Padding","same")
% %     batchNormalizationLayer("Name","batchnorm_5")
% %     reluLayer("Name","relu_5")
% %     convolution2dLayer([3 3],256,"Name","conv6","Padding","same","Stride",[2 2])
% %     batchNormalizationLayer("Name","batchnorm_6")
% %     reluLayer("Name","relu_6")
% %    dropoutLayer
% %    fullyConnectedLayer(5,'Name','fc')
% %     softmaxLayer("Name","softmax")
% %     classificationLayer("Name","classoutput")];
% %%%%%%%%%%%%%%%%%%%%%model3
% %  layers = [
% %     imageInputLayer([6 6 1])
% %     
% %     convolution2dLayer(3,6,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     
% %     maxPooling2dLayer(2,'Stride',2)
% %     
% %     convolution2dLayer(3,18,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     
% %     maxPooling2dLayer(2,'Stride',2)
% %     
% %     convolution2dLayer(3,36,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     dropoutLayer
% %     fullyConnectedLayer(5)
% %     softmaxLayer
% %     classificationLayer];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% model 4
% % layers = [imageInputLayer([6 6 1]);
% %           convolution2dLayer(3,48);
% %           reluLayer();
% %           maxPooling2dLayer(2,'Stride',2);
% %           fullyConnectedLayer(5);
% %           softmaxLayer();
% %           classificationLayer()];
% %%%%%%%%%%%%%%%%%%%%%%%% model 5
% 
% % layers = [
% %     imageInputLayer([6 6 1])
% % 	
% %     convolution2dLayer(3,96,'Padding',1)
% %     batchNormalizationLayer
% %     reluLayer
% % 	
% %     maxPooling2dLayer(2,'Stride',2)
% % 	
% %     convolution2dLayer(3,128,'Padding',1)
% %     batchNormalizationLayer
% %     reluLayer
% % 	
% %     maxPooling2dLayer(2,'Stride',2)
% % 	
% %     convolution2dLayer(3,256,'Padding',1)
% %     batchNormalizationLayer
% %     reluLayer
% % 	
% %     fullyConnectedLayer(5)
% %     softmaxLayer
% %     classificationLayer];
% %%%%%%%%%%%%%%%%%%%%
% layers = [imageInputLayer([6 6 1]);
%           convolution2dLayer(3,48);
%           reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
%           fullyConnectedLayer(5);
%           softmaxLayer();
%           classificationLayer()];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % options=trainingOptions('sgdm', ...
% %     'MiniBatchSize',300, ...
% %     'MaxEpochs',15, ...
% %     'InitialLearnRate',0.005, ...
% %     'Shuffle','every-epoch', ...
% %     'ValidationFrequency',5, ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
% % options = trainingOptions('sgdm', ...
% %     'Momentum', 0.9, ...
% %     'MaxEpochs',10, ...
% %     'Shuffle','every-epoch', ...
% %     'LearnRateDropPeriod',10, ...
% %       'LearnRateDropFactor', 0.9, ...
% %       'LearnRateSchedule', 'piecewise', ...
% %     'MiniBatchSize', 300, ...
% %      'InitialLearnRate',1e-3, ...
% %     'ValidationFrequency',30, ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
% options=trainingOptions('sgdm', ...
%     'MiniBatchSize',300, ...
%     'MaxEpochs',15, ...
%     'InitialLearnRate',0.005, ...
%     'Shuffle','every-epoch', ...
%     'ValidationFrequency',5, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% % layers = [ ...
% %     imageInputLayer([6 6 1])
% %     convolution2dLayer([5 5],10)
% %     reluLayer
% %     fullyConnectedLayer(5)
% %     softmaxLayer
% %     classificationLayer];
% 
% % options=trainingOptions('sgdm',...
% %     'MiniBatchSize',300, ...
% %     'MaxEpochs',200, ...
% %     'InitialLearnRate',.005, ...
% %     'Shuffle','every-epoch', ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
% % options=trainingOptions('adam', ...
% %     'MiniBatchSize',300, ...
% %     'MaxEpochs',5, ...
% %     'InitialLearnRate',0.001, ...
% %     'Shuffle','every-epoch', ...
% %     'ValidationFrequency',5, ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
% %%%%%%%%%%%%%%%%%%%DGNETwork
% % layers = [
% %     imageInputLayer([6 6 1])
% %     
% %     convolution2dLayer(3,16,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer('Name','relu_1')
% %     
% %     convolution2dLayer(3,32,'Padding','same','Stride',2)
% %     batchNormalizationLayer
% %     reluLayer
% %     convolution2dLayer(3,32,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer
% %     
% %     additionLayer(2,'Name','add')
% %     
% %     averagePooling2dLayer(2,'Stride',2)
% %     fullyConnectedLayer(5)
% %     softmaxLayer
% %     classificationLayer];
% % %%%%%%%%%%%%%%%%%% plot graph
% % lgraph = layerGraph(layers);
% % figure
% % plot(lgraph)
% % %%%%%%%%%%%%%%%%%%%%%%
% % skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
% % lgraph = addLayers(lgraph,skipConv);
% % figure
% % plot(lgraph)
% % %%%%%%%%%%%%%%%%%%%%%%%
% % lgraph = connectLayers(lgraph,'relu_1','skipConv');
% % 
% % lgraph = connectLayers(lgraph,'skipConv','add/in2');
% % figure
% % plot(lgraph);
% %%%%%%%%%%%%%%%%%%%%%%5
% % options = trainingOptions('sgdm', ...
% %    'MiniBatchSize',300,... 
% %     'MaxEpochs',15, ...
% %      'InitialLearnRate',0.001, ...
% %     'Shuffle','every-epoch', ...
% %     'ValidationFrequency',5, ...
% %     'ValidationFrequency',100, ...
% %     'Verbose',false, ...
% %     'Plots','training-progress');
% 
% 
% 
%   net = trainNetwork(trainimg,train_Lable,layers,options);
% testpreds = classify(net,testimg);
%         accuracy = mean(testpreds == test_Lable);
%         % Tabulate the results using a confusion matrix.
% confMat =  confusionmat(test_Lable,testpreds);
% 
% % Convert confusion matrix into percentage form
% confMat = bsxfun(@rdivide,confMat,sum(confMat,2));