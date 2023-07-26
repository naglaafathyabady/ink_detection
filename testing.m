clc
clear
close all
% test=load('E:\naglaa\ink\HE+0.5GAMMA+GAussian\newink+HE+0.5gamma +gaussian2 before\new\blue\ink_mix\two_ink\1_1\newspect\0001\newspect_01_05.mat');
% test=load("E:\naglaa\ink\HE+0.5GAMMA+GAussian\newink+HE+0.5gamma +gaussian2 before\new\blue\newspect\0001\newspect_05.mat")

%test=load("E:\naglaa\ink\HE+0.5GAMMA+GAussian\newink+HE+0.5gamma +gaussian2 before\new\blue\ink_mix\1_1_1_1\newspect\0001\newspect_02_03_04_05.mat");

%%load image   %%
disp('test image')
 [filename, pathname] = uigetfile( ...
{'*.mat','Image Files (*.mat)';
'*.*',  'All Files (*.*)'}, ...
'Select mat file');
test= load([pathname filename]);

%%%%%%%%%%%%%%%%%%%%%%%% load network
disp('model')
% [filename1, pathname1] = uigetfile( ...
% {'*.mat','Image Files (*.mat)';
% '*.*',  'All Files (*.*)'}, ...
% 'Select mat file');
% load([pathname1 filename1]);
 test=test.result;
%load('E:\naglaa\ink\HE+0.5GAMMA+GAussian\newink+HE+0.5gamma +gaussian2 before\new\blue\ink_mix\two_ink\blackmodelWithHE+0.5GAMMA+gaussian_2_97.4_III+sgdm) bef.mat');

load('E:\naglaa\ink\HE+0.5GAMMA+GAussian\newink+HE+0.5gamma +gaussian2 before\new\blue\ink_mix\two_ink\bluemodelWithHE+0.5GAMMA+gaussian_2_99.91_III+sgdm) bef.mat');
test(:,34:36)=0;
for i=1:size(test,1)
z=reshape(test(i,:), [6 6 1 size(test(1,:),1) ]);
z=z';
testimg(:,:,1,i)=z;
end
testpreds = classify(net,testimg);
c1=0;
c2=0;
c3=0;
c4=0;
c5=0;


for i=1:size(testpreds,1)
    if(testpreds(i)=='1')
        c1=c1+1;
    elseif(testpreds(i)=='2')
        c2=c2+1;
    elseif(testpreds(i)=='3')
        c3=c3+1;
    elseif(testpreds(i)=='4')
        c4=c4+1;
    elseif(testpreds(i)=='5')
        c5=c5+1;
        end
        end 
c1new=round(c1*100/size(testpreds,1));
c2new=round(c2*100/size(testpreds,1));
c3new=round(c3*100/size(testpreds,1));
c4new=round(c4*100/size(testpreds,1));
c5new=round(c5*100/size(testpreds,1));
a=[c1new c2new c3new c4new c5new];
b=['1' '2' '3' '4' '5'];
[newA newB]=swap(a,b);

%%%%%%%%%%%%%%%%%%%%%% five ink 

if(c1new>=14&c2new>=14&c3new>=14&c4new>=14&c5new>=14)
    disp('1&2&3&4&5   five ink    forgery ' )
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% four ink
elseif(c1new >=18&c2new>=18&c3new>=18&c4new>=18)
    disp('1&2&3&4   four ink    forgery')
elseif(c1new>=18&c2new>=18&c3new>=18&c5new>=18)
    disp('1&2&3&5   four ink     forgery')
elseif(c1new>=18&c2new>=18&c4new>=18&c5new>=18)
    disp('1&2&4&5   four ink     forgery')
elseif(c1new>=18&c3new>=18&c4new>=18&c5new>=18)
    disp('1&3&4&5   four ink     forgery')
elseif(c2new>=18&c3new>=18&c4new>=18&c5new>=18)
    disp('2&3&4&5   four ink    forgery')

    %%%%%%%%%%%%%%  three ink
elseif(c1new>=28&c2new>=28&c3new>=28)
    disp('1&2&3   three ink  forgery')
elseif(c1new>=28&c2new>=28&c4new>=28)
    disp('1&2&4   three ink    forgery')
elseif(c1new>=28&c2new>=28&c5new>=28)
    disp('1&2&5   three ink    forgery')
elseif(c1new>=28&c3new>=28&c4new>=28)
    disp('1&3&4   three ink     forgery')
elseif(c1new>=28&c3new>=28&c5new>=28)
    disp('1&3&5   three ink     forgery')
elseif(c2new>=28&c3new>=28&c4new>=28)
    disp('2&3&4    three ink     forgery')
elseif(c2new>=28&c3new>=28&c5new>=28)
    disp('2&3&5    three ink     forgery')
elseif(c1new>=28&c4new>=28&c5new>=28)
    disp('1&4&5    three ink     forgery')
elseif(c3new>=28&c4new>=28&c5new>=28)
    disp('3&4&5    three ink     forgery')
elseif(c2new>=28&c4new>=28&c5new>=28)
    disp('2&4&5    three ink     forgery')
%%%%%%%%%%%%%%%%%%%%%%%%%%%  two ink
elseif(c1new>=40&c2new>=40)
    disp('1&2   two ink    forgery')
elseif(c1new>=40&c3new>=40)
    disp('1&3   two ink    forgery')
elseif(c1new>=40&c4new>=40)
    disp('1&4   two ink    forgery')
elseif(c1new>=40&c5new>=40)
    disp('1&5   two ink    forgery')
elseif(c2new>=40&c3new>=40)
    disp('2&3   two ink    forgery')
elseif(c2new>=40&c4new>=40)
    disp('2&4   two ink    forgery')
elseif(c2new>=40&c5new>=40)
    disp('2&5   two ink     forgery')
elseif(c3new>=40&c4new>=40)
    disp('3&4   two ink     forgery')
elseif(c3new>=40&c5new>=40)
    disp('3&5   two ink      forgery')
elseif(c4new>=40&c5new>=40)
    disp('4&5   two ink      forgery')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  one ink
elseif(c1new>95)
    disp('1    one ink   not forgery ')
elseif(c2new>=95)  
    disp('2     one ink   not forgery')
elseif(c3new>=95)
    disp('3     one ink    not forgery')
elseif(c4new>=95)
    disp('4     one ink      not forgery')
elseif(c5new>=95)
    disp('5     one ink      not forgery')
else

disp(newB(1))
disp('&')
disp(newB(2))
disp(' forgery')
end
disp('locy image')
[filename, pathname] = uigetfile( ...
{'*.mat','Image Files (*.mat)';
'*.*',  'All Files (*.*)'}, ...
'Select mat file');
lx= load([pathname filename]);
disp('locx image')
 [filename, pathname] = uigetfile( ...
{'*.mat','Image Files (*.mat)';
'*.*',  'All Files (*.*)'}, ...
'Select mat file');
ly= load([pathname filename]);
disp('old image')
[filename1, pathname1] = uigetfile( ...
{'*.mat','Image Files (*.mat)';
'*.*',  'All Files (*.*)'}, ...
'Select mat file');
old=load([pathname1 filename1]);
figure,imshow(old.new);
newimage=restore(test(:,1),lx.locx,ly.locy,testpreds);

%%%%%%%%%%%%%%% grouth %%%%%%%%%%%%%%%
res=detect(old.new,lx.locx,ly.locy);
%%%%%%%%%%%%accuracy%%%%%%%%%%%%%%%%%
accuracy = mean(testpreds == (res)');
 cm = confusionmat(res,testpreds);
%%%%%%%%%%%%%%%% precision%%%%%%%%%%%%%
y = diag(cm) ./ sum(cm,2);
precision=mean(y)
%%%%%%%%%%%%%%%recall%%%%%%%%%%
 x= diag(cm) ./ sum(cm,1)';
recall=mean(x)
%%%%%%%%%%%%%%f1 score%%%%%%%%%%%%
F_score=2*recall*precision/(precision+recall)
%%%%%%%%%%%%%%kappa%%%%%%%%%%%%%%
kappa = Kapp(testpreds, res)
if(newimage==old.new)
    disp('ok')
else 
    disp('not')
end

function [a b]=swap(a,b)
for i=1:5
    for j=i+1:5
        if(a(i)<a(j))
            x=a(i);
            a(i)=a(j);
            a(j)=x;
            y=b(i);
            b(i)=b(j);
            b(j)=y;
        end
    end
end
end 
 