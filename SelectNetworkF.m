function [network,inputSize] = SelectNetworkF(NameNetwork)
%% select type of network
switch NameNetwork
    case 'Squeezenet'   % DAG Network
       network=squeezenet;
     case 'Alexnet'     % Series Network
       network=alexnet;
     case 'Googlenet'   % DAG Network
       network=googlenet;
     case 'Resnet18'    % DAG Network
       network=resnet18;
     case 'Resnet50'    % DAG Network
       network=resnet50;
     case 'Resnet101'   % DAG Network
       network=resnet101;
     case 'Vgg16'       % Series Network 
       network=vgg16;
     case 'Vgg19'       % Series Network
       network=vgg19;
     case 'Inceptionv3' % DAG Network
       network=inceptionv3;
     case 'Inceptionresnetv2' % DAG Network
       network=Inceptionresnetv2;
end
%% Network input image size
 inputSize = network.Layers(1, 1).InputSize(1:2);