% JN Kather 2018

function [lgraph,imageInputSize,networkType] = getAndModifyNet(nnmodel,hyperparam,numOutputClasses)

% load pre-trained network model for transfer learning
switch nnmodel
    case 'vgg19'
        rawnet = vgg19;   
        networkType = 'series';
    case 'vgg16'
        rawnet = vgg16;
        networkType = 'series';
    case 'alexnet'
        rawnet = alexnet;
        networkType = 'series';
    case 'inceptionv3'
        rawnet = inceptionv3;
        networkType = 'DAG';
        layersForRemoval = {'predictions', 'predictions_softmax','ClassificationLayer_predictions'};
        layersForReconnection = {'avg_pool','fc'};
    case 'googlenet'
        rawnet = googlenet;
        networkType = 'DAG';
        layersForRemoval = {'loss3-classifier','prob','output'};
        layersForReconnection = {'pool5-drop_7x7_s1','fc'};
    case 'resnet18' 
        rawnet = resnet18;
        networkType = 'DAG';
        layersForRemoval = {'fc1000', 'prob','ClassificationLayer_predictions'};
        layersForReconnection = {'pool5','fc'};
    case 'resnet50' 
        rawnet = resnet50;
        networkType = 'DAG';
        layersForRemoval = {'ClassificationLayer_fc1000', 'fc1000_softmax','fc1000'};
        layersForReconnection = {'avg_pool','fc'};
    case 'resnet101'
        rawnet = resnet101;
        networkType = 'DAG';
        layersForRemoval = {'fc1000', 'prob','ClassificationLayer_predictions'};
        layersForReconnection = {'pool5','fc'};
    case 'squeezenet'
        rawnet = squeezenet;
        networkType = 'DAG';   
        layersForRemoval = {'pool10', 'prob','ClassificationLayer_predictions'};
        layersForReconnection = {'relu_conv10','fc'};
    case 'inceptionresnetv2'
        rawnet = inceptionresnetv2;
        networkType = 'DAG';   
        layersForRemoval = {'predictions', 'predictions_softmax','ClassificationLayer_predictions'};
        layersForReconnection = {'avg_pool','fc'};
    otherwise
        error('wrong network model specified');
end

% prune and rewire network
switch networkType
    case 'series' % e.g. alexnet
        lgraph = rawnet.Layers;
        % freeze shallow layers
        freezeIndex = 1:(numel(lgraph)-hyperparam.hotLayers);
        lgraph(freezeIndex) = freezeWeights(lgraph(freezeIndex));
        % overwrite penultimate and last layer
        lgraph(end-2) = fullyConnectedLayer(numOutputClasses,'Name','fc',...
            'WeightLearnRateFactor',hyperparam.learnRateFactor,...
            'BiasLearnRateFactor',hyperparam.learnRateFactor);
        lgraph(end) = classificationLayer;
        imageInputSize = lgraph(1).InputSize(1:2);
    case 'DAG' % e.g. googlenet
        % freeze shallow layers
        lgraph = layerGraph(rawnet); % convert network to layer graph
        layers = lgraph.Layers;      % extract layers
        connections = lgraph.Connections; % exctract connections
        freezeIndex = 1:(numel(layers)-hyperparam.hotLayers);
        layers(freezeIndex) = freezeWeights(layers(freezeIndex));
        lgraph = createLgraphUsingConnections(layers,connections);
        % remove old layers
        lgraph = removeLayers(lgraph,layersForRemoval);
        % add new layers and connect
        newLayers = [
            fullyConnectedLayer(numOutputClasses,'Name','fc',...
            'WeightLearnRateFactor', hyperparam.learnRateFactor,...
            'BiasLearnRateFactor', hyperparam.learnRateFactor),...
            softmaxLayer('Name','softmax'),...
            classificationLayer('Name','classoutput')];
        lgraph = addLayers(lgraph,newLayers);
        lgraph = connectLayers(lgraph,layersForReconnection{1},layersForReconnection{2});
        imageInputSize = lgraph.Layers(1).InputSize(1:2);
    otherwise, error('undefined network type');
end
end
