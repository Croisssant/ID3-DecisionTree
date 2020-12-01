data = readtable('.\Data\Airfoil_self_noise\airfoil_self_noise.dat');
training = data(1:1000, :);
testing = data(1000:end, :);
[m,n] = size(training);
[o,p] = size(testing);
%---------------------------------------------------------------------------
tree = struct('op', 0, 'kids', [], 'class', [], 'threshold',[]);
[tree,count] = Decision_Tree(training,1);
DrawDecisionTree(tree,"regression");

test_label = testing{:,6};
test_label = test_label.';
predict_label = [];

for i = 1:o
    predict_label(i) = Test(tree, testing);
end

% RMSE = test_label - predict_label;
% RMSE = RMSE.^2;
MSE = mean((test_label - predict_label).^2)
RMSE = sqrt(MSE)
count

function predict = Test(tree, test)
    if isempty(tree.op)
        predict = tree.class;
    else
        if test{:,tree.op}== tree.threshold
            predict = Test(tree.kids{1,1},test);
        else
            predict = Test(tree.kids{1,2},test);
        end
    end
end

function [node,count] = Decision_Tree(training,count)
    count = count +1;
    [m,n] = size(training);
    SD = std(training{:,n});
    
    % the stopping condition 
    if m <= 50
       label = mean(training{:,end});
       node.class = label;
       node.kids = {}; 
       node.op = {};
       fprintf("number of rows: %d, label: %d \n",m,label);
       return 
    else
        node.class = 0;
    end
    
    % for all features, choose best gain and threshold
    best_SDR = 0;
    best_threshold = 0;
    feature = 0;

    for x = 1:n-1
        [threshold, SDR] = Feature_entrophy(SD,training,x);
        if SDR > best_SDR
           best_SDR = SDR;
           best_threshold = threshold;
           feature = x;
        end
    end

    node.op = feature;

    node.threshold = best_threshold;
    [left_t,right_t] = slice_train(training,feature,best_threshold);
    
    [node.kids{1,1},count_l] = Decision_Tree(left_t,count);
    [node.kids{1,2},count_r] = Decision_Tree(right_t,count);
    
    count = count + count_l + count_r;
end

% Feature_entrophy function calculate the features' gain(SDR)and its
% threshold
function [threshold,SDR] = Feature_entrophy(SD,training,feature)
    [m,n] = size(training);
    C = unique(training{:,feature}); %classify all data into unique values
    [a,b] = size(C);
    D = zeros(a,2);
    D(:,1) = C; %use to count how many training set belongs to each unique value
    y = []; %use to store the labels of each dataset into their unique column
    
    for i = 1:m %loop for all training row
       for j = 1:a %loop for all unique values 
           if training{i,feature} == D(j,1) %add count and store their labels accordingly
               D(j,2) = D(j,2) +1;
               y(D(j,2),j) = training{i,n};
               break
           end
       end
    end

	y(y==0)=NaN; %avoid calculating 0 into SD
    sd = nanstd(y,[],1); %calculate SD for all unique value
    [min_sd, min_index] = min(sd); %get best threshold
    threshold = D(min_index,1);
    SDR = SD - SDA(D,sd,training); %get SDR of the feature
end

% SDA takes the unique value matrix and the sd of all unique value
% to get entropy of the features
function st_f = SDA(D,sd,training)
    st_f = 0;
    [m,n] = size(training);
    [a,b] = size(D);
    for i = 1:a
        st_f = st_f + D(i,2)/m * sd(i);
    end
end

function [left_training, right_training] = slice_train(training,feature,threshold)
    [y,z] = size(training);
    left_training = [];
    right_training = [];
    for i = 1:y
        if training{i,feature} == threshold
            left_training = [left_training; training{i,:}];
        else
            right_training = [right_training; training{i,:}];
        end
    end
    left_training = array2table(left_training);
    right_training = array2table(right_training);
end



