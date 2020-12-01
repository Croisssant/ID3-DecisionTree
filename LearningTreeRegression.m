function [node] = LearningTreeRegression(training)
    global number_nodes;
    number_nodes = number_nodes + 1;
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
        [threshold, SDR] = Feature_Entropy(SD,training,x);
        if SDR > best_SDR
           best_SDR = SDR;
           best_threshold = threshold;
           feature = x;
        end
    end

    node.op = feature;

    node.threshold = best_threshold;
    [left_t,right_t] = slice_train(training,feature,best_threshold);
    
    node.kids{1,1} = LearningTreeRegression(left_t);
    node.kids{1,2} = LearningTreeRegression(right_t);
    
end

% Feature_Entropy function calculate the features' gain(SDR)and its
% threshold
function [threshold,SDR] = Feature_Entropy(SD,training,feature)
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



