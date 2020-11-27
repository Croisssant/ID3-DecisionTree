% =-=-=-=-=-=-=-=-=-=-=-=-=-= Data Loading =-=-=-=-=-=-=-=-=-=-=-=-=-=
data = readtable('.\Data\Wine\refined_data.csv', 'PreserveVariableNames', true);   

data = data(randperm(size(data,1)), :);
tree = LearningTreeClassification(data(1:50,:), []);
DrawDecisionTree(tree, 'Wine Classification')
 
f1_score = Test_Accuracy(tree, data(51:101, :));

% [best_acc, best_tree] = Ten_Fold_CV(data)

training_x = data(1:120, 1:13);
training_y = data(1:120, 14);

testing_x = data(120:end, 1:13);
testing_y = table2array(data(120:end, 14));

% =-=-=-=-=-=-=-=-=-=-=-=-=-= Training Functions =-=-=-=-=-=-=-=-=-=-=-=-=-=

function [node] = LearningTreeClassification(data, tested_attributes)
    [class_1, class_2, class_3] = ClassCount(table2array(data(:, end)));

    total_pn = class_1 + class_2 + class_3;
    total_p = class_1;
    total_n = class_2 + class_3;
    
   
    node = struct('op', '', 'kids', [], 'class', [], 'attribute', [], 'threshold', []);

    if length(unique(table2array(data(:,end)))) == 1
        node.op = '';
        node.kids = {};
        if unique(table2array(data(:,end))) == 1
            node.class = 1;
        else
            node.class = 0;
        end
        node.attribute = 0;
        node.threshold = 0;
        
    else
    
        entropy_all = Entropy(class_1, class_2 + class_3);

        best_best_gain = 0;
        best_feature = 0;
        all_gain = [];
        best_best_threshold = 0;
        best_best_left = [];
        best_best_right = [];
        % tree = struct('op', data.Properties.VariableNames(1), 'kids', {}, 'threshold', best_threshold)
        for i = 1:13
            if ismember(i, tested_attributes)
                continue
            end
            [best_gain,  best_threshold, left_label, right_label, left_child, right_child] = Remainder(data, i, entropy_all, total_pn);
            all_gain(end+1) = best_gain;
            if best_gain > best_best_gain
                best_best_gain = best_gain;
                best_feature = i;
                best_best_threshold = best_threshold;
                best_best_left = left_child;
                best_best_right = right_child;
                best_left_label = left_label;
                best_right_label = right_label;
            end
        end

        tested_attributes = [tested_attributes, best_feature]

        if tested_attributes(end) == 0
            node.op = '';
            node.attribute = '';
            node.kids = {};
            node.threshold = '';
            if total_p > total_n
                node.class = 1;
            else
                node.class = 0;
            end
        else
            
            node.op = char(data.Properties.VariableNames(best_feature));
            node.attribute = best_feature;
            node.threshold = best_best_threshold;
            node.class = [];

            if size(best_best_left) ~= 0
                node.kids{1}.class = best_left_label
                node.kids{1} = LearningTreeClassification(best_best_left, tested_attributes);

            end
            if size(best_best_right) ~= 0
                node.kids{2}.class = best_right_label
                node.kids{2} = LearningTreeClassification(best_best_right, tested_attributes);
            end

        end

    end

end

function [a, b, c] = ClassCount(label)  
    a = 0;
    b = 0;
    c = 0;
 
    for i = 1:length(label)
        if label(i) == 1
            a = a + 1;
    
        elseif label(i) == 2
            b = b + 1;
        
        elseif label(i) == 3
            c = c + 1;
        end
    end
end
   
function entropy_val = Entropy(p, n)
     if (p == n)
         entropy_val = 1;
     elseif (n == 0 || p == 0)
         entropy_val = 0;
     else
        entropy_val = -p/(p+n) * log2(p/(p+n)) - n/(p+n) * log2(n/(p+n));
     end
end      

function [best_gain,  best_threshold, left_label, right_label, left_child, right_child] = Remainder(dataset, feature_no, entropy, total_pn)
    best_gain = 0;
    best_threshold = 0;
    left_label = [];
    right_label = [];
   
   
    [m, n] = size(dataset);
    
    for i = 1:m
        threshold = dataset{i, feature_no};
        
        left = [];
        right = [];
        temp_left = table();
        temp_right = table();
    
        for j = 1:m
            if dataset{j, feature_no} <= threshold
                left(end+1) = dataset{j, end};
                temp_left = [temp_left; dataset(j,:)];
            else
                right(end+1) = dataset{j, end};
                temp_right = [temp_right; dataset(j,:)];
            end
        end
        
        [class_1, class_2, class_3] = ClassCount(left);
        left_p = class_1;
        left_n = class_2 + class_3;
        entropy_left = Entropy(left_p, left_n);
        
        [class_1, class_2, class_3] = ClassCount(right);
        right_p = class_1;
        right_n = class_2 + class_3;
        entropy_right = Entropy(right_p, right_n );
        
        avg_i_ent = (((left_p + left_n)/total_pn) * entropy_left) + (((right_p + right_n)/total_pn) * entropy_right );
        current_gain = entropy - avg_i_ent;
        left_child = temp_left;
        right_child = temp_right;
        
        if current_gain > best_gain 
            best_gain = current_gain;
            best_threshold = threshold;
         	left_label = left;
            right_label = right;
            left_child = table();
            right_child = table();
            left_child = temp_left;
            right_child = temp_right;
        end
    end
end


% =-=-=-=-=-=-=-=-=-=-=-=-=-= Testing Function =-=-=-=-=-=-=-=-=-=-=-=-=-=

function f1_score = Test_Accuracy(tree, dataset)
    [m, n] = size(dataset);
    all_predicted = [];
    
    true_positive = 0;
    false_positive = 0;
    
    true_negative = 0;
    false_negative = 0;
     
    for i = 1:m
        predicted_val = TreePrediction(tree, dataset(i,:));
        all_predicted(end+1) = predicted_val;
    end
    
    for z = 1:m
        if (dataset{z, end} == 3 | dataset{z, end} == 2)
            actual_value = 0;
        else
            actual_value = 1;
        end
        
        if (all_predicted(z) == actual_value & actual_value == 1)
            true_positive = true_positive + 1;
        
        elseif (all_predicted(z) ~= actual_value & actual_value == 1)
            false_negative = false_negative + 1;
            
        elseif (all_predicted(z) == actual_value & actual_value == 0)
            true_negative = true_negative + 1;
        
        elseif (all_predicted(z) ~= actual_value & actual_value == 0)
            false_positive = false_positive + 1;
        end
            
    end
    
    precision = true_positive / (true_positive + false_positive);
    recall = true_positive / (true_positive + false_negative);
    
    f1_score = 2 * ((precision * recall) / (precision + recall));
    
    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 - Score: %.2f\n', f1_score);
            
end

function pred = TreePrediction(tree, dataset)
    if isempty(tree.op)
        pred = tree.class;
    else
        if dataset{:, tree.attribute} <= tree.threshold
            pred = TreePrediction(tree.kids{1, 1}, dataset);
        else
            pred = TreePrediction(tree.kids{1, 2}, dataset);
        end
    end
end

% =-=-=-=-=-=-=-=-=-=-=-=-=-= 10-Fold Cross Validation Function =-=-=-=-=-=-=-=-=-=-=-=-=-=
function [best_acc, best_tree] = Ten_Fold_CV(dataset)
    best_acc = 0;
    for i = 1:10
        data = dataset
        l = 0 + (i * 17)
        j = 17 + (i * 17)
        testing_fold = data(l:j, :);
        training_fold = data;
        training_fold(l:j, :) = [];
        if i == 9
            training_fold = data;
            testing_fold = data(l:end, :)
            training_fold(l:j, :) = [];
        end
        
        tree = LearningTreeClassification(training_fold, []);
        acc = Test_Accuracy(tree, testing_fold);
        
        if acc > best_acc
            best_acc = acc;
            best_tree = tree;
        end
    end
end