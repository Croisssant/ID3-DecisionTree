function f1_score = F1_Score_Test(tree, dataset)
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
    if ~isempty(tree.class)
        pred = tree.class;
    elseif size(tree.kids, 2) == 1
        pred = TreePrediction(tree.kids{1, 1}, dataset);
    else
        if dataset{:, tree.attribute} <= tree.threshold
            pred = TreePrediction(tree.kids{1, 1}, dataset);
        else
            pred = TreePrediction(tree.kids{1, 2}, dataset);
        end
    end
end