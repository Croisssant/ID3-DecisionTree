function [best_acc, best_tree] = Ten_Fold_CV(dataset)
    global number_nodes;
    [m, n] = size(dataset);
    fold_size = ceil(m/10);
    best_acc = 0;
    num_node = 0;
    
    for i = 0:9
        fprintf('\n========== This is Fold %d ==========\n', i);
        data = dataset;
        
        if i == 0
            l = 1;
            j = fold_size;
        
        else
            
            l = i * fold_size;
            l = l + 1;
            j = i * fold_size + fold_size;
        end
        
        

        if i == 9
            training_fold = data;
            testing_fold = data(l:end, :)
            training_fold(l:end, :) = [];
        else
            testing_fold = data(l:j, :);
            training_fold = data;
            training_fold(l:j, :) = [];
        end
        
        number_nodes = 0;
        tree = LearningTreeClassification(training_fold, []);
        DrawDecisionTree(tree, 'Wine Classification');
        acc = F1_Score_Test(tree, testing_fold);
        
        if i == 0
            best_acc = acc;
            num_node = number_nodes;
            best_tree = tree;
        end
        
        if acc > best_acc 
            best_acc = acc;
            num_node = number_nodes;
            best_tree = tree;
            
        elseif acc == best_acc & number_nodes < num_node
            best_acc = acc;
            num_node = number_nodes;
            best_tree = tree;
        end
    end
    
    fprintf('\n============================\n');
    fprintf('The best tree has F1-Score of %d and %d number of nodes.\n', best_acc, num_node);
    fprintf('============================\n');
    DrawDecisionTree(best_tree, 'Best Tree')
end
