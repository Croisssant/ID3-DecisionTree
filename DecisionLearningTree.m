% Classification data
data = readtable('.\Data\Wine\refined_data.csv', 'PreserveVariableNames', true); 
data = data(randperm(size(data,1)), :);

train_data = data(1:100, :);
test_data = data(101:end, :);

% Regression data
reg_data = readtable('.\Data\Airfoil_self_noise\airfoil_self_noise.dat');
reg_data = reg_data(randperm(size(reg_data,1)), :);

reg_train_data = reg_data(1:1000, :);
reg_test_data = reg_data(1000:end, :);

	
i = 1;
while i == 1;
    fprintf('Would you like to run Classification or Regression decision tree?\n');
    fprintf('0 - Regression\n');
    fprintf('1 - Classification\n');
    fprintf('2 - Exit\n');
    user_ans = input('>> ');
 
    if user_ans == 0
        Decision_Learning_Tree(reg_train_data, reg_test_data, reg_data, user_ans)
        continue;
        
    elseif user_ans == 1
        Decision_Learning_Tree(train_data, test_data, data, user_ans)
        continue;
        
    elseif user_ans == 2
        i = 0;
        return;
        
    else
        disp('Please provide a valid input')
        continue;
    
    end
end


function Decision_Learning_Tree(train_data, test_data, data, flag)
    
    if flag == 0
        fprintf('Would you like to run 10-Fold Crossvalidation or No?\n');
        fprintf('0 - NO Cross Validation\n');
        fprintf('1 - 10-Fold Cross Validation\n');        
        input_ans = input('>> ');
        
        if input_ans == 0
            global number_nodes;
            number_nodes = 0;

            tree = LearningTreeRegression(train_data);
            DrawDecisionTree(tree, 'Air Foil Noise')
            rmse = RMSE_Test(tree, test_data);
               
        elseif input_ans == 1
            global number_nodes;
            number_nodes = 0;
            [best_acc, best_tree] = Ten_Fold_CV(data, flag);
        
        else
            disp('Please provide a valid input')
            return;
        
        end
        
    elseif flag == 1
        fprintf('Would you like to run 10-Fold Crossvalidation or No?\n');
        fprintf('0 - NO Cross Validation\n');
        fprintf('1 - 10-Fold Cross Validation\n');        
        input_ans = input('>> ');
        
        if input_ans == 0
            global number_nodes;
            number_nodes = 0;

            tree = LearningTreeClassification(train_data, []);
            DrawDecisionTree(tree, 'Wine Classification')
            f1_score = F1_Score_Test(tree, test_data);
               
        elseif input_ans == 1
            global number_nodes;
            number_nodes = 0;
            [best_acc, best_tree] = Ten_Fold_CV(data);
        
        else
            disp('Please provide a valid input')
            return;
        
        end
             
    else
        disp('Please provide a valid input')
        return;
        
    end
end
