function RMSE = RMSE_Test(tree, test_set)
    test_label = test_set{:,6};
    test_label = test_label.';

    [o, p] = size(test_set);
    for i = 1:o
        predict_label(i) = TreePrediction(tree, test_set);
    end
    
	MSE = mean((test_label - predict_label).^2)
	RMSE = sqrt(MSE)
end

function predict = TreePrediction(tree, test)
    if isempty(tree.op)
        predict = tree.class;
    else
        if test{:,tree.op} == tree.threshold
            predict = TreePrediction(tree.kids{1,1},test);
        else
            predict = TreePrediction(tree.kids{1,2},test);
        end
    end
end