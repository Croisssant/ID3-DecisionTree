T = readtable('.\Data\Wine\refined_data.csv', 'PreserveVariableNames', true)

data = T(randperm(size(T,1)), :)

training_x = data(1:120, 1:13)
training_y = data(1:120, 14)

testing_x = data(120:end, 1:13)
testing_y = data(120:end, 14)

function decision_tree = DECISION-TREE-LEARNING(features, labels)
    entropy = -p/(p+n) * log2(p/(p+n)) - n/(p+n) * log2(n/(p+n))
    