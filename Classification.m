data = readtable('.\Data\Wine\refined_data.csv', 'PreserveVariableNames', true);   

% data = T(randperm(size(T,1)), :);

training_x = data(1:120, 1:13);
training_y = data(1:120, 14);

testing_x = data(120:end, 1:13);
testing_y = table2array(data(120:end, 14));
   
[class_1, class_2, class_3] = ClassCount(table2array(data(:, 14)))

total_pn = class_1 + class_2 + class_3
entropy_all = Entropy(class_1, class_2 + class_3)
first_col_x = table2array(training_x(1:60, 1));
first_col_y = table2array(training_y(1:60, 1));
[best_gain,  best_threshold] = Remainder(first_col_x, first_col_y, entropy_all, total_pn)

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
        


function [best_gain,  best_threshold] = Remainder(feature, label, entropy, total_pn)
    best_gain = 0;
    best_threshold = 0;
    for i = 1:length(feature)
        treshold = feature(i);
        
        left = [];
        right = [];
        for j = 1:length(feature)
            if feature(j) <= treshold
                left(end+1) = label(j);
            else
                right(end+1) = label(j);
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
        
        avg_i_ent = ((left_p + left_n)/total_pn * entropy_left) + ((right_p + right_n)/total_pn * entropy_right );
        current_gain = entropy - avg_i_ent;
        
        if current_gain > best_gain 
            best_gain = current_gain;
            best_threshold = treshold;
        end
    end
end

    