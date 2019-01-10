# Machine-learning  

# http://wiki.octave.org/Octave_for_Microsoft_Windows  

# https://www.coursera.org/learn/machine-learning/lecture


# equation  
1  
![image](https://user-images.githubusercontent.com/16419246/50993929-73d3c900-14e0-11e9-9a20-baa62c393ea1.png)  

J = (sum((theta * x) - y) .^2 ) / (2 * m)  


2  
![image](https://user-images.githubusercontent.com/16419246/50994019-b2698380-14e0-11e9-8ef5-9ccbfce791cd.png)  

h = theta(1) + (theta(2) * x)  


3  
![image](https://user-images.githubusercontent.com/16419246/50994097-f2306b00-14e0-11e9-9d59-bcd8977b7ac0.png)  

theta_zero = theta(1) - ((alpha * (1/m)) * sum(h - y))  
theta_one = theta(2) - ((alpha * (1/m)) * (sum(h - y) .* x))  

theta = [theta_zero ; theta_one]  

THEN =>   J_history(iter) = computeCost(X, y, theta)   % for iter = 1:num_iters  
          disp(min(J_history))  
          
          
