### How does the value of C affects the boundaries?
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. For large 
values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting 
all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a 
larger-margin separating hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, you 
should get misclassified examples, often even if your training data is linearly separable
### What happens when C i very low? What about when it is very high
When c is very low boundaries are missing, when it is very high is unuseful because boundaries do not change after a 
certain threshold. 
### decision_function_shape parameter
- What is its default value? Is it consistent to the result?
default=’ovr’
- Is it different to one-versus-one policy?
Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or 
the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification.