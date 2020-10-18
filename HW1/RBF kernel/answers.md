# Radial-basis function kernel (aka squared-exponential kernel).

### Gamma
gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and 
high values meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples 
selected by the model as support vectors.

#### C
The C parameter trades off correct classification of training examples against maximization of the decision function’s 
margin. For larger values of C, a smaller margin will be accepted if the decision function is better at classifying all 
training points correctly. A lower C will encourage a larger margin, therefore a simpler decision function, at the cost 
of training accuracy. In other words``C`` behaves as a regularization parameter in the SVM.