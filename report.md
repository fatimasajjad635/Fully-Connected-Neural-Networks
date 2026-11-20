# MSE 641 Assignment 4 Report

**Student Name:** [Fatima Sajjad]  
**Student ID:** [20718749]  
**Date:** [June 19th, 2025]

*Maximum 10 sentences total for all analysis sections below.*

## Final Results Summary

| Activation Function | Best L2 Regularization | Best Dropout Rate | Validation Accuracy | Test Accuracy |
|---------------------|------------------------|-------------------|---------------------|---------------|
| ReLU                | [0.01]               | [0.3]           | [0.5001610824742269]        | [0.5028139849743821] |
| Sigmoid             | [0.001]               | [0.3]           | [0.4998389175257732]        | [0.4971860150256179] |
| Tanh                | [0.01]               | [0.3]           | [0.4998389175257732]        | [0.4971860150256179] |

## Analysis (Maximum 10 sentences total)

### Effect of Activation Functions (2-3 sentences)
*
[Compared to Sigmoid and Tanh, ReLU outperformed both with slightly higher Validation Accuracy and Test Accuracy values. This is because of ReLU's capability to refrain from vanishing gradients. ReLU also advances into sparce activations which can assist in learning in deeper networks.]

### Effect of L2 Regularization (2-3 sentences)
*How did L2 regularization affect your model performance? Did it help prevent overfitting?*

[According to the results, models with a moderate L2 regularization seemed to help prevent overfitting without penalizing the weights, this suggests that ReLU with L2 value of 0.01 gave slightly better results.]. Sigmoid and Tanh with values of 0.01 and 0.001 respectively had similar Validation and Test Accuracy values.

### Effect of Dropout (2-3 sentences)
*How did dropout affect your results? What dropout rates worked best?*

[The Dropout supported with the reduction of co-adaptation of neurons as well as the improvment of generalization. All three models used a dropout rate of 0.3, showing this was the best in regularizing the models.]

### Best Configuration and Key Insight (2-3 sentences)
*What was your best overall model and why do you think this combination worked well? What was the most important thing you learned from this assignment?*

**Best Model:** [ReLU] with L2=[0.01], Dropout=[0.3], Test Accuracy=[0.5028139849743821]

[This model had a good balance in learning and regularization which helped with the prevention of overfitting. The model is efficient with its L2 and Dropout]