# Phase_Transition_Points

### Motivation and Background:
Boiling and melting points are crucial fundamental physical properties of organic compounds. Besides being an indicator for determining the physical state of the compounds, they are also heavily utilized in chemical identification and purification in synthetic chemistry. Moreover, boiling and melting points can be used to predict the physiochemical properties of a compound such as volatility, vapor pressure, critical temperature, flash points, etc., which are widely used in industrial application for pharmaceutical screening processes or determining safety hazard or methods of handling chemicals.
Various mathematical and machine learning models have been developed to predict the boiling and melting points of compounds. However, these models often involve complicated and semiempirical parameters, such as group vector space, multitier group contribution, topological and geometric information, and quantitative structure-property relationships. Although the methods mentioned above are able predict phase transition points with good accuracy, they are often cumbersome for the target user of these models to implement.  A more simple model requiring minimum input would be more usable for bench chemists.
Here, I describe a regression model that can predict boiling points using simply the name and molecular formula of the compound.  The model is able to achieve comparable accuracy to literature machine learning models that require far more inputs.

### Data:
I used datasets obtained from published literature. The datasets include name, molecular formula, and the empirical boiling or melting point. From the chemical formula, I calculated a number of other variables, such as amount of atoms of each element, degrees of unsaturation, types of heteroatoms, and functional groups. I used pandas and the periodictable package to compute these variables.

### Featured Techniques:
 * Data Visualization
 * Regression Algorithms (Linear Regression, Decision Tree Regressor, Random Forrest Regressor)
 * Ridge and Lasso Regularization
 * Feature Engineering and Extraction
