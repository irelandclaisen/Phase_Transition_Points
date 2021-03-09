# Creates machine learning models that predict boiling and melting points

import pandas as pd
import periodictable
from periodictable import C, H, N, O, F, Cl, Br, I, S
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy import stats
import statistics
import matplotlib.pyplot as plt


def valuecall(key, atom_dict):
    """
    Takes in an element name (key) and returns the number of
    atoms of that element type. 
    """
    if key not in atom_dict:
        return 0
    else:
        return atom_dict[key]


def alcohol(string, o_value):
    """
    Returns 1 if compound contains an alcohol.  Else returns 0
    """
    if "ol" in string and o_value > 0:
        return 1.0
    else:
        return 0.0


def acid(string):
    """
    Returns 1 if compound contains an alcohol.  Else returns 0
    """
    if "acid" in string:
        return 1.0
    else:
        return 0.0


def data_processing(df):
    """
    Feature Engineering: Adds number of atoms for select common elements
    to the dataframe. Adds mass, number of atoms and functional group
    information into the dataframe
    """
    df["Object"] = df.apply(lambda row: periodictable.formula(row.Formula),
                            axis=1)
    df["Mass"] = df.apply(lambda row: row.Object.mass, axis=1)
    df["Atom_counts"] = df.apply(lambda row: sum(row.Object.atoms.values()),
                                 axis=1)
    df["C"] = df.apply(lambda row: valuecall(C, row.Object.atoms), axis=1)
    df["H"] = df.apply(lambda row: valuecall(H, row.Object.atoms), axis=1)
    df["N"] = df.apply(lambda row: valuecall(N, row.Object.atoms), axis=1)
    df["O"] = df.apply(lambda row: valuecall(O, row.Object.atoms), axis=1)
    df["F"] = df.apply(lambda row: valuecall(F, row.Object.atoms), axis=1)
    df["Cl"] = df.apply(lambda row: valuecall(Cl, row.Object.atoms), axis=1)
    df["Br"] = df.apply(lambda row: valuecall(Br, row.Object.atoms), axis=1)
    df["I"] = df.apply(lambda row: valuecall(I, row.Object.atoms), axis=1)
    df["S"] = df.apply(lambda row: valuecall(S, row.Object.atoms), axis=1)
    df["Alcohol"] = df.apply(lambda row: alcohol(row.Name, row.O), axis=1)
    df["Acid"] = df.apply(lambda row: acid(row.Name), axis=1)
    df["Halide"] = df[["F", "Cl", "Br", "I"]].sum(axis=1)
    df["Unsaturation"] = (2 * df['C'] + 2 - df['H'] -
                          df['Halide'] + df['N']) / 2


def ml_df(df, parameters, t_size, model = DecisionTreeRegressor()):
    """
    Takes in a dataframe, a list of variables (parameters), test size (t_size),
    and a model.  Fits the data in the dataframe using the specified data into 
    the model to generate predictions that are put into another data frame.
    Returns the dataframe with the predictions.
    """
    ndf = df[parameters]
    x = ndf.loc[:, ndf.columns != 'T_exp']
    y = ndf['T_exp']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size)
    model = model
    p = PolynomialFeatures(degree = 2)
    X_poly = p.fit_transform(x_train)
    X_poly_test = p.fit_transform(x_test)
    model.fit(X_poly,y_train)
    y_train_pred = model.predict(X_poly)
    y_test_pred = model.predict(X_poly_test)
    result = pd.DataFrame()
    result['T_exp'] = y_test
    result['T_prd'] = y_test_pred
    result['ratio'] = result['T_exp']/result['T_prd']
    return result


def plot_scatter(x_variable, y_variable, df, x_title, y_title):
    """
    Takes in an x variable, y variable, dataframe and titles to generate a 
    scatter plot using the specified variables, data, and labeled with the
    appropriate titles.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                df[x_variable], df[y_variable])
    plot = sns.regplot(x=x_variable, y=y_variable, data=df, line_kws={'label':
                       "y={0:.3f}x+{1:.3f}".format(slope, intercept)})
    plot.set(xlabel=x_title, ylabel=y_title + ' (K)')
    title = y_title + ' Versus ' + x_title
    plot.set_title(title)
    plot.legend()
    plot.figure.savefig(title.replace(' ', '_'))
    plt.figure()


def plot_T(df, parameters_list, title, point_type):
    """
    Takes in a dataframe, a variable list and titles to generate a 
    scatter plot showing the relationship between experimentally determined
    variables and our model's predicted values.
    """
    slope, intercept, r, p, std_er = stats.linregress(df['T_exp'], df['T_prd'])
    plot = sns.regplot(x='T_exp', y='T_prd', data=df, line_kws={'label':
                       "y={0:.3f}x+{1:.3f}".format(slope, intercept)})
    plot.legend()
    plot.set(xlabel="Experimental Temperature (K)",
             ylabel="Predicted Temperature (K)")
    plot.set_title("Parameters: " + ", ".join(parameters_list[:-1]))
    plot.figure.suptitle(point_type + ": Experimental Versus Predicted")
    plot.figure.savefig(title)
    plt.figure()


def slope_average(df, parameters, t_size, n, model = DecisionTreeRegressor()):
    """
    Takes in a dataframe (df), list of variables (parameters), size of test set 
    (t_size), and number of models (n).  Generates n number of models and gets 
    the average slope of the predicted vs experimental values, standard deviation
    of the slope, and average absolute error for these n models.
    """
    slopes = list()
    abs_error = list()
    for i in range(n):
        df2 = ml_df(df, parameters, t_size, modle)
        slope, intercept, r, p, std_er = stats.linregress(df2["T_exp"],
                                                          df2["T_prd"])
        slopes.append(slope)
        abs_error.append(abs(df2['T_exp']-df2['T_prd']).mean())
    print("Average Slope:", sum(slopes)/n)
    print("Standard Deviation:", statistics.stdev(slopes))
    print("Average Absolute Error:", sum(abs_error)/n)


def main():
    """
    Reads data, plots scatterplots of our parameters and creates various
    machine learning models to predict boiling and melting points
    """
    # Data processing and parsing
    mp_df = pd.read_csv("melting_points.csv")
    bp_df = pd.read_csv("boiling_points.csv")
    data_processing(mp_df)
    data_processing(bp_df)
    # Plots scatterplots of our variables with the boiling and melting point
    plot_scatter('Mass', 'T_exp', mp_df, 'Molecular Weight', 'Melting Point')
    plot_scatter('Mass', 'T_exp', bp_df, 'Molecular Weight', 'Boiling Point')
    plot_scatter('Atom_counts', 'T_exp', mp_df, 'Number of Atoms',
                 'Melting Point')
    
    # Creates models and prints statistics using simply mass and atom counts
    simple_parameters = ['Mass', 'Atom_counts', 'T_exp']
    model = ml_df(mp_df, simple_parameters, 0.2)
    plot_T(model, simple_parameters, "simple_melting.png", 'Melting Point')
    slope_average(mp_df, simple_parameters, 0.2, 100)
    model2 = ml_df(bp_df, simple_parameters, 0.2)
    plot_T(model2, simple_parameters, 'simple_boiling.png', 'Boiling Point')
    slope_average(bp_df, simple_parameters, 0.2, 100)
    # Creates models and prints statistics using more variables
    complex_parameters = ["Mass", "Atom_counts", "C", "H", "Acid", "Alcohol",
                          "Unsaturation", "T_exp"]
    model3 = ml_df(mp_df, complex_parameters, 0.2)
    plot_T(model3, complex_parameters, "complex_melting.png", 'Melting Point')
    slope_average(mp_df, complex_parameters, 0.2, 100)
    model4 = ml_df(bp_df, complex_parameters, 0.2)
    plot_T(model4, complex_parameters, "complex_boiling.png", 'Boiling Point')
    slope_average(bp_df, complex_parameters, 0.2, 100)
    # Creates models and prints statistics for our model using all available variables
    complex_parameters2 = ["Mass", "Atom_counts", "C", "H", 'O', 'N', 'F',
                           'Cl', 'Br', 'I', 'S', 'Si', 'Halide', "Acid", "Alcohol",
                           "Unsaturation", "T_exp"]
    model = ml_df(mp_df, complex_parameters2, 0.2)
    plot_T(model, complex_parameters2, "more_complex_melting.png",
           'Melting Point')
    slope_average(mp_df, complex_parameters2, 0.2, 100)
    # Tries a few different regressor models
    slope_average(mp_df, complex_parameters2, 0.2, 100, LinearRegression())
    slope_average(bp_df, complex_parameters2, 0.2, 100, RandomForestRegressor())


if __name__ == '__main__':
    main()
