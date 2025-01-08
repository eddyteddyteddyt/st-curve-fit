import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# %% WEBPAGE

col1, col2 = st.columns([3, 8])

with col2:
    st.title('Curve Fitting App')

with st.sidebar:
    st.text('Upload a .csv file or manually input your x and y-values!')

    x_values = st.text_input(label='Enter the x-values, separated by commas.')
    # Stores x-values to variable
    y_values = st.text_input(label='Enter the y-values, separated by commas.')
    # Stores y-values to variable
    file_name = st.file_uploader(label='Alternatively, upload your .CSV file'
                                 ' here')
    st.text('Note: while a .csv file is attached, it will have priority\nover'
            ' manually inputted data.')  # File uploader
    plotoption = st.selectbox(label='Select a fit for your data',
                              options=('Linear', 'Polynomial', 'Exponential',
                                       'Logarithmic', 'Sinusoidal (Radians)',
                                       'Histogram (Input x-values only)',
                                       'Boxplot (Input x-values only)'))
    if plotoption == 'Polynomial':
        degree = st.number_input(label='Select the degree of your polynomial',
                                 min_value=1, step=1)
    else:
        degree = None

    polynomial_graph = plt.Figure(figsize=(10, 10))
    conf = st.checkbox(label='Add 95% Confidence Interval')
    plot = st.button(label='Plot')

# %% Data Processing

if file_name is not None:
    values = pd.read_csv(file_name, header=None)
    try:
        x_list = values.iloc[:, 0].tolist()
        x_list = np.array(x_list)
        y_list = values.iloc[:, 1].tolist()
        y_list = np.array(y_list)
    except IndexError:
        y_list = []
    except ValueError:
        st.error('Please only enter numeric values')
        x_list = []
        y_list = []
elif x_values and y_values:
    try:
        x_list = [float(i) for i in x_values.split(',')]
        x_list = np.array(x_list, dtype=float)
        y_list = [float(i) for i in y_values.split(',')]
        y_list = np.array(y_list, dtype=float)
    except ValueError:
        st.error('Please only enter numeric values')
        x_list = []
        y_list = []
elif x_values and not y_values:
    try:
        x_list = [float(i) for i in x_values.split(',')]
        x_list = np.array(x_list, dtype=float)
    except ValueError:
        st.error('Please only enter numeric values')
        x_list = []

# %% Direct Value Data Processing


def linear():
    x = x_list
    y1 = y_list
    plt.scatter(x, y1, color='Black')
    poly_coeffs = np.polyfit(x, y1, 1)
    y = np.polyval(poly_coeffs, x)
    plt.title("Linear")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, ls="-", color='Purple')
    plt.grid()
    if conf:
        ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
    st.pyplot(plt)
    A = round(poly_coeffs[0], 3)
    B = round(poly_coeffs[1], 3)
    error = round(np.square(np.subtract(y1, y)).mean(), 3)
    st.text(f'The equation of the fitted curve is y = {A}x + {B}')
    st.text(f'The mean squared error of the curve fit is {error}')
    return poly_coeffs


def polynomial(degree):
    x = x_list
    y1 = y_list
    plt.scatter(x, y1, color='Black')
    degree = float(degree)
    poly_coeffs = np.polyfit(x, y1, degree)
    y = np.polyval(poly_coeffs, x)
    plt.title("Polynomial")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y_smooth = np.polyval(poly_coeffs, x_smooth)
    y2 = np.polyval(poly_coeffs, x)
    if conf:
        ci = 1.96 * np.std(y_smooth)/np.sqrt(len(x))
        plt.fill_between(x_smooth, (y_smooth-ci), (y_smooth+ci), color='b',
                         alpha=.1)
    plt.plot(x_smooth, y_smooth, ls="-", color='Purple')
    st.pyplot(plt)
    equation = 'The equation of the fitted curve is y = '
    i = 0
    while i < len(poly_coeffs):
        exp = len(poly_coeffs) - 1 - i
        roundcoeffs = round(poly_coeffs[i], 3)
        equation += f'{roundcoeffs}x^{exp} + '
        i += 1
    error = round(np.square(np.subtract(y1, y2)).mean(), 3)
    st.text(equation)
    st.text(f'The mean squared error of the curve fit is {error}')
    return poly_coeffs


def exp():
    x = x_list
    y1 = y_list
    plt.scatter(x, y1, color='Black')

    def func(x, a, b, c):
        return a * np.exp(b * x) + c
    popt, pcov = curve_fit(func, x, y1, maxfev=10000)
    plt.title("Exponential")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y = func(x_smooth, *popt)
    y2 = func(x, *popt)
    if conf:
        ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.fill_between(x_smooth, (y-ci), (y+ci), color='b', alpha=.1)
    plt.plot(x_smooth, func(x_smooth, *popt), ls="-")
    st.pyplot(plt)
    A = round(popt[0], 3)
    B = round(popt[1], 3)
    C = round(popt[2], 3)
    error = round(np.square(np.subtract(y1, y2)).mean(), 3)
    st.text(f'The equation of the fitted curve is y = {A}e^{B}x + {C}')
    st.text(f'The mean squared error of the curve fit is {error}')
    return popt, pcov


def logarithm():
    x = x_list
    y1 = y_list
    plt.scatter(x, y1, color='Black')

    def func(x, a, b):
        return a * np.log(x) + b
    popt, pcov = curve_fit(func, x, y1, maxfev=10000)
    plt.title("Logarithmic")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y = func(x_smooth, *popt)
    y2 = func(x, *popt)
    if conf:
        ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.fill_between(x_smooth, (y-ci), (y+ci), color='b', alpha=.1)
    plt.plot(x_smooth, func(x_smooth, *popt), ls="-")
    st.pyplot(plt)
    A = round(popt[0], 3)
    B = round(popt[1], 3)
    error = round(np.square(np.subtract(y1, y2)).mean(), 3)
    st.text(f'The equation of the fitted curve is y = {A}ln(x) + {B}')
    st.text(f'The mean squared error of the curve fit is {error}')
    return popt, pcov


def sine():
    x = x_list
    y1 = y_list
    plt.scatter(x, y1, color='Black')

    def func(x, a, b, c, d):
        return a * np.sin(b * x + c) + d
    popt, pcov = curve_fit(func, x, y1)
    plt.title("Sinusoidal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y = func(x_smooth, *popt)
    y2 = func(x, *popt)
    if conf:
        ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.fill_between(x_smooth, (y-ci), (y+ci), color='b', alpha=.1)
    plt.plot(x_smooth, func(x_smooth, *popt), ls="-")
    st.pyplot(plt)
    A = round(popt[0], 3)
    B = round(popt[1], 3)
    C = round(popt[2], 3)
    D = round(popt[3], 3)
    error = round(np.square(np.subtract(y1, y2)).mean(), 3)
    st.text(f'The equation of the fitted curve is: y = {A}sin({B}x+{C}) + {D}')
    st.text(f'The mean squared error of the curve fit is {error}')
    return popt, pcov


def hist():
    x = x_list
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.hist(x, color='Purple')
    st.pyplot(plt)
    return None


def box():
    x = x_list
    plt.title("Boxplot")
    plt.xlabel("Value")
    plt.ylabel("Group")
    plt.boxplot(x, vert=False)
    st.pyplot(plt)
    return None

# %% Plotting


if plot and plotoption == 'Polynomial':
    polynomial(degree)

if plot and plotoption == 'Linear':
    linear()

if plot and plotoption == 'Exponential':
    exp()

if plot and plotoption == 'Logarithmic':
    logarithm()

if plot and plotoption == 'Sinusoidal (Radians)':
    sine()

if plot and plotoption == 'Histogram (Input x-values only)':
    hist()

if plot and plotoption == 'Boxplot (Input x-values only)':
    box()
