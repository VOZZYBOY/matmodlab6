# task3.py
import streamlit as st
import numpy as np
import timeit
from sympy import symbols, sin, cos, tan, diff, lambdify

# --- Вспомогательные функции ---
def round_to_sf(num, sf):
    """Округляет число до заданного количества значащих цифр."""
    if num == 0:
        return 0
    if abs(num) < 1e-12:
        return 0
    return round(num, sf - int(np.floor(np.log10(abs(num)))) - 1)

# --- Расчетные функции ---
def f_theta_numeric(theta_c, M1, beta, k):
    """Уравнение (из task1), которое нужно решить: f(theta_c) = 0."""
    beta_rad = np.deg2rad(beta)
    if np.isclose(np.tan(theta_c), 0) or M1**2 * (k + np.cos(2*theta_c)) + 2 == 0:
        return np.inf
    numerator = 2 * (1/np.tan(theta_c)) * (M1**2 * np.sin(theta_c)**2 - 1)
    denominator = M1**2 * (k + np.cos(2*theta_c)) + 2
    return np.tan(beta_rad) - numerator / denominator

# --- Численные методы (версии без встроенного таймера) ---
def secant_method(f, x0, x1, epsilon, max_iter, *args):
    """Метод секущих. Возвращает корень и количество итераций."""
    iterations = 0
    for i in range(max_iter):
        iterations += 1
        f_x1 = f(x1, *args)
        f_x0 = f(x0, *args)
        if abs(f_x1 - f_x0) < 1e-15: return x1, iterations
        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(x_next - x1) < epsilon: return x_next, iterations
        x0, x1 = x1, x_next
    return x1, iterations

def newton_method(f, df, x0, epsilon, max_iter, *args):
    """Метод Ньютона. Возвращает корень и количество итераций."""
    iterations = 0
    x = x0
    for i in range(max_iter):
        iterations += 1
        fx = f(x, *args)
        dfx = df(x, *args)
        if abs(dfx) < 1e-15: return x, iterations
        x_next = x - fx / dfx
        if abs(x_next - x) < epsilon: return x_next, iterations
        x = x_next
    return x, iterations

# --- Streamlit UI ---
st.header("Задание №3: Сравнение численных методов")
st.write(
    "Решим уравнение из **Задания №1** (`f(θс) = 0`), используя метод, указанный в **Задании №2** (метод Ньютона), "
    "и сравним его эффективность с методом из **Задания №1** (метод секущих)."
)

# --- Входные данные ---
with st.sidebar:
    st.header("Входные параметры для Задания 3")
    M1 = st.number_input("Число Маха M1", value=4.0, step=0.1, format="%.1f", key="m1_task3")
    beta_deg = st.number_input("Угол клина β, °", value=15.0, step=0.5, format="%.1f", key="beta_task3")
    epsilon = st.number_input("Точность ε", value=1e-9, format="%.e", key="eps_task3")
    max_iter = st.number_input("Макс. итераций", value=100, format="%d", key="iter_task3")
    num_runs = st.number_input("Количество запусков для замера времени", value=10000, min_value=100, format="%d")

k = 1.4
st.info(f"**Условия эксперимента:** M₁ = {M1}, β = {beta_deg}°, k = {k}, ε = {epsilon}")

# --- 1. Подготовка производной для метода Ньютона ---
st.subheader("1. Подготовка к методу Ньютона")
st.write("Для метода Ньютона требуется первая производная `f'(θс)`, найденная аналитически.")
theta_c_sym, M1_sym, beta_sym, k_sym = symbols('theta_c M1 beta k')
f_sym = tan(beta_sym) - (2 * (1/tan(theta_c_sym)) * (M1_sym**2 * sin(theta_c_sym)**2 - 1) /
                         (M1_sym**2 * (k_sym + cos(2*theta_c_sym)) + 2))
df_sym = diff(f_sym, theta_c_sym)
with st.expander("Показать аналитическую производную"):
    st.latex("f'(\\theta_c) = " + str(df_sym))
df_lambda = lambdify([theta_c_sym, M1_sym, beta_sym, k_sym], df_sym, 'numpy')

def df_theta_numeric(theta_c, M1, beta, k):
    beta_rad = np.deg2rad(beta)
    return df_lambda(theta_c, M1, beta_rad, k)

# --- 2. Запуск и сравнение методов ---
x0_rad_secant = np.deg2rad(beta_deg + 1)
x1_rad_secant = np.deg2rad(40)
x0_rad_newton = np.deg2rad(30)
args = (M1, beta_deg, k)

# Выполняем по одному разу, чтобы получить корень и число итераций
root_s, iter_s = secant_method(f_theta_numeric, x0_rad_secant, x1_rad_secant, epsilon, max_iter, *args)
root_n, iter_n = newton_method(f_theta_numeric, df_theta_numeric, x0_rad_newton, epsilon, max_iter, *args)

secant_callable = lambda: secant_method(f_theta_numeric, x0_rad_secant, x1_rad_secant, epsilon, max_iter, *args)
newton_callable = lambda: newton_method(f_theta_numeric, df_theta_numeric, x0_rad_newton, epsilon, max_iter, *args)

total_time_s = timeit.timeit(stmt=secant_callable, number=num_runs)
avg_time_s = total_time_s / num_runs

total_time_n = timeit.timeit(stmt=newton_callable, number=num_runs)
avg_time_n = total_time_n / num_runs


# --- 3. Вывод результатов и выводы ---
st.subheader("2. Результаты и выводы")
st.info(f"Замеры времени производились как среднее по **{num_runs}** запускам каждого метода.")
results_data = {
    "Параметр": ["Метод", "Найденный корень, °", "Количество итераций", "Среднее время расчета, сек"],
    "Метод Секущих": [
        "Секущих",
        f"{round_to_sf(np.rad2deg(root_s), 7)}",
        iter_s,
        f"{avg_time_s:.5e}"
    ],
    "Метод Ньютона": [
        "Ньютона",
        f"{round_to_sf(np.rad2deg(root_n), 7)}",
        iter_n,
        f"{avg_time_n:.5e}"
    ]
}
st.table(results_data)
