# task1.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time


# --- Вспомогательные функции ---
def round_to_sf(num, sf):
    """Округляет число до заданного количества значащих цифр."""
    if num == 0:
        return 0
    return round(num, sf - int(np.floor(np.log10(abs(num)))) - 1)


# --- Основные расчетные функции ---

def f_theta(theta_c, M1, beta, k):
    """Целевая функция f(theta_c) = 0 для нахождения угла скачка."""
    beta_rad = np.deg2rad(beta)
    term1 = np.sin(theta_c) ** 2
    if np.isclose(np.cos(theta_c - beta_rad), 0):
        return np.inf
    term2 = ((k + 1) / 2) * (M1 ** 2 * np.sin(theta_c) * np.sin(beta_rad)) / np.cos(theta_c - beta_rad)
    term3 = 1 + M1 ** 2 * np.sin(theta_c) ** 2

    numerator = 2 * (1 / np.tan(theta_c)) * (M1 ** 2 * np.sin(theta_c) ** 2 - 1)
    denominator = M1 ** 2 * (k + np.cos(2 * theta_c)) + 2

    return np.tan(beta_rad) - numerator / denominator


def secant_method(f, x0, x1, epsilon, max_iter, M1, beta, k):
    """Реализация метода секущих."""
    iterations = 0
    history = []

    for i in range(max_iter):
        iterations += 1
        f_x1 = f(x1, M1, beta, k)
        f_x0 = f(x0, M1, beta, k)

        if abs(f_x1 - f_x0) < 1e-12:
            st.warning("Знаменатель в методе секущих стал слишком мал.")
            return x1, iterations, history

        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        history.append(x_next)

        if abs(x_next - x1) < epsilon:
            return x_next, iterations, history

        x0, x1 = x1, x_next

    return x1, iterations, history


# --- Streamlit UI ---
st.header("Задание №1: Определение параметров потока за косым скачком уплотнения")
st.subheader("Метод решения: Метод Секущих")

# --- Боковая панель с входными данными ---
with st.sidebar:
    st.header("Входные параметры для Задания 1")
    p1 = st.number_input("Давление p1, Па", value=165000, format="%d")
    T1 = st.number_input("Температура T1, К", value=300, format="%d")
    M1 = st.number_input("Число Маха M1", value=4.0, step=0.1, format="%.1f")
    beta_deg = st.number_input("Угол клина β, °", value=15.0, step=0.5, format="%.1f")
    epsilon = st.number_input("Точность ε", value=1e-6, format="%.e")
    max_iter = st.number_input("Макс. итераций", value=100, format="%d")

# --- Константы ---
k = 1.4
R = 287

st.info(f"""
**Исходные данные (Вариант 6):**
- Давление набегающего потока, p₁ = {p1} Па
- Температура набегающего потока, T₁ = {T1} K
- Число Маха набегающего потока, M₁ = {M1}
- Угол полураствора клина, β = {beta_deg}°
- Показатель адиабаты, k = {k}
- Газовая постоянная, R = {R} Дж/(кг·К)
""")

# --- 1. Визуализация и отделение корня ---
st.subheader("1. Графическое отделение корня")
st.write(
    "Для нахождения угла наклона скачка `θс` необходимо решить нелинейное уравнение `f(θс) = 0`. "
    "Построим график функции на интервале от `β` до `π/2`, чтобы визуально определить примерное положение корня."
)

theta_range_rad = np.linspace(np.deg2rad(beta_deg) + 0.001, np.pi / 2, 500)
y_values = [f_theta(th, M1, beta_deg, k) for th in theta_range_rad]

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.rad2deg(theta_range_rad), y=y_values, mode='lines', name='f(θc)'))
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.update_layout(
    title="График функции f(θс)",
    xaxis_title="Угол скачка θс, °",
    yaxis_title="f(θс)",
    xaxis=dict(range=[beta_deg, 90])
)
st.plotly_chart(fig, use_container_width=True)

# --- 2. Численное решение ---
st.subheader("2. Поиск угла скачка `θс` методом секущих")

# Начальные приближения берем из графика
x0_rad = np.deg2rad(beta_deg + 1)
x1_rad = np.deg2rad(40)

root_rad, num_iterations, _ = secant_method(f_theta, x0_rad, x1_rad, epsilon, max_iter, M1, beta_deg, k)
root_deg = np.rad2deg(root_rad)

st.success(f"Найденный угол скачка уплотнения θс: **{round_to_sf(root_deg, 4)}°** ({round_to_sf(root_rad, 4)} рад)")
st.write(f"Количество итераций: **{num_iterations}**")

# --- 3. Расчет параметров за скачком ---
st.subheader("3. Расчет параметров потока за скачком уплотнения")

# Промежуточные расчеты
rho1 = p1 / (R * T1)
a1 = np.sqrt(k * R * T1)
v1 = M1 * a1

# Стандартные формулы для косого скачка
M1n = M1 * np.sin(root_rad)
p2 = p1 * (1 + (2 * k / (k + 1)) * (M1n ** 2 - 1))
rho2 = rho1 * ((k + 1) * M1n ** 2) / ((k - 1) * M1n ** 2 + 2)
T2 = T1 * (p2 / p1) * (rho1 / rho2)

beta_rad = np.deg2rad(beta_deg)
M2 = np.sqrt(
    ((k - 1) * M1n ** 2 + 2) /
    (2 * k * M1n ** 2 - (k - 1))
) / np.sin(root_rad - beta_rad)

# --- 4. Определение режима течения и вывод результатов ---
flow_regime = "Сверхзвуковой" if M2 > 1 else "Дозвуковой" if M2 < 1 else "Звуковой"

st.write("Используя найденный угол `θс`, рассчитаем остальные параметры:")

results = {
    "Параметр": ["Давление, p", "Температура, T", "Плотность, ρ", "Число Маха, M"],
    "До скачка (индекс 1)": [
        f"{round_to_sf(p1, 3)} Па",
        f"{round_to_sf(T1, 3)} К",
        f"{round_to_sf(rho1, 3)} кг/м³",
        f"{round_to_sf(M1, 3)}"
    ],
    "После скачка (индекс 2)": [
        f"{round_to_sf(p2, 3)} Па",
        f"{round_to_sf(T2, 3)} К",
        f"{round_to_sf(rho2, 3)} кг/м³",
        f"{round_to_sf(M2, 3)}"
    ]
}
st.table(results)

st.info(f"Режим течения после скачка (при M₂ = {round_to_sf(M2, 3)}): **{flow_regime}**")