# task2.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, diff, exp, lambdify


# --- Вспомогательные и расчетные функции ---
def round_to_sf(num, sf):
    """Округляет число до заданного количества значащих цифр."""
    if num == 0:
        return 0
    return round(num, sf - int(np.floor(np.log10(abs(num)))) - 1)


def newton_method(f, df, x0, epsilon, max_iter):
    """Реализация метода Ньютона."""
    iterations = 0
    history = [x0]
    x = x0

    for i in range(max_iter):
        iterations += 1
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            st.warning("Производная близка к нулю. Метод может расходиться.")
            return x, iterations, history

        x_next = x - fx / dfx
        history.append(x_next)

        if abs(x_next - x) < epsilon:
            return x_next, iterations, history

        x = x_next

    return x, iterations, history


# --- Streamlit UI ---
st.header("Задание №2: Решение нелинейного уравнения")
st.subheader("Уравнение: `e^x - x - 2 = 0`")
st.subheader("Метод решения: Метод Ньютона (касательных)")

# --- Боковая панель с входными данными ---
with st.sidebar:
    st.header("Входные параметры для Задания 2")
    a, b = st.slider("Отрезок [a, b]", -5.0, 5.0, (-2.0, -1.0))
    epsilon = st.number_input("Точность ε", value=1e-6, format="%.e", key="eps_task2")
    max_iter = st.number_input("Макс. итераций", value=100, format="%d", key="iter_task2")

# --- Определение функций и их производных ---
x_sym = symbols('x')
f_sym = exp(x_sym) - x_sym - 2
df_sym = diff(f_sym, x_sym)
d2f_sym = diff(df_sym, x_sym)

# Преобразование в числовые функции
f = lambdify(x_sym, f_sym, 'numpy')
df = lambdify(x_sym, df_sym, 'numpy')
d2f = lambdify(x_sym, d2f_sym, 'numpy')

st.info(f"""
**Исходные данные (Вариант 6):**
- Уравнение: `f(x) = e^x - x - 2 = 0`
- Отрезок изоляции корня: `[{a}, {b}]`
- Первая производная: `f'(x) = e^x - 1`
- Вторая производная: `f''(x) = e^x`
""")

# --- 1. Проверка условий сходимости ---
st.subheader("1. Проверка условий существования и единственности корня")

# Теорема Больцано-Коши
f_a = f(a)
f_b = f(b)
bolzano_cauchy_check = f_a * f_b < 0

st.write(f"**a) Условие существования корня (Теорема Больцано-Коши): `f(a) * f(b) < 0`**")
st.write(f"`f({a})` = {round_to_sf(f_a, 4)}")
st.write(f"`f({b})` = {round_to_sf(f_b, 4)}")
st.write(f"`f(a) * f(b)` = {round_to_sf(f_a * f_b, 4)}")
if bolzano_cauchy_check:
    st.success("Условие выполняется. На отрезке есть как минимум один корень.")
else:
    st.error("Условие не выполняется. Наличие корня не гарантировано. Выберите другой отрезок.")
    st.stop()

# Проверка знакопостоянства производных
x_range_check = np.linspace(a, b, 100)
df_signs = np.sign(df(x_range_check))
d2f_signs = np.sign(d2f(x_range_check))

df_const_sign = np.all(df_signs == df_signs[0])
d2f_const_sign = np.all(d2f_signs == d2f_signs[0])

st.write(f"**b) Условие единственности корня: `f'(x)` и `f''(x)` сохраняют знак на отрезке [{a}, {b}]**")
if df_const_sign:
    st.success("`f'(x)` сохраняет знак на отрезке.")
else:
    st.error("`f'(x)` не сохраняет знак. Единственность корня не гарантирована.")

if d2f_const_sign:
    st.success("`f''(x)` сохраняет знак на отрезке.")
else:
    st.error("`f''(x)` не сохраняет знак. Условие сходимости может быть нарушено.")

# --- 2. Графическое представление ---
st.subheader("2. График функции")
x_vals = np.linspace(a - 0.5, b + 0.5, 400)
y_vals = f(x_vals)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x) = e^x - x - 2'))
fig.add_hline(y=0, line_dash="dash", line_color="red")
fig.add_vrect(x0=a, x1=b, fillcolor="lightgrey", opacity=0.3, line_width=0, name="Отрезок изоляции")
fig.update_layout(title="График функции f(x)", xaxis_title="x", yaxis_title="f(x)")
st.plotly_chart(fig, use_container_width=True)

# --- 3. Численное решение ---
st.subheader("3. Поиск корня методом Ньютона")

# Выбор начального приближения x0
if f(a) * d2f(a) > 0:
    x0 = a
    st.write(
        f"Условие `f(x₀) * f''(x₀) > 0` выполняется в точке `x₀ = a = {a}`. Выбираем ее как начальное приближение.")
elif f(b) * d2f(b) > 0:
    x0 = b
    st.write(
        f"Условие `f(x₀) * f''(x₀) > 0` выполняется в точке `x₀ = b = {b}`. Выбираем ее как начальное приближение.")
else:
    x0 = (a + b) / 2
    st.warning(
        f"Ни одна из границ отрезка не удовлетворяет условию `f(x₀) * f''(x₀) > 0`. Выбрана середина отрезка `x₀ = {x0}`.")

root, num_iterations, history = newton_method(f, df, x0, epsilon, max_iter)

st.success(f"Найденный корень уравнения: **x = {round_to_sf(root, 7)}**")
st.write(f"Количество итераций: **{num_iterations}**")

# --- Визуализация итераций ---
if history:
    st.write("**Визуализация сходимости метода:**")
    fig_iter = go.Figure()
    fig_iter.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)'))
    fig_iter.add_hline(y=0, line_dash="dash", line_color="red")
    for i, x_i in enumerate(history[:-1]):
        x_next = history[i + 1]
        tangent_x = np.array([x_i, x_next])
        tangent_y = np.array([f(x_i), 0])
        fig_iter.add_trace(go.Scatter(
            x=tangent_x, y=tangent_y, mode='lines+markers',
            name=f'Итерация {i + 1}',
            line=dict(dash='dot')
        ))

    st.plotly_chart(fig_iter, use_container_width=True)