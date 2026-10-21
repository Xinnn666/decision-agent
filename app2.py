import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from scipy.optimize import linprog, minimize_scalar

st.set_page_config(page_title="管理决策智能体", layout="wide")
st.title("管理决策智能体")
st.caption("EOQ 经济订货批量 · 边际分析决策 · 线性规划")

method = st.sidebar.radio(
    "选择决策方法",
    ["① 经济批量订货 (EOQ)", "② 边际分析决策", "③ 线性规划决策"],
)

# ==================== ① EOQ ====================
if method == "① 经济批量订货 (EOQ)":
    st.header("经济批量订货（EOQ）决策")
    with st.sidebar:
        D = st.number_input("年需求量 D（件/年）", value=10000, min_value=1, step=100)
        S = st.number_input("每次订货成本 S（元/次）", value=200.0, min_value=1.0, step=10.0)
        H = st.number_input("单位年储存成本 H（元/件·年）", value=10.0, min_value=0.01, step=1.0)

    eoq = math.sqrt(2 * D * S / H)
    orders = D / eoq
    total_cost = math.sqrt(2 * D * S * H)

    c1, c2, c3 = st.columns(3)
    c1.metric("最优订货批量 EOQ", f"{eoq:,.0f} 件")
    c2.metric("全年最优订货次数", f"{orders:.1f} 次")
    c3.metric("最低年相关总成本", f"{total_cost:,.0f} 元")

    st.info(
        f"**决策建议**：每次订购约 **{eoq:,.0f} 件**，全年订购 **{orders:.1f} 次**，"
        f"可将订货成本与储存成本之和控制在最低水平 **{total_cost:,.0f} 元**。"
        f"若实际批量偏离该值，总成本会沿 U 形曲线上升。"
    )

    qs = np.linspace(1, eoq * 2.5, 300)
    holding = qs / 2 * H
    ordering = D / qs * S
    total = holding + ordering
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=holding, mode="lines", name="年储存成本"))
    fig.add_trace(go.Scatter(x=qs, y=ordering, mode="lines", name="年订货成本"))
    fig.add_trace(go.Scatter(x=qs, y=total, mode="lines", name="年总成本"))
    fig.add_vline(x=eoq, line_dash="dash", line_color="red",
                  annotation_text=f"EOQ = {eoq:,.0f}")
    fig.update_layout(xaxis_title="订货批量（件）", yaxis_title="年成本（元）", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ==================== ② 边际分析 ====================
elif method == "② 边际分析决策":
    st.header("边际分析决策")
    st.caption("利润最大化的条件是边际收入 = 边际成本。以下用线性需求与线性变动成本模型演示。")

    with st.sidebar:
        a = st.number_input("需求函数截距 a（p = a - b·Q）", value=200.0, step=10.0)
        b = st.number_input("需求函数斜率 b", value=2.0, min_value=0.01, step=0.5)
        vc = st.number_input("单位变动成本 vc（元/件）", value=20.0, min_value=0.0, step=5.0)
        fc = st.number_input("固定成本 FC（元）", value=5000.0, min_value=0.0, step=1000.0)
        q_current = st.number_input("当前产量 Q（件）", value=20.0, min_value=0.0, step=5.0)

    def total_revenue(q):
        return (a - b * q) * q if q <= a / b else 0.0

    def total_cost(q):
        return fc + vc * q

    def total_profit(q):
        return total_revenue(q) - total_cost(q)

    res = minimize_scalar(lambda q: -total_profit(q), bounds=(0, a / b), method="bounded")
    q_opt = res.x
    p_opt = total_profit(q_opt)
    mr_opt = a - 2 * b * q_opt
    mc_opt = vc

    c1, c2, c3 = st.columns(3)
    c1.metric("最优产量 Q*", f"{q_opt:,.1f} 件")
    c2.metric("最大利润", f"{p_opt:,.0f} 元")
    c3.metric("当前利润", f"{total_profit(q_current):,.0f} 元")

    if abs(q_current - q_opt) < 0.5:
        st.success("当前产量已接近利润最大化水平。")
    elif q_current < q_opt:
        st.warning(
            f"当前产量 **{q_current:,.0f} 件** 低于最优产量 **{q_opt:,.1f} 件**。"
            f"此时边际收入（约 {a - 2 * b * q_current:,.1f} 元）仍高于边际成本（{vc:,.1f} 元），"
            "继续增产可增加利润。"
        )
    else:
        st.error(
            f"当前产量 **{q_current:,.0f} 件** 已超过最优产量 **{q_opt:,.1f} 件**。"
            "此时边际收入低于边际成本，继续增产会侵蚀利润，应考虑减产。"
        )

    st.info(
        f"**决策依据**：在最优产量 Q* = {q_opt:,.1f} 处，边际收入 MR = {mr_opt:,.1f} 元，"
        f"边际成本 MC = {mc_opt:,.1f} 元，两者相等，利润达到峰值。"
    )

    qs = np.linspace(0, a / b, 300)
    profits = [total_profit(q) for q in qs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=profits, mode="lines", name="总利润"))
    fig.add_vline(x=q_opt, line_dash="dash", line_color="red",
                  annotation_text=f"Q* = {q_opt:,.1f}")
    fig.add_vline(x=q_current, line_dash="dot", line_color="blue",
                  annotation_text=f"当前 Q = {q_current:,.0f}")
    fig.update_layout(xaxis_title="产量 Q（件）", yaxis_title="总利润（元）", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==================== ③ 线性规划 ====================
elif method == "③ 线性规划决策":
    st.header("线性规划决策")
    st.caption("在资源约束下最大化利润。默认场景：两种产品共享劳动力和原料。")

    with st.sidebar:
        p1 = st.number_input("产品 A 单位利润（元）", value=50.0, step=5.0)
        p2 = st.number_input("产品 B 单位利润（元）", value=80.0, step=5.0)
        labor_a = st.number_input("A 消耗劳动力（工时/件）", value=2.0, step=1.0)
        labor_b = st.number_input("B 消耗劳动力（工时/件）", value=4.0, step=1.0)
        labor_cap = st.number_input("劳动力上限（工时）", value=120.0, step=10.0)
        mat_a = st.number_input("A 消耗原料（kg/件）", value=3.0, step=1.0)
        mat_b = st.number_input("B 消耗原料（kg/件）", value=1.0, step=1.0)
        mat_cap = st.number_input("原料上限（kg）", value=90.0, step=10.0)

    c = [-p1, -p2]
    A_ub = [[labor_a, labor_b], [mat_a, mat_b]]
    b_ub = [labor_cap, mat_cap]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")

    if res.success:
        xa, xb = res.x
        max_profit = -res.fun
        c1, c2, c3 = st.columns(3)
        c1.metric("产品 A 最优产量", f"{xa:,.1f} 件")
        c2.metric("产品 B 最优产量", f"{xb:,.1f} 件")
        c3.metric("最大总利润", f"{max_profit:,.0f} 元")

        used_labor = labor_a * xa + labor_b * xb
        used_mat = mat_a * xa + mat_b * xb
        lines = [
            f"劳动力消耗 {used_labor:.1f} / {labor_cap} 工时，"
            f"剩余 {labor_cap - used_labor:.1f}",
            f"原料消耗 {used_mat:.1f} / {mat_cap} kg，剩余 {mat_cap - used_mat:.1f}",
        ]
        if abs(used_labor - labor_cap) < 1e-6:
            lines.append("**劳动力约束是紧约束**，增加劳动力可进一步提升利润。")
        if abs(used_mat - mat_cap) < 1e-6:
            lines.append("**原料约束是紧约束**，增加原料可进一步提升利润。")
        st.info("**决策建议**：\n\n- " + "\n- ".join(lines))

        max_x = max(xa, xb) * 1.5 + 10
        x1 = np.linspace(0, max_x, 400)
        if labor_b > 0:
            y_labor = (labor_cap - labor_a * x1) / labor_b
            fig_labor = go.Scatter(x=x1, y=y_labor, mode="lines", name="劳动力约束", fill="tozeroy")
        else:
            fig_labor = None
        if mat_b > 0:
            y_mat = (mat_cap - mat_a * x1) / mat_b
            fig_mat = go.Scatter(x=x1, y=y_mat, mode="lines", name="原料约束", fill="tozeroy")
        else:
            fig_mat = None

        fig = go.Figure()
        if fig_labor:
            fig.add_trace(fig_labor)
        if fig_mat:
            fig.add_trace(fig_mat)
        fig.add_trace(go.Scatter(x=[xa], y=[xb], mode="markers+text",
                                 marker=dict(size=12, color="red"),
                                 text=[f"最优 ({xa:.0f}, {xb:.0f})"],
                                 textposition="top center", name="最优解"))
        fig.update_layout(xaxis_title="产品 A 产量", yaxis_title="产品 B 产量",
                          height=450, xaxis_range=[0, max_x])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("在当前约束下无可行解，请检查参数是否合理。")
