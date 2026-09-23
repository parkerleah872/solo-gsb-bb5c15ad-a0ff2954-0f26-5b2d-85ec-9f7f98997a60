"""溶液配制：由若干母液混合配制满足目标浓度的终溶液。

给定至多 4 种母液（每种母液给出各溶质的浓度 mol/L 与可用体积 mL）、
目标终体积 V（mL）与各溶质目标浓度（mol/L），求非负取用量
``x_1, …, x_n``（mL）满足：

* 每种目标溶质 j 的浓度约束 ``Σ_i c_ij·x_i = t_j·V``（体积以 mL
  计，比值中 1000 抵消，等式两侧同为 mmol）；
* 容量约束 ``0 ≤ x_i ≤ 可用体积_i``；
* ``Σ_i x_i ≤ V``，余量 ``V − Σ_i x_i`` 用水补足。

多解时先最小化母液总取用量 ``Σ_i x_i``（即最大化补水量），再按母液的
**输入顺序**对取用量向量取字典序最小。

求解方法（全程 :class:`fractions.Fraction` 精确有理数，不经过 float）：

1. 对浓度等式约束做高斯-若尔当消元（RREF），矛盾行（0 = 非零）即无解；
2. 通解写成 ``x = x0 + N·t``，``t`` 为自由变量（至多 4 个）；
3. 容量与总用量约束化为自由变量空间内的线性不等式组 ``G·t ≤ h``，
   可行域是有界凸多面体，线性目标的最优解必在顶点；
4. 枚举至多 f 个线性无关约束同时取等得到的全部顶点（约束数 ≤ 9、
   f ≤ 4，组合数至多 C(9,4)=126），逐一回代校验可行性，按
   （总取用量, 输入顺序字典序）取最优。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations


@dataclass(frozen=True)
class Stock:
    """一种母液：标识、各溶质浓度（mol/L）与可用体积（mL）。"""

    id: str
    concentrations: dict[str, Fraction]
    available_volume: Fraction


@dataclass(frozen=True)
class Target:
    """一条目标溶质浓度记录（mol/L）。"""

    solute: str
    concentration: Fraction


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[tuple[int, int]]]:
    """对增广矩阵做精确有理数高斯-若尔当消元，返回 (RREF, 主元(行,列) 列表)。"""
    m = [list(row) for row in matrix]
    if not m:
        return m, []
    rows, cols = len(m), len(m[0])
    pivots: list[tuple[int, int]] = []
    col = 0
    r = 0
    while r < rows and col < cols - 1:
        pivot = next((i for i in range(r, rows) if m[i][col] != 0), None)
        if pivot is None:
            col += 1
            continue
        m[r], m[pivot] = m[pivot], m[r]
        pivot_value = m[r][col]
        m[r] = [value / pivot_value for value in m[r]]
        for i in range(rows):
            if i != r and m[i][col] != 0:
                factor = m[i][col]
                m[i] = [a - factor * b for a, b in zip(m[i], m[r])]
        pivots.append((r, col))
        r += 1
        col += 1
    return m, pivots


def _solve_square(
    matrix: list[list[Fraction]], rhs: list[Fraction]
) -> list[Fraction] | None:
    """解 k×k 线性方程组；矩阵奇异返回 None。"""
    k = len(matrix)
    m = [list(matrix[i]) + [rhs[i]] for i in range(k)]
    for col in range(k):
        pivot = next((r for r in range(col, k) if m[r][col] != 0), None)
        if pivot is None:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pivot_value = m[col][col]
        m[col] = [value / pivot_value for value in m[col]]
        for r in range(k):
            if r == col:
                continue
            factor = m[r][col]
            if factor != 0:
                m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
    return [m[i][k] for i in range(k)]


def solve_blend(
    stocks: list[Stock],
    targets: list[Target],
    final_volume: Fraction,
) -> dict[str, Fraction | list[Fraction] | list[tuple[str, Fraction]]] | None:
    """求最优母液取用量；不存在可行配制时返回 None。

    返回字典包含：``taken``（按输入顺序的各母液取用量 mL）、
    ``total_stock``（母液总取用量 mL）、``water``（补水量 mL）与
    ``final_concentrations``（``(溶质, 终浓度 mol/L)`` 列表，目标溶质
    按 targets 首次出现顺序在前、其余溶质按名字升序在后）。
    """
    n = len(stocks)

    # 溶质全集：目标溶质按首次出现顺序，母液中额外出现的溶质按名字补入
    solute_order: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target.solute not in seen:
            seen.add(target.solute)
            solute_order.append(target.solute)
    for stock in stocks:
        for name in sorted(stock.concentrations):
            if name not in seen:
                seen.add(name)
                solute_order.append(name)

    target_map = {target.solute: target.concentration for target in targets}

    # ---- 1. 浓度等式 A·x = b（每个目标溶质一行；b = 目标浓度 × 终体积）
    equations: list[list[Fraction]] = []
    for target in targets:
        row = [stock.concentrations.get(target.solute, Fraction(0)) for stock in stocks]
        row.append(target.concentration * final_volume)
        equations.append(row)

    rref, pivots = _rref(equations)
    var_count = n
    for row in rref:
        if all(row[c] == 0 for c in range(var_count)) and row[var_count] != 0:
            return None  # 0 = 非零，浓度约束本身矛盾

    # ---- 2. 通解 x = basic + N·t
    pivot_cols = {c for _r, c in pivots}
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basic = [Fraction(0)] * n
    for r, c in pivots:
        basic[c] = rref[r][n]
    nullspace = [[Fraction(0)] * n for _ in free_cols]
    for k, free_col in enumerate(free_cols):
        nullspace[k][free_col] = Fraction(1)
        for r, pivot_col in pivots:
            nullspace[k][pivot_col] = -rref[r][free_col]

    # ---- 3. 不等式 G·t ≤ h：各母液 0 ≤ x_i ≤ 可用量，且 Σx_i ≤ V
    rows_g: list[list[Fraction]] = []
    rows_h: list[Fraction] = []

    def add_inequality(g: list[Fraction], h: Fraction) -> bool:
        """追加一行；0·t ≤ h 类约束直接判可行性，返回当前是否仍可行。"""
        if all(value == 0 for value in g):
            return h >= 0
        rows_g.append(g)
        rows_h.append(h)
        return True

    for j, stock in enumerate(stocks):
        g_lower = [-nullspace[k][j] for k in range(len(free_cols))]
        g_upper = [nullspace[k][j] for k in range(len(free_cols))]
        if not add_inequality(g_lower, basic[j]):
            return None
        if not add_inequality(g_upper, stock.available_volume - basic[j]):
            return None

    g_total = [sum(nullspace[k][j] for j in range(n)) for k in range(len(free_cols))]
    if not add_inequality(g_total, final_volume - sum(basic)):
        return None

    f = len(free_cols)

    def feasible(t: list[Fraction]) -> bool:
        return all(
            sum(g[k] * t[k] for k in range(f)) <= h for g, h in zip(rows_g, rows_h)
        )

    def taken_at(t: list[Fraction]) -> list[Fraction]:
        return [
            basic[j] + sum(nullspace[k][j] * t[k] for k in range(f))
            for j in range(n)
        ]

    # ---- 4. 枚举可行域顶点，按 (总取用量, 输入顺序字典序) 选最优
    candidates: list[list[Fraction]] = []
    if f == 0:
        if feasible([]):
            candidates.append([])
    else:
        for indices in combinations(range(len(rows_g)), f):
            sub_g = [rows_g[i] for i in indices]
            sub_h = [rows_h[i] for i in indices]
            t = _solve_square(sub_g, sub_h)
            if t is not None and feasible(t):
                candidates.append(t)

    if not candidates:
        return None

    best_t = min(candidates, key=lambda t: (sum(taken_at(t)), tuple(taken_at(t))))
    taken = taken_at(best_t)
    total_stock = sum(taken)
    water = final_volume - total_stock

    final_concentrations: list[tuple[str, Fraction]] = []
    for name in solute_order:
        amount = sum(stock.concentrations.get(name, Fraction(0)) * x_j
                     for stock, x_j in zip(stocks, taken))
        final_concentrations.append((name, amount / final_volume))

    return {
        "taken": taken,
        "total_stock": total_stock,
        "water": water,
        "final_concentrations": final_concentrations,
    }
