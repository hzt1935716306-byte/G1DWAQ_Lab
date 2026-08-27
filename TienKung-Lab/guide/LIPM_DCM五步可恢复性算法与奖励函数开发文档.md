# LIPM/DCM 五步可恢复性算法与奖励函数开发文档

> 文档用途：作为编辑器 AI 实现恢复认证器、训练奖励和恢复状态机的直接开发依据。  
> 当前锁定版本：保留跨步耦合的扩展状态算法，将最大预测时域从 3 次触地扩展为 5 次触地。  
> 本文中的“步”均指一次新的 touchdown（新增触地），不是 policy control step。

---

## 1. 目标与最终约定

本方案解决的问题是：机器人受到扰动以后，从当前状态出发，在给定 LIPM、CoP、落脚几何和摆动速度约束下，理论上最少还需要多少次触地才能回到指定的站立或周期行走终端。

认证器输出：

$$
N_{\min}\in\{0,1,2,3,4,5,>5\}
$$

其语义为：

| 输出 | 含义 |
|---|---|
| $N_{\min}=0$ | 当前已经位于恢复终端，不需要新增触地 |
| $N_{\min}=n,\ n=1,\ldots,5$ | 存在一条恰好使用 $n$ 次未来触地的可行恢复见证，且没有更短的可行见证 |
| $N_{\min}>5$ | 在当前模型、约束和 5 步时域内没有认证出可行恢复方案 |

必须保留以下区别：

$$
\boxed{N_{\min}>5\neq\text{机器人已经摔倒}}
$$

$N_{\min}>5$ 只表示“5 步内未认证可恢复”；`FALL` 必须由仿真器或真实机器人健康条件单独判断。

训练和部署的用法不同：

| 阶段 | 认证器的作用 |
|---|---|
| 训练 | 认证器是 physics teacher / reward generator；恢复期间按配置周期计算 $N_{\min}(t)$ 和连续裕度 $m(t)$，并在 touchdown 时计算事件奖励 |
| 部署 | Actor 独立运行；默认只在每次 touchdown 计算一次 $N_{\min}$，用于监视和日志，不把 LP 计划作为 Actor 的控制输入 |

---

## 2. 建模假设与符号

第一版采用以下模型边界：

- 质心水平运动采用恒高 LIPM；
- 左右脚交替支撑；
- 每个支撑阶段内 CoP 采用常值决策变量；
- touchdown 被视为瞬时支撑切换；
- 落脚域、CoP 域和摆动速度约束均可写成线性不等式；
- 正式多步认证使用扩展状态 $z=[b,q]$，不得退化成只含 $b$ 的二维多步递推；
- 所有二维量必须位于同一个冻结的 heading frame，并使用米、秒、弧度等统一 SI 单位。

主要符号：

| 符号 | 含义 |
|---|---|
| $g$ | 重力加速度 |
| $h$ | LIPM 有效 CoM 高度 |
| $c,\dot c\in\mathbb R^2$ | CoM 水平位置和速度 |
| $\xi\in\mathbb R^2$ | DCM |
| $p_\sigma$ | 当前支撑脚位置，$\sigma\in\{L,R\}$ |
| $p_{\mathrm{sw}}$ | 当前摆动脚位置 |
| $b=\xi-p_\sigma$ | 支撑脚相对坐标中的 DCM |
| $q=p_{\mathrm{sw}}-p_\sigma$ | 摆动脚相对当前支撑脚的位置 |
| $u=r_{\mathrm{CoP}}-p_\sigma$ | CoP 相对当前支撑脚的位置 |
| $\ell=p_{\bar\sigma}-p_\sigma$ | 下一落脚点相对当前支撑脚的位移 |
| $\mathcal C_\sigma$ | 当前支撑侧的 CoP 可行域 |
| $\mathcal L_\sigma$ | 当前支撑侧的几何落脚域 |
| $v_{\max}$ | 摆动脚逐轴最大速度 |
| $T$ | 一个完整支撑步的时间 |
| $T_{\mathrm{rem}}$ | 当前步剩余时间 |
| $N_{\max}$ | 最大认证触地数，本文固定为 5 |

注意 $q$ 的方向必须固定为：

$$
\boxed{q=p_{\mathrm{sw}}-p_\sigma}
$$

不能在代码的不同模块中反向定义。

---

## 3. 从 LIPM 到 DCM 的推导

### 3.1 LIPM 动力学

恒高 LIPM 的水平动力学为：

$$
\ddot c=\omega^2(c-r_{\mathrm{CoP}}),
\qquad
\omega=\sqrt{\frac gh}.
$$

定义 DCM：

$$
\boxed{\xi=c+\frac{\dot c}{\omega}}.
$$

对其求导：

$$
\dot\xi
=\dot c+\frac{\ddot c}{\omega}
=\dot c+\omega(c-r_{\mathrm{CoP}})
=\omega(\xi-r_{\mathrm{CoP}}).
$$

因此 DCM 是 LIPM 的发散分量，其动力学为：

$$
\boxed{\dot\xi=\omega(\xi-r_{\mathrm{CoP}})}.
$$

### 3.2 支撑脚相对坐标

在一个支撑阶段内，支撑脚 $p_\sigma$ 固定。定义：

$$
b=\xi-p_\sigma,
\qquad
u=r_{\mathrm{CoP}}-p_\sigma.
$$

于是：

$$
\dot b=\omega(b-u).
$$

若传播时间为 $\tau$，且该阶段内 $u$ 为常值，则解析解为：

$$
\boxed{
b(t+\tau)
=e^{\omega\tau}b(t)
-\left(e^{\omega\tau}-1\right)u
}.
$$

记：

$$
A_\tau=e^{\omega\tau},
$$

则触地前的相对 DCM 为：

$$
b^-_{k+1}=A_kb_k-(A_k-1)u_k.
$$

### 3.3 扰动对 DCM 的影响

若水平外力形成瞬时冲量：

$$
J=\int F(t)\,dt,
$$

用 $M$ 表示机器人总质量，则：

$$
\Delta\dot c=\frac JM.
$$

CoM 位置和双脚位置在理想瞬时冲量下不跳变，因此：

$$
\boxed{
\Delta\xi=\Delta b=\frac{J}{M\omega},
\qquad
\Delta q=0
}.
$$

于是扰动后的扩展状态为：

$$
\boxed{
z^+=
\begin{bmatrix}
b^-+J/(M\omega)\\q^-
\end{bmatrix}
}.
$$

若外力持续一段时间，不应只在开始或结束时机械套用瞬时冲量公式；应由仿真器积分得到外力结束时的真实 $c,\dot c$，再重建 $b,q$ 并启动 `RECOVERY`。

---

## 4. 跨步耦合触地映射

### 4.1 DCM 坐标切换

第 $k$ 个支撑阶段结束时，新支撑脚位于：

$$
p_{k+1}=p_k+\ell_k.
$$

理想瞬时换脚时，DCM 的世界坐标连续，但其支撑脚相对表示发生改变：

$$
\begin{aligned}
b_{k+1}
&=\xi^-_{k+1}-p_{k+1}\\
&=(\xi^-_{k+1}-p_k)-\ell_k\\
&=A_kb_k-(A_k-1)u_k-\ell_k.
\end{aligned}
$$

所以：

$$
\boxed{
b_{k+1}=A_kb_k-(A_k-1)u_k-\ell_k
}.
$$

### 4.2 摆动脚状态更新

触地以后，旧支撑脚变成新摆动脚。因此：

$$
q_{k+1}=p_k-p_{k+1}=-\ell_k.
$$

所以：

$$
\boxed{q_{k+1}=-\ell_k}.
$$

该等式是认证器内部的理想无滑移触地模型。接入全身仿真或实机时，每次 touchdown 后必须重新用实测脚位计算：

$$
q_{k+1}^{\mathrm{meas}}
=p_{\mathrm{sw},k+1}^{\mathrm{meas}}
-p_{\mathrm{sup},k+1}^{\mathrm{meas}}.
$$

若存在支撑脚滑移，实测量不一定严格满足 $q_{k+1}^{\mathrm{meas}}=-\ell_k^{\mathrm{meas}}$。代码应记录残差 $q_{k+1}^{\mathrm{meas}}+\ell_k^{\mathrm{meas}}$，然后用实测 $q$ 重新认证；不得为了满足理想公式而改写或瞬移机器人状态。

### 4.3 扩展状态映射

定义正式认证状态：

$$
\boxed{
z_k=
\begin{bmatrix}
b_k\\q_k
\end{bmatrix}
\in\mathbb R^4
}.
$$

完整触地映射为：

$$
\boxed{
z_{k+1}=
\begin{bmatrix}
A_kb_k-(A_k-1)u_k-\ell_k\\
-\ell_k
\end{bmatrix}
}.
$$

$q_{k+1}=-\ell_k$ 会把本步落脚决策带入下一步的摆动速度约束，因此形成真正的跨步耦合。旧的二维 $b$-only 多步递推会丢失该耦合，只能保留为解析基线，不能作为正式证书。

---

## 5. 每一步的物理约束

对预测步 $i=0,\ldots,N-1$，必须同时满足以下约束。

### 5.1 CoP 约束

$$
\boxed{u_i\in\mathcal C_{\sigma_i}}.
$$

### 5.2 几何落脚约束

$$
\boxed{\ell_i\in\mathcal L_{\sigma_i}}.
$$

### 5.3 有限摆动速度约束

摆动脚需要从当前相对位置 $q_i$ 到达落脚位置 $\ell_i$，所以：

$$
\boxed{
-v_{\max}\odot T_i
\le \ell_i-q_i
\le v_{\max}\odot T_i
}.
$$

这里是逐轴约束。不得误写成 $\ell_i+q_i$，也不得只约束第一步而忽略后续步。

### 5.4 传播时间

训练期间若在步中任意时刻调用认证器：

$$
T_0=T_{\mathrm{rem}},
\qquad
T_i=T\quad(i\ge1),
$$

$$
A_i=e^{\omega T_i}.
$$

若认证器在 touchdown 截面调用，例如部署监视，则新支撑阶段刚刚开始：

$$
T_i=T,
\qquad
A_i=e^{\omega T},
\quad i=0,\ldots,N-1.
$$

### 5.5 左右支撑交替

$$
\boxed{\sigma_{i+1}=\bar\sigma_i}.
$$

左右脚的 $\mathcal C_\sigma$、$\mathcal L_\sigma$ 和名义落脚均应由支撑侧显式索引，禁止靠符号猜测当前侧。

---

## 6. 恢复终端

### 6.1 站立终端

站立时，若 DCM 位于可用 CoP 域内，就可以选择 $u=b$ 使 DCM 不再发散。正式扩展终端为：

$$
\boxed{
\mathcal G_\sigma^{z,\mathrm{stand}}
=\{(b,q):b\in\mathcal C_\sigma,\ q\in\mathcal Q_\sigma^{\mathrm{stand}}\}
}.
$$

$q$ 的终端范围必须显式配置，不能在正式扩展状态模型中把 $q_N$ 留成无约束变量。

### 6.2 行走周期终端

行走模式使用 touchdown 截面上的左右周期点：

$$
\boxed{
\mathcal G_\sigma^{z,\mathrm{walk}}
=\left\{
z_\sigma^\star
\right\},
\qquad
z_\sigma^\star=
\begin{bmatrix}
b_\sigma^\star\\q_\sigma^\star
\end{bmatrix}
}.
$$

设名义 CoP 和名义落脚为 $u_L^\star,u_R^\star,\ell_L^\star,\ell_R^\star$，并定义：

$$
d_L=(A-1)u_L^\star+\ell_L^\star,
\qquad
d_R=(A-1)u_R^\star+\ell_R^\star.
$$

周期闭环满足：

$$
b_R^\star=Ab_L^\star-d_L,
\qquad
b_L^\star=Ab_R^\star-d_R.
$$

解得：

$$
\boxed{
b_L^\star=\frac{Ad_L+d_R}{A^2-1},
\qquad
b_R^\star=\frac{d_L+Ad_R}{A^2-1}
}.
$$

摆动脚周期状态为：

$$
\boxed{
q_L^\star=-\ell_R^\star,
\qquad
q_R^\star=-\ell_L^\star
}.
$$

代码可用很小的数值容差 $\varepsilon_{\mathrm{num}}$ 检查周期点，但该容差不能被解释为额外的物理稳定区域。

---

## 7. 1–5 步联合可恢复性认证

### 7.1 指定 $N$ 的联合 LP

给定当前状态 $z_0=(b_0,q_0)$、支撑侧 $\sigma_0$ 和各步持续时间，对任意 $N\ge1$ 创建变量：

$$
\{b_i,q_i\}_{i=0}^{N},
\qquad
\{u_i,\ell_i\}_{i=0}^{N-1}.
$$

$b_0,q_0$ 固定为测量值。对每个 $i=0,\ldots,N-1$ 加入：

$$
u_i\in\mathcal C_{\sigma_i},
\qquad
\ell_i\in\mathcal L_{\sigma_i},
$$

$$
-v_{\max}\odot T_i
\le\ell_i-q_i
\le v_{\max}\odot T_i,
$$

$$
b_{i+1}=A_ib_i-(A_i-1)u_i-\ell_i,
$$

$$
q_{i+1}=-\ell_i,
$$

以及终端约束：

$$
\boxed{z_N\in\mathcal G^z_{\sigma_N}}.
$$

可行性阶段目标函数取零。若全部集合和终端均为线性等式/不等式，该问题是小规模线性规划。

定义：

$$
\operatorname{Feasible}_N(z_0)=
\begin{cases}
\mathrm{true},&N\text{ 步联合 LP 可行},\\
\mathrm{false},&N\text{ 步联合 LP 不可行}.
\end{cases}
$$

### 7.2 $N=0$ 的含义

$N=0$ 不传播动力学，也不施加控制，只做当前终端成员检查：

$$
\boxed{
\operatorname{Feasible}_0(z_0)
\Longleftrightarrow
z_0\in\mathcal G^z_{\sigma_0}
}.
$$

行走中若摆动脚已经承诺必须完成当前触地，步中认证应从 $N=1$ 开始；行走的 $N=0$ 成功主要在 touchdown 截面检查。实现中建议把 `terminal_now` 与未来时域 LP 查询分开，避免混淆。

### 7.3 最少恢复触地数

按如下顺序查询：

$$
F_0\rightarrow F_1\rightarrow F_2\rightarrow F_3\rightarrow F_4\rightarrow F_5.
$$

第一个可行的时域就是：

$$
\boxed{
N_{\min}
=\min\{n\in\{0,1,2,3,4,5\}:\operatorname{Feasible}_n(z_0)\}
}.
$$

若 $F_0,\ldots,F_5$ 全部不可行，则返回 `OVER_HORIZON`，对外显示为 $N_{\min}>5$。

精确 $N$ 步可行集合不保证彼此嵌套，所以：

- 不得通过假设集合嵌套跳过中间时域；
- 不得分别手写五套求解器；
- 必须使用同一个通用构造器，仅改变 horizon；
- 在线模式可在找到第一个可行时停止；验证模式应支持求解全部时域并保存布尔结果。

### 7.4 求解状态不能混为一类

建议使用显式枚举：

```text
FINITE          # N_min 为 0..5
OVER_HORIZON    # 1..5 均被可靠求解为 infeasible
INVALID_INPUT   # NaN、Inf、单位或状态非法
SOLVER_FAILURE  # 超时、数值失败、残差检查失败
```

`SOLVER_FAILURE` 和 `INVALID_INPUT` 不能伪装成 $N_{\min}>5$，也不能因此惩罚 Actor。

### 7.5 见证计划与 Actor 的关系

LP 可返回完整见证：

$$
\{b_i,q_i,u_i,\ell_i,T_i,\sigma_i\}.
$$

见证必须经过独立残差检查。但在当前训练方案中：

$$
\boxed{\text{LP 见证只用于认证、诊断和奖励，不由 Actor 强制执行}}
$$

Actor 仍由原本的观测直接产生动作。

所有传入 reward buffer 的 $N_{\min}$、margin 和势函数值都应视为停止梯度的外部标量；当前方案不要求也不允许通过 LP 求解器反向传播。

---

## 8. 连续有符号恢复裕度

仅使用离散的 $N_{\min}$ 会产生大片平台。例如两个状态都可能满足 $N_{\min}>5$，但一个离 5 步可行边界很远，另一个已经非常接近边界。为给训练提供连续信号，需要计算有符号 LP 裕度。

### 8.1 指定时域的裕度

把 CoP、落脚、摆动速度和终端不等式写成：

$$
a_r^Tx\le b_r,
$$

并为每类约束给出正的归一化尺度 $s_r$。

若原 $N$ 步问题可行，最大化统一内缩量 $\rho\ge0$：

$$
a_r^Tx\le b_r-\rho s_r.
$$

若原问题不可行，最小化使问题恢复可行的统一放宽量 $\eta\ge0$：

$$
a_r^Tx\le b_r+\eta s_r.
$$

动力学等式和 $q_{i+1}=-\ell_i$ 必须保持精确。定义：

$$
\boxed{
m_N(z)=
\begin{cases}
+\rho_N^\star,&\operatorname{Feasible}_N(z),\\
-\eta_N^\star,&\operatorname{Feasible}_N(z)=\mathrm{false}.
\end{cases}
}.
$$

所以：

- $m_N>0$：该时域可行，且数值越大表示离约束边界越远；
- $m_N=0$：处于边界附近；
- $m_N<0$：该时域不可行，越接近零表示越接近可行边界。

### 8.2 在线选用哪个裕度

定义训练实际使用的标量裕度：

$$
\boxed{
m(z)=
\begin{cases}
m_0(z),&N_{\min}=0,\\
+\rho_{N_{\min}}^\star,&1\le N_{\min}\le5,\\
-\eta_5^\star,&N_{\min}>5.
\end{cases}
}.
$$

其中：

- 已找到 $N_{\min}$ 时，只计算最短可行时域的内缩裕度；
- 1–5 步都不可行时，只计算 5 步问题的最小放宽量；
- 精确点终端的 $N=0$ 可定义为

$$
m_0(z)=-\|D^{-1}(z-z_\sigma^\star)\|_\infty.
$$

物理尺度不同的约束不能在未经归一化时共同使用一个裕度变量。$\eta$、$\rho$ 和 $\beta$ 的数值尺度必须由同一套归一化定义决定。

---

## 9. “正常走一步后剩余步数应至少减一”的推导

假设当前状态 $z_j$ 的最少恢复触地数为：

$$
N_j=n,
\qquad 1\le n\le5.
$$

根据 $n$ 步可恢复性的定义，存在一条可行见证：

$$
\Pi_j=
\{(u_j,\ell_j),(u_{j+1},\ell_{j+1}),\ldots,(u_{j+n-1},\ell_{j+n-1})\},
$$

使机器人在 $n$ 次触地以后进入终端。

如果真实 Actor 的下一步状态转移实现了该可行见证的第一段，那么触地后的剩余序列：

$$
\Pi_{j+1}=
\{(u_{j+1},\ell_{j+1}),\ldots,(u_{j+n-1},\ell_{j+n-1})\}
$$

仍然是一条合法的 $(n-1)$ 步恢复方案。因此：

$$
\boxed{N_{j+1}\le N_j-1}.
$$

这不是说所有真实动作都必然满足等号，而是提供了一个物理基准：

- $N_{j+1}=N_j-1$：按理论尾部正常推进；
- $N_{j+1}<N_j-1$：真实动作取得了比原见证更快的恢复进展；
- $N_{j+1}=N_j$：没有跨过新的离散恢复等级，需要用裕度判断内部进展；
- $N_{j+1}>N_j$：恢复能力恶化；
- $N_{j+1}>5$：离开了 5 步可认证区域。

由于 Actor 并不执行 LP 见证，所以该不等式是奖励所鼓励的目标，不是仿真必须始终满足的动力学恒等式。

---

## 10. 恢复状态机

### 10.1 状态

```text
NORMAL   -> 正常行走或站立
PUSH     -> 正在施加有限时长外力；瞬时冲量可视为零时长 PUSH
RECOVERY -> 外力已经结束，开始计算恢复进展
```

`SUCCESS`、`TIMEOUT` 和 `FALL` 是一次 recovery trial 的终止结果，不应与 $N_{\min}$ 的枚举值混用。

### 10.2 启动

有限时长外力结束以后进入 `RECOVERY`；瞬时冲量则在冲量施加完成后立即进入。初始化：

$$
j=0,
$$

其中 $j$ 是扰动以后已经真实发生的新增 touchdown 数。立即计算：

$$
(N_0,m_0).
$$

这里下标 0 表示“恢复事件序列的初始评估”，不是说 $N_0=0$。初始评估不增加真实恢复步数。

### 10.3 touchdown 更新

每发生一次新的有效 touchdown：

$$
j\leftarrow j+1,
$$

然后用触地后的真实状态重新计算：

$$
(N_j,m_j).
$$

### 10.4 终止条件

满足任一条件即结束本次恢复阶段：

$$
\boxed{
\begin{aligned}
N_j=0 &\Rightarrow \mathrm{SUCCESS},\\
j\ge5\ \land\ N_j\ne0 &\Rightarrow \mathrm{TIMEOUT},\\
\mathrm{fall}=1 &\Rightarrow \mathrm{FALL}.
\end{aligned}
}
$$

推荐冲突优先级为：

```text
FALL > SUCCESS > TIMEOUT
```

也就是物理摔倒条件优先；未摔倒时先判断成功，再判断 5 次 touchdown 超时。

即使序列为：

$$
>5\rightarrow5\rightarrow4\rightarrow3\rightarrow2\rightarrow1,
$$

第 5 次真实 touchdown 后仍未达到 $N=0$，仍应判为 `TIMEOUT`。因为：

$$
\boxed{
N_j=\text{从当前状态预测还需要多少步},
\qquad
j=\text{扰动以后实际已经走了多少步}
}
$$

二者不能混淆。

---

## 11. 奖励函数总体结构

训练时保留原 locomotion reward，只在 `RECOVERY` 门控打开时增加恢复相关项：

$$
\boxed{
r_t
=r_{\mathrm{locomotion},t}
+g_{\mathrm{rec},t}
\left(
r_{\mathrm{prog},t}
+\mathbf1_{\mathrm{TD},t}\lambda_{\mathrm{TD}}r_{\mathrm{TD},j}
\right)
+r_{\mathrm{terminal},t}
}.
$$

其中：

- $g_{\mathrm{rec},t}\in\{0,1\}$：只有外力结束后的恢复阶段为 1；
- $r_{\mathrm{prog},t}$：policy-step 时间尺度上的周期性连续塑形；
- $r_{\mathrm{TD},j}$：touchdown 时间尺度上的实际步进质量奖励；
- $r_{\mathrm{terminal},t}$：成功、超时和摔倒的终端奖励。

这三个恢复奖励组件必须分别记录，并通过配置开关支持单独消融。

---

## 12. Policy-step 时间尺度：连续恢复势函数

### 12.1 离散恢复等级

定义粗粒度恢复等级：

$$
\boxed{
R(N)=
\begin{cases}
6-N,&N\in\{0,1,2,3,4,5\},\\
0,&N>5.
\end{cases}
}
$$

因此：

| $N$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $>5$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| $R(N)$ | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

### 12.2 恢复势函数

定义：

$$
\boxed{
\Phi_t
=R(N_{\min,t})
+\alpha
\tanh\left(\frac{m_t}{\beta}\right)
}.
$$

其中：

- $R(N)$ 提供跨恢复步数等级的粗粒度信号；
- $m$ 在同一等级内部提供连续信号；
- $\beta>0$ 负责裕度归一化；
- $\alpha>0$ 控制连续裕度相对离散等级的权重。

建议保守满足：

$$
0<\alpha<0.5,
$$

从而避免 margin 波动压过相邻 $N$ 等级之间的单位间隔。若使用其他 $R$ 间距，应相应重新推导该约束。

### 12.3 周期性塑形奖励

认证器不必每个 simulation step 都运行。设两次认证相隔 $K$ 个 policy steps，则：

$$
\boxed{
r_{\mathrm{prog},t}
=\lambda_{\mathrm{prog}}
\left[
\gamma^K\Phi_{t+K}-\Phi_t
\right]
}.
$$

若实际间隔可变，则必须使用真实的 $\Delta k$：

$$
r_{\mathrm{prog}}
=\lambda_{\mathrm{prog}}
\left[
\gamma^{\Delta k}\Phi_{\mathrm{new}}-
\Phi_{\mathrm{old}}
\right].
$$

两次认证之间该项为零。终止或 reset 时必须清空缓存，绝不能跨 trial 或跨环境计算势函数差。

当 $N_{\min}>5$ 时，$R(N)=0$，但：

$$
m=-\eta_5
$$

仍能形成：

$$
-1.2\rightarrow-0.6\rightarrow-0.1,
$$

从而给 Actor 提供“正在接近 5 步可恢复边界”的正向信号。这是避免训练初期奖励完全稀疏的关键。

---

## 13. Touchdown 时间尺度：剩余恢复步数奖励

### 13.1 两套缓存必须分开

训练中存在两个时钟：

1. `dense_prev`：上一次周期性认证结果，用于 $r_{\mathrm{prog}}$；
2. `td_prev`：恢复开始或上一次真实 touchdown 的认证结果，用于 $r_{\mathrm{TD}}$。

周期性认证不能覆盖 `td_prev`。否则 touchdown 奖励比较的将是“最近一个 policy-step 状态”，而不是“真实走完一步前后的状态”。

### 13.2 同等级 margin 奖励

定义：

$$
\Delta m_j=m_{j+1}-m_j.
$$

为满足“有明显改善才奖励，真正停滞要惩罚”，定义：

$$
\boxed{
\mathcal M(\Delta m)=
\begin{cases}
\lambda_m\tanh(\Delta m/\beta_m),
&\Delta m>\varepsilon_m,\\[1mm]
-r_{\mathrm{stall}},
&|\Delta m|\le\varepsilon_m,\\[1mm]
\lambda_m\tanh(\Delta m/\beta_m)-r_{\mathrm{stall}},
&\Delta m<-\varepsilon_m.
\end{cases}
}
$$

$\varepsilon_m$ 应依据 LP 数值容差和无扰动 baseline 的裕度抖动标定，不能随意设为零。

### 13.3 有限步数到有限步数

若：

$$
N_j,N_{j+1}\in\{0,1,2,3,4,5\},
$$

且恢复尚未在 $j$ 时结束，定义：

$$
d_j=N_j-N_{j+1}.
$$

touchdown 转移奖励为：

$$
\boxed{
r_{\mathrm{TD},j}^{\mathrm{trans}}
=
\begin{cases}
r_0+\lambda_+(d_j-1),&d_j>1,\\
r_0,&d_j=1,\\
\mathcal M(\Delta m_j),&d_j=0,\\
-\lambda_-|d_j|,&d_j<0.
\end{cases}
}
$$

其含义为：

- 比预计多减少一步或更多：较大正奖励；
- 正好减少一步：小正奖励 $r_0$；
- $N$ 不变：由 margin 判断内部改善；若 margin 也停滞，则惩罚；
- $N$ 增大：按恶化的等级数惩罚。

### 13.4 与 $N>5$ 相关的转移

`OVER_HORIZON` 不能在算术中直接当成整数 6。相关转移单独处理：

$$
\boxed{
r_{\mathrm{TD},j}^{\mathrm{trans}}
=
\begin{cases}
\mathcal M(\Delta m_j),
&>5\rightarrow>5,\\[1mm]
r_{\mathrm{enter5}}+\lambda_{\mathrm{enter}}(5-N_{j+1}),
&>5\rightarrow N_{j+1}\le5,\\[1mm]
-r_{\mathrm{leave5}},
&N_j\le5\rightarrow>5.
\end{cases}
}
$$

其中：

- $>5\rightarrow>5$：只根据 5 步放松裕度是否改善判断；
- $>5\rightarrow5$：获得进入 5 步可恢复区域的 bonus；
- $>5\rightarrow3$ 等更大的跨越，在进入 bonus 上再按最终步数加奖励；
- 有限步数退化为 $>5$：施加明显惩罚。

### 13.5 成功 bonus

若本次 touchdown 后：

$$
N_{j+1}=0,
$$

在转移奖励之外增加：

$$
\boxed{r_{\mathrm{success}}>0}.
$$

所以例如 $1\rightarrow0$ 得到：

$$
r_0+r_{\mathrm{success}}.
$$

而 $3\rightarrow0$ 得到更大的跨级奖励和 success bonus。

---

## 14. 终端奖励与事件优先级

定义：

$$
\boxed{
r_{\mathrm{terminal},t}
=
\begin{cases}
-r_{\mathrm{fall}},&\mathrm{FALL},\\
+r_{\mathrm{success}},&\mathrm{SUCCESS},\\
-r_{\mathrm{timeout}},&\mathrm{TIMEOUT},\\
0,&\text{otherwise}.
\end{cases}
}
$$

要求：

$$
r_{\mathrm{fall}}>r_{\mathrm{timeout}}>0.
$$

实现规则：

- 若同一仿真步触发 `FALL`，抑制同一时刻可能产生的正 success / touchdown bonus；
- `SUCCESS` 时可以保留最后一步的正常或超额进展奖励，并额外加 success bonus；
- `TIMEOUT` 时可以保留第 5 步的转移奖励，但 timeout penalty 应足以表达“虽在改善，实际恢复仍然过慢”；
- solver failure 不得自动产生 fall 或 timeout penalty，应单独标记并根据实验协议决定丢弃、重试或终止为 invalid trial。

建议只在一个位置添加 success bonus，避免在 `r_TD` 和 `r_terminal` 中重复累加。本文公式把它归入 `r_terminal`。

---

## 15. 奖励序列示例

### 15.1 正常且逐渐加快

$$
5\rightarrow4\rightarrow2\rightarrow1\rightarrow0.
$$

对应解释：

| 转移 | 解释 | 事件奖励 |
|---|---|---|
| $5\to4$ | 正常减少一步 | $r_0$ |
| $4\to2$ | 比正常多减少一步 | $r_0+\lambda_+$ |
| $2\to1$ | 正常减少一步 | $r_0$ |
| $1\to0$ | 正常完成恢复 | $r_0+r_{\mathrm{success}}$ |

### 15.2 离散等级不变但内部改善

$$
(N,m):(4,0.05)\rightarrow(4,0.40).
$$

若 $0.35>\varepsilon_m$：

$$
r_{\mathrm{TD}}=\lambda_m\tanh(0.35/\beta_m)>0.
$$

因此不会错误惩罚“尚未跨级但已经明显接近下一等级”的动作。

### 15.3 真正停滞

$$
(4,0.20)\rightarrow(4,0.20).
$$

此时：

$$
r_{\mathrm{TD}}=-r_{\mathrm{stall}}<0.
$$

### 15.4 仍在 5 步范围外但持续改善

$$
(>5,-1.2)\rightarrow(>5,-0.5)\rightarrow(>5,-0.1).
$$

虽然离散标签始终是 $>5$，但 $\Delta m>0$，连续 shaping 和 touchdown margin 奖励均可提供正信号。

### 15.5 达到实际 5 步上限

$$
>5\rightarrow5\rightarrow4\rightarrow3\rightarrow2\rightarrow1.
$$

预测能力一直改善，但第 5 次 touchdown 后仍未达到 $N=0$，所以结果为：

$$
\boxed{\mathrm{TIMEOUT}}.
$$

第 5 步的进展奖励可以保留，但同时施加 $-r_{\mathrm{timeout}}$。

---

## 16. 权重与数值稳定性要求

所有权重必须在配置文件中给出，不能散落硬编码。至少包括：

```yaml
recoverability:
  n_max: 5
  eval_interval_policy_steps: null
  terminal_mode: null             # stand 或 walk
  solver_tolerance: null
  plan_residual_tolerance: null
  margin_constraint_scales: null

recovery_state_machine:
  max_actual_touchdowns: 5
  start_after_push_end: true
  fall_priority: true

recovery_reward:
  enable_dense_progress: true
  enable_touchdown_progress: true
  lambda_progress: null
  alpha_margin_in_potential: null
  beta_potential_margin: null
  lambda_touchdown: null
  normal_progress_reward_r0: null
  lambda_extra_step: null
  lambda_worse_step: null
  lambda_margin: null
  beta_touchdown_margin: null
  margin_deadband_epsilon: null
  stall_penalty: null
  enter_five_step_bonus: null
  lambda_enter_depth: null
  leave_five_step_penalty: null
  success_bonus: null
  timeout_penalty: null
  fall_penalty: null
  component_clip: null
```

参数必须满足基本符号关系：

$$
r_0,\lambda_+,\lambda_-,\lambda_m,\beta,\beta_m,
r_{\mathrm{stall}},r_{\mathrm{enter5}},r_{\mathrm{leave5}},
r_{\mathrm{success}},r_{\mathrm{timeout}},r_{\mathrm{fall}}>0.
$$

并建议：

$$
r_{\mathrm{fall}}>r_{\mathrm{timeout}},
\qquad
0<\alpha<0.5.
$$

不要在未知原 locomotion reward 尺度时直接固定一套最终数值。首次调参应记录各奖励分量的均值、标准差、P95 和最大绝对值，确认恢复奖励不会完全淹没 tracking reward，也不会小到没有作用。

---

## 17. 推荐代码接口

### 17.1 认证结果

```python
class CertificateStatus(Enum):
    FINITE = "finite"
    OVER_HORIZON = "over_horizon"
    INVALID_INPUT = "invalid_input"
    SOLVER_FAILURE = "solver_failure"


@dataclass
class RecoverabilityCertificate:
    status: CertificateStatus
    terminal_now: bool
    minimum_touchdowns: int | None       # 仅 FINITE 时为 0..5
    feasible_by_horizon: dict[int, bool]
    selected_margin: float | None
    margin_horizon: int | None           # 0、N_min 或 5
    selected_plan: HorizonPlan | None
    failure_category: str | None
    solver_diagnostics: dict
```

### 17.2 认证器

```python
def certify_recoverability(
    z0,
    support_side,
    durations,
    target,
    n_max=5,
    committed_swing=True,
    compute_margin=True,
) -> RecoverabilityCertificate:
    ...
```

### 17.3 奖励状态

```python
@dataclass
class RecoveryRewardState:
    active: bool
    actual_touchdowns: int
    dense_prev_certificate: RecoverabilityCertificate | None
    dense_prev_policy_step: int | None
    td_prev_certificate: RecoverabilityCertificate | None
    outcome: str | None                  # success / timeout / fall / invalid
```

---

## 18. 状态机与奖励伪代码

```python
def on_push_end(env):
    state.active = True
    state.actual_touchdowns = 0
    state.outcome = None

    if physical_fall(env):
        queue_terminal_reward(-fall_penalty)
        finish_recovery("fall")
        return

    cert = certify_current_state(env)
    if cert.status in {INVALID_INPUT, SOLVER_FAILURE}:
        mark_invalid_trial(cert)
        finish_recovery("invalid")
        return

    state.dense_prev_certificate = cert
    state.dense_prev_policy_step = env.policy_step
    state.td_prev_certificate = cert

    if cert.terminal_now:
        queue_terminal_reward(+success_bonus)
        finish_recovery("success")


def on_policy_step(env):
    reward = locomotion_reward(env)

    if not state.active:
        return reward

    if physical_fall(env):
        reward -= fall_penalty
        finish_recovery("fall")
        return reward

    if certificate_evaluation_is_due(env):
        cert = certify_current_state(env)
        if cert.status in {FINITE, OVER_HORIZON}:
            delta_k = env.policy_step - state.dense_prev_policy_step
            reward += dense_progress_reward(
                state.dense_prev_certificate,
                cert,
                delta_k,
            )
            state.dense_prev_certificate = cert
            state.dense_prev_policy_step = env.policy_step

            # 站立且没有必须完成的摆动触地时，允许在认证时刻结束。
            # 行走周期终端仍只在 touchdown 截面判定。
            if stand_terminal_can_end_now(env, cert):
                reward += success_bonus
                finish_recovery("success")
                return reward
        else:
            log_invalid_certificate(cert)

    if valid_new_touchdown(env):
        state.actual_touchdowns += 1

        # FALL 优先，不能先发放可能为正的 touchdown 奖励。
        if physical_fall(env):
            reward -= fall_penalty
            finish_recovery("fall")
            return reward

        td_cert = certify_post_touchdown_state(env)

        if td_cert.status in {INVALID_INPUT, SOLVER_FAILURE}:
            mark_invalid_trial(td_cert)
            finish_recovery("invalid")
            return reward

        reward += touchdown_transition_reward(
            state.td_prev_certificate,
            td_cert,
        )
        state.td_prev_certificate = td_cert

        if td_cert.terminal_now or td_cert.minimum_touchdowns == 0:
            reward += success_bonus
            finish_recovery("success")
        elif state.actual_touchdowns >= 5:
            reward -= timeout_penalty
            finish_recovery("timeout")

    return reward
```

实现时需要保证同一个 touchdown 不会被接触抖动重复计数，应使用接触迟滞和事件边沿检测。

若周期性认证恰好与 touchdown 落在同一个 policy step，可复用同一次触地后认证结果以减少求解次数，但仍必须分别更新 `dense_prev_certificate` 和 `td_prev_certificate`，并按固定事件顺序计算奖励。

---

## 19. 部署流程

部署时不计算任何 reward，Actor 控制链保持：

$$
o_t\rightarrow\pi_\theta\rightarrow a_t.
$$

每次有效 touchdown 后，旁路执行：

$$
(c,\dot c,p_\sigma,p_{\mathrm{sw}})
\rightarrow(b,q)
\rightarrow F_0,F_1,\ldots,F_5
\rightarrow N_{\min}.
$$

touchdown 截面上第一预测步使用完整 $T$。认证结果默认只用于：

- 恢复能力监视；
- 日志与实验评价；
- 后续安全 supervisor 的研究输入。

第一版不得把 $N_{\min}$、margin 或 LP 见证悄悄加入 Actor observation，也不得用 LP 见证替换 Actor 动作。

---

## 20. 必须记录的日志

每次 recovery trial 至少记录：

```text
trial_id
recovery_state
push_start_time
push_end_time
push_direction
push_magnitude_or_impulse
support_side
phase
remaining_time
b
q
certificate_status
terminal_now
N_min
feasible_by_horizon
selected_margin
margin_horizon
phi_potential
dense_progress_reward
touchdown_transition_reward
terminal_reward
actual_touchdown_count
outcome
solver_status_by_horizon
solver_time_by_horizon
constraint_residuals
failure_category
```

同时按 rollout 汇总：

- $N_{\min}=0,1,\ldots,5,>5$ 的比例；
- margin 的均值、标准差和分位数；
- $\Delta m$ 的分布；
- `SUCCESS`、`TIMEOUT`、`FALL`、invalid trial 比例；
- 实际恢复 touchdown 数分布；
- 三个恢复奖励组件与原 locomotion reward 的量级对比；
- 按扰动方向、强度和步态相位分层的结果。

---

## 21. 必须通过的测试

### 21.1 理论与求解器

1. DCM 解析传播与数值积分一致；
2. 触地映射等于“先连续传播，再切换支撑坐标”；
3. 每次理想换脚均满足 $q_{k+1}=-\ell_k$；
4. 左右周期点经过两个名义触地映射后闭环；
5. 固定 $q$ 的一步 LP 与解析一步切片一致；
6. $N=1,\ldots,5$ 的所有可行计划经独立前向重放后满足动力学、CoP、落脚、摆动速度和终端约束；
7. 求解顺序确实返回第一个可行 horizon；
8. 代码不依赖“精确 $N$ 步集合嵌套”的错误假设；
9. `OVER_HORIZON`、`INVALID_INPUT` 和 `SOLVER_FAILURE` 可被明确区分；
10. 裕度在可行、边界和不可行样本上的符号正确。

### 21.2 状态机

1. 扰动结束时 $j=0$，初始认证不增加 touchdown 数；
2. 接触抖动不会重复计数；
3. $N=0$ 立即成功；
4. 第 5 次 touchdown 后 $N\ne0$ 必须 timeout；
5. $>5\to5\to4\to3\to2\to1$ 在第 5 步仍是 timeout；
6. 任何时刻的真实 fall 都与 $N>5$ 分开，并按更高优先级终止；
7. reset 后所有 dense / touchdown 奖励缓存均被清空。

### 21.3 奖励单调性

在其他量相同时必须满足：

$$
r(4\to2)>r(4\to3)>0,
$$

$$
r(4\to4,\Delta m>\varepsilon_m)>0,
$$

$$
r(4\to4,|\Delta m|\le\varepsilon_m)<0,
$$

$$
r(4\to5)<0,
$$

$$
r(4\to>5)<0,
$$

$$
r(>5\to>5,\Delta m>\varepsilon_m)>0,
$$

$$
r(>5\to5)>0.
$$

还必须测试：

- 周期性认证不会覆盖 touchdown 基准缓存；
- solver failure 不产生对 Actor 的伪惩罚；
- success bonus 只添加一次；
- fall 与 success 同时出现时，fall 优先且正 bonus 被抑制；
- 不同认证间隔使用正确的 $\gamma^{\Delta k}$。

---

## 22. 实现顺序

编辑器 AI 应严格按以下顺序实现：

1. 统一 heading frame、左右脚、单位和事件时刻定义；
2. 实现并测试 DCM 连续传播；
3. 实现扩展状态触地映射；
4. 计算并验证站立/行走终端；
5. 实现一个支持任意有限 horizon 的联合 LP 构造器；
6. 验证 $N=1$ 后再扩展并测试到 $N=5$；
7. 实现顺序搜索和明确的求解状态枚举；
8. 实现有符号 margin，并验证尺度和符号；
9. 在无扰动 baseline gait 上验证认证器与实际步态匹配；
10. 实现 `NORMAL -> PUSH -> RECOVERY` 状态机；
11. 实现连续势函数奖励；
12. 实现 touchdown 转移奖励和独立缓存；
13. 实现 success / timeout / fall 终端逻辑；
14. 加入完整日志、消融开关和自动化测试；
15. 最后才开始调奖励权重或重新训练。

若无扰动正常行走状态仍大量返回 $N_{\min}>5$，应先检查 $T,h,\mathcal C,\mathcal L,v_{\max},z_\sigma^\star$、左右脚符号、heading frame 和第一步剩余时间，不得优先通过增大奖励权重掩盖认证器失配。

---

## 23. 禁止事项

- 不得把 $>5$ 直接当成整数 6 参与步数差运算；
- 不得把 solver failure 当成不可恢复；
- 不得把 $N_j$ 与实际已走步数 $j$ 混淆；
- 不得把 policy step 与 touchdown 混称为“步”；
- 不得遗漏 $q_{i+1}=-\ell_i$；
- 不得把摆动约束写成 $|\ell_i+q_i|\le v_{\max}T_i$；
- 不得只给第一预测步加摆动速度约束；
- 不得把步中第一步错误地使用完整 $T$；
- 不得让连续认证更新 touchdown 奖励缓存；
- 不得重复添加 success bonus；
- 不得在部署时默认让 Actor 依赖认证器输出；
- 不得在认证器尚未通过无扰动标定和残差测试前开始调 reward。

---

## 24. 最终算法摘要

理论认证器：

$$
\boxed{
z=[b,q]
\xrightarrow[
u_i\in\mathcal C_{\sigma_i},\
\ell_i\in\mathcal L_{\sigma_i},\
|\ell_i-q_i|\le v_{\max}T_i
]{
b_{i+1}=A_ib_i-(A_i-1)u_i-\ell_i,\
q_{i+1}=-\ell_i
}
z_N\in\mathcal G^z_{\sigma_N}
}
$$

依次求解：

$$
\boxed{F_0\rightarrow F_1\rightarrow F_2\rightarrow F_3\rightarrow F_4\rightarrow F_5}
$$

得到 $N_{\min}$，并以最短可行时域内缩裕度或 5 步最小放松量得到 $m$。

训练奖励：

$$
\boxed{
r_t
=r_{\mathrm{locomotion},t}
+g_{\mathrm{rec},t}
\left[
\lambda_{\mathrm{prog}}
(\gamma^{\Delta k}\Phi_{\mathrm{new}}-\Phi_{\mathrm{old}})
+\mathbf1_{\mathrm{TD},t}\lambda_{\mathrm{TD}}r_{\mathrm{TD},j}
\right]
+r_{\mathrm{terminal},t}
}
$$

其中：

$$
\boxed{
\Phi=R(N_{\min})+\alpha\tanh(m/\beta)
}
$$

状态机：

$$
\boxed{
\begin{cases}
N_j=0 &\Rightarrow \mathrm{SUCCESS},\\
j\ge5,\ N_j\ne0 &\Rightarrow \mathrm{TIMEOUT},\\
\mathrm{fall}=1 &\Rightarrow \mathrm{FALL}.
\end{cases}
}
$$

这套设计的核心含义是：理论认证器给出“从当前状态预计还需要多少次触地”，Actor 每完成一次真实触地，就检查剩余步数是否至少按理论尾部减少一；离散等级没有改变时，再用连续裕度判断动作是在改善、停滞还是恶化。
