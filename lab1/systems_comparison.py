import plotly.graph_objects as go
import pandas as pd


data = {
    "x": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "linux_common": [0.04596,0.168387,0.373702,0.700039,1.09525,1.57321,2.13973,2.79138,3.53065,4.35455],
    "linux_cache": [0.040879,0.162294,0.362208,0.694059,1.06262,1.52766,2.07866,2.7144,3.43424,4.23557],
    "windows_common": [0.177265,0.694892,1.56996,2.77991,4.94824,7.12545,9.73914,12.6925,16.1477,21.0986],
    "windows_cache": [0.172696,0.682741,1.54783,3.12034,4.85525,7.01034,9.52158,12.4528,16.4763,20.8175],
    "arm_common": [0.096983,0.388256,0.897815,1.56563,2.51223,3.70118,5.33141,6.92089,9.23911,11.1012],
    "arm_cache": [0.097468, 0.386142, 0.873021, 1.54822, 2.41863, 3.48932, 4.7749, 6.20014, 7.85957, 9.69097]
}

df = pd.DataFrame(data)


SYSTEM_STYLES = {
    "linux": {
        "color": "#4C78A8",
        "symbol": "circle",
        "name": "Linux"
    },
    "windows": {
        "color": "#E45756",
        "symbol": "square",
        "name": "Windows"
    },
    "arm": {
        "color": "#54A24B",
        "symbol": "diamond",
        "name": "ARM"
    }
}


fig = go.Figure()


for system in SYSTEM_STYLES:
    style = SYSTEM_STYLES[system]
    
   
    fig.add_trace(go.Scatter(
        x = df["x"],
        y = df[f"{system}_common"],
        name = f"{style['name']} - 常规",
        mode = "lines+markers",
        line = dict(color=style['color'], width=3),
        marker = dict(
            symbol=style['symbol'],
            size=10,
            line=dict(width=1.5, color="white")
        )
    ))
    
    fig.add_trace(go.Scatter(
        x = df["x"],
        y = df[f"{system}_cache"],
        name = f"{style['name']} - 缓存优化",
        mode = "lines+markers",
        line = dict(color=style['color'], width=3, dash="dot"),
        marker = dict(
            symbol=style['symbol'],
            size=10,
            line=dict(width=1.5, color="white")
        )
    ))


fig.update_layout(
    title=dict(
        text="系统性能对比分析",
        font=dict(size=24, family="SimHei")
    ), 
    xaxis=dict(
        title="请求大小（单位）",
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True
    ),
    yaxis=dict(
        title="执行时间（秒）",
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True
    ),
    legend=dict(
        title="配置类型",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor="white",
    width=1200,
    height=800,
    margin=dict(l=100, r=100, t=100, b=100),
    hovermode="x unified"
)  


fig.show()