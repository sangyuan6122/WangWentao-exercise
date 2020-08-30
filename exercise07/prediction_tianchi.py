import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 数据加载
#filename='sample_train_user.csv'
filename='tianchi_fresh_comp_train_user.csv'
user=pd.read_csv(filename)
#切10W行调试，正式跑用完整数据
#user.iloc[0:100000].to_csv('sample_train_user.csv',index=False)

user.drop_duplicates(keep='last',inplace=True)

user['time']=pd.to_datetime(user['time'])

user['dates'] = user.time.dt.date
user['month'] = user.dates.values.astype('datetime64[M]')
user['hours'] = user.time.dt.hour

user['behavior_type']=user['behavior_type'].apply(str)
user['user_id']=user['user_id'].apply(str)
user['item_id']=user['item_id'].apply(str)

pv_day=user[user.behavior_type=="1"].groupby("dates")["behavior_type"].count()
uv_day=user[user.behavior_type=="1"].drop_duplicates(["user_id","dates"]).groupby("dates")["user_id"].count()

import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.charts import Grid
import numpy as np

# 做出每天的pv与uv趋势图
attr=list(pv_day.index)
print(attr)
pv=(
    Line(init_opts=opts.InitOpts(width="1000px",height="500px"))
    .add_xaxis(xaxis_data=attr)
    .add_yaxis(
        "pv",
        np.around(pv_day.values/10000,decimals=2),
        label_opts=opts.LabelOpts(is_show=False)
    )
    .add_yaxis(
        series_name="uv",
        yaxis_index=1,
        y_axis=np.around(uv_day.values/10000,decimals=2),
        label_opts=opts.LabelOpts(is_show=False),
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="uv",
            type_="value",
            min_=0,
            max_=1.6,
            interval=0.4,
            axislabel_opts=opts.LabelOpts(formatter="{value} 万人"),
        )
    )
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            is_show=True,trigger="axis",axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axispointer_opts=opts.AxisPointerOpts(is_show=True,type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="pv",
            type_="value",
            min_=0,
            max_=100,
            interval=20,
            axislabel_opts=opts.LabelOpts(formatter="{value} 万次"),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        title_opts=opts.TitleOpts(title="pv与uv趋势图"),
    )
)
pv.render('pv与uv趋势图.html')
#每日pv、uv差异分析
pv_uv = pd.concat([pv_day, uv_day], join='outer', axis=1)
pv_uv.columns = ['pv_day', 'uv_day']

new_day=pv_uv.diff()
new_day.columns=['new_pv','new_uv']

attr = new_day.index
v = new_day.new_uv
w = new_day.new_pv
li=(
    Line(init_opts=opts.InitOpts(width="1000px",height="500px"))
    .add_xaxis(xaxis_data=attr)
    .add_yaxis(
        "新增pv",
        w,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .extend_axis(
        yaxis=opts.AxisOpts(
            name="新增uv",
            type_="value",
            min_=-2000,
            max_=1600,
            interval=400,
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        )
    )
     .set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            is_show=True, trigger="axis", axis_pointer_type="cross"
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name="新增pv",
            type_="value",
            min_=-350000,
            max_=250000,
            interval=100000,
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
         title_opts=opts.TitleOpts(title="pv、uv差异分析"),
    )
)
il=(
    Line()
    .add_xaxis(xaxis_data=attr)
    .add_yaxis("新增uv",v,yaxis_index='1',label_opts=opts.LabelOpts(is_show=False),)
)
c=li.overlap(il)
c.render('pv、uv差异分析.html')

#不同时期用户行为分析
shopping_cart= user[user.behavior_type == '3'].groupby('dates')['behavior_type'].count()
collect=user[user.behavior_type=='2'].groupby('dates')['behavior_type'].count()
buy=user[user.behavior_type=='4'].groupby('dates')['behavior_type'].count()

attr_a=list(shopping_cart.index)
v_1=shopping_cart.values.tolist()
v_2=collect.values.tolist()
v_3=buy.values.tolist()

b=(
    Line()
    .add_xaxis(xaxis_data=attr_a)
    .add_yaxis(
        "加购人数",
        v_1,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .add_yaxis(
        "收藏人数",
        v_2,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .add_yaxis(
        "购买人数",
        v_3,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .set_global_opts(title_opts=opts.TitleOpts(title="不同时期用户行为数据"))
)
b.render('不同时期用户行为分析.html')

#把数据拆分为活动数据和日常数据做不同时段的分析
user['dates']=pd.to_datetime(user['dates'])
active=user[user["dates"].isin(["2014/12/11","2014/12/12","2014/12/13"])]
daily=user[~user["dates"].isin(["2014/12/11","2014/12/12","2014/12/13"])]

from pyecharts.charts import Bar
# 活动数据
cart_h= active[active.behavior_type == '3'].groupby('hours')['behavior_type'].count()
collect_h=active[active.behavior_type=='2'].groupby('hours')['behavior_type'].count()
buy_h=active[active.behavior_type=='4'].groupby('hours')['behavior_type'].count()
uv_h=active[active.behavior_type== '1'].groupby('hours')['user_id'].count()

attr_h=list(cart_h.index)
h1=np.around(cart_h.values/3,decimals=0).tolist()
h2=np.around(collect_h.values/3,decimals=0).tolist()
h3=np.around(buy_h.values/3,decimals=0).tolist()
h4=np.around(uv_h.values/3,decimals=0).tolist()


h=(
    Line(init_opts=opts.InitOpts(width="1000px",height="500px"))
    .add_xaxis(xaxis_data=attr_h)
    .add_yaxis(
        "加购人数",
        h1,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .add_yaxis(
        "收藏人数",
        h2,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .add_yaxis(
        "购买人数",
        h3,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15)),
        title_opts=opts.TitleOpts(title="日均各时段活动用户行为",pos_top="48%"),
        legend_opts=opts.LegendOpts(pos_top="48%"),
    )
)
bar=(
    Bar()
    .add_xaxis(xaxis_data=attr_h)
    .add_yaxis(
    "浏览人数",
        h4,
        label_opts=opts.LabelOpts(is_show=False)
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="活动pv对比数据"),
    )
)

ggrid = (
    Grid()
    .add(bar, grid_opts=opts.GridOpts(pos_bottom="60%"))
    .add(h, grid_opts=opts.GridOpts(pos_top="60%"))
)
ggrid.render('活动期间不同时段的用户行为分析.html')