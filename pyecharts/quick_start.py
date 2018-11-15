# coding=utf-8
from __future__ import unicode_literals

from pyecharts import Bar, Line, Pie
import fire
from pyecharts.engine import create_default_environment


def my_first_echart():
    bar = Bar('MyFirstEcharts', 'This is the subTitle')
    bar.use_theme("dark")
    bar.add("服装",
            ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"],
            [5, 20, 36, 10, 75, 90],
            is_more_utils=True)
    # bar.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
    bar.render('./render_file/first_echarts.html')  # 生成本地 HTML 文件


def draw_echart_chain_call():
    CLOTHES = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
    clothes_v1 = [5, 20, 36, 10, 75, 90]
    clothes_v2 = [10, 25, 8, 60, 20, 80]
    (Bar('Example')
     .add("A", CLOTHES, clothes_v1, is_stack=True)
     .add('B', CLOTHES, clothes_v2, is_stack=True, mark_point=["max", "min"], mark_line=["average"])
     .render('./render_file/echarts_chain_call'))


def mutiply_echarts():
    bar = Bar("我的第一个图表", "这里是副标题")
    bar.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90])

    line = Line("我的第一个图表", "这里是副标题")
    line.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"],
             [5, 20, 36, 10, 75, 90])

    env = create_default_environment("html")
    # 为渲染创建一个默认配置环境
    # create_default_environment(filet_ype)
    # file_type: 'html', 'svg', 'png', 'jpeg', 'gif' or 'pdf'

    env.render_chart_to_file(bar, path='./render_file/bar.html')
    env.render_chart_to_file(line, path='./render_file/line.html')


def echarts_pie():
    CLOTHES = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
    clothes_v1 = [5, 20, 36, 10, 75, 90]
    clothes_v2 = [10, 25, 8, 60, 20, 80]
    pie = Pie("Pie", title_pos='center', width=900)
    pie.add("A", CLOTHES, clothes_v1, center=[25, 50], is_random=True, radius=[30, 75], rosetype="radius")
    pie.add("A", CLOTHES, clothes_v2, center=[
        75, 50], is_random=True, radius=[30, 75], rosetype="area", is_legend_show=False, is_label_show=True)


def main():
    fire.Fire()
    # we can try to use: fire.Fire(Example) to see the diff in CLI


if __name__ == '__main__':
    main()
