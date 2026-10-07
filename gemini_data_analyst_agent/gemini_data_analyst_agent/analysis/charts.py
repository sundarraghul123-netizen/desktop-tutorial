import plotly.express as px

def make_chart(df,kind,x=None,y=None,title='Chart'):
    if kind=='line':return px.line(df,x=x,y=y,title=title,markers=True)
    if kind=='bar':return px.bar(df,x=x,y=y,title=title)
    if kind=='scatter':return px.scatter(df,x=x,y=y,title=title)
    if kind=='histogram':return px.histogram(df,x=x,title=title)
    if kind=='box':return px.box(df,y=y,title=title)
    return px.bar(df,x=x,y=y,title=title)
