import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=fillde]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

#DOT言語に変換
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))#y is weakref
    return txt

#計算グラフの可視化
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)
        
        add_func(output.creator)
        txt += _dot_var(output, verpose)

        while funcs:
            func = funcs.pop()
            txt += _dot_func(func)
            for x in func.inputs:
                txt += _dot_var(x, verbose)

                if x.creator is not None:
                    add_func(x.creator)
        return 'digraph g {\n' + txt + '}' 

#画像かまでを含めた変換
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir): #dezeroディレクトリがなかったら作成
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:] #拡張子png,pdf etc
    cmd = 'dot {} -T {} o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)   