import PySimpleGUI as sg
import trayectoryDraw
import pnn
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def initUI():
    l1 = [
        [sg.Text("Select file", key="file")],
        [sg.FileBrowse(key="-IN-"),sg.Button("Draw")],
        [sg.Button("Identify"), sg.Button("Manual")]
    ]
    l2 = [
        [sg.Canvas(background_color='white', size=(500,500), key="plt")],
        [sg.Text("NA", key="Num")]
    ]

    layout = [
        [sg.Column(l1),
        sg.VerticalSeparator(),
        sg.Column(l2)]
    ]

    return sg.Window(title="App", layout=layout, margins=(100, 50))

matplotlib.use("TkAgg")

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=0)
    return figure_canvas_agg

def manageUI(w, model):
    d = False
    while True:
        event, values = w.read()
        if event == "Draw":
            print(values["-IN-"])
            fig, ax = trayectoryDraw.draw(values["-IN-"])
            if d:
                try:
                    f_agg.get_tk_widget().destroy()
                except:
                    pass
            f_agg = draw_figure(w["plt"].TKCanvas, fig)
            d = True
        if event == "Manual":
            print("Manual")
            sg.popup("Browse: Elegir fichero.\nDraw: Muestra el carácter en pantalla.\nIdentify: Clasifica el carácter.\nManual: Abre esta ventana.", keep_on_top=True, title="Manual")
        if event == "Identify":
            w["Num"].update(pnn.predict(model, values["-IN-"]))

        if event == sg.WIN_CLOSED:
            break
    w.close()