from functions.visualize import *
from functions.extract_stanford import *
from functions.create_model import *
from functions.test_model import *

from consolemenu import *
from consolemenu.items import *

menu = ConsoleMenu("TA72", "Menu")
menu.append_item(FunctionItem("Visualize datas", visualize))
menu.append_item(FunctionItem("[GPU] Extract features from Stanford dataset", extract_stanford, [True]))
menu.append_item(FunctionItem("[CPU] Extract features from Stanford dataset", extract_stanford, [False]))
menu.append_item(FunctionItem("[GPU] Create and save model", create_model, [True]))
menu.append_item(FunctionItem("[CPU] Create and save model", create_model, [False]))
menu.append_item(FunctionItem("Display test sample", test_models))
menu.show()
